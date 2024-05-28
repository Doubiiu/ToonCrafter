import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from einops import rearrange

import torch
import torchvision
from torch import Tensor
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_tensor


def frames_to_mp4(frame_dir,output_path,fps):
    def read_first_n_frames(d: os.PathLike, num_frames: int):
        if num_frames:
            images = [Image.open(os.path.join(d, f)) for f in sorted(os.listdir(d))[:num_frames]]
        else:
            images = [Image.open(os.path.join(d, f)) for f in sorted(os.listdir(d))]
        images = [to_tensor(x) for x in images]
        return torch.stack(images)
    videos = read_first_n_frames(frame_dir, num_frames=None)
    videos = videos.mul(255).to(torch.uint8).permute(0, 2, 3, 1)
    torchvision.io.write_video(output_path, videos, fps=fps, video_codec='h264', options={'crf': '10'})


def tensor_to_mp4(video, savepath, fps, rescale=True, nrow=None):
    """
    video: torch.Tensor, b,c,t,h,w, 0-1
    if -1~1, enable rescale=True
    """
    n = video.shape[0]
    video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
    nrow = int(np.sqrt(n)) if nrow is None else nrow
    frame_grids = [torchvision.utils.make_grid(framesheet, nrow=nrow, padding=0) for framesheet in video] # [3, grid_h, grid_w]
    grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [T, 3, grid_h, grid_w]
    grid = torch.clamp(grid.float(), -1., 1.)
    if rescale:
        grid = (grid + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1) # [T, 3, grid_h, grid_w] -> [T, grid_h, grid_w, 3]
    torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})

    
def tensor2videogrids(video, root, filename, fps, rescale=True, clamp=True):
    assert(video.dim() == 5) # b,c,t,h,w
    assert(isinstance(video, torch.Tensor))

    video = video.detach().cpu()
    if clamp:
        video = torch.clamp(video, -1., 1.)
    n = video.shape[0]
    video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
    frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(np.sqrt(n))) for framesheet in video] # [3, grid_h, grid_w]
    grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [T, 3, grid_h, grid_w]
    if rescale:
        grid = (grid + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1) # [T, 3, grid_h, grid_w] -> [T, grid_h, grid_w, 3]
    path = os.path.join(root, filename)
    torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'})


def log_local(batch_logs, save_dir, filename, save_fps=10, rescale=True):
    if batch_logs is None:
        return None
    """ save images and videos from images dict """
    def save_img_grid(grid, path, rescale):
        if rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        grid = grid.numpy()
        grid = (grid * 255).astype(np.uint8)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(grid).save(path)

    for key in batch_logs:
        value = batch_logs[key]
        if isinstance(value, list) and isinstance(value[0], str):
            ## a batch of captions
            path = os.path.join(save_dir, "%s-%s.txt"%(key, filename))
            with open(path, 'w') as f:
                for i, txt in enumerate(value):
                    f.write(f'idx={i}, txt={txt}\n')
                f.close()
        elif isinstance(value, torch.Tensor) and value.dim() == 5:
            ## save video grids
            video = value # b,c,t,h,w
            ## only save grayscale or rgb mode
            if video.shape[1] != 1 and video.shape[1] != 3:
                continue
            n = video.shape[0]
            video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
            frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(1), padding=0) for framesheet in video] #[3, n*h, 1*w]
            grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
            if rescale:
                grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
            path = os.path.join(save_dir, "%s-%s.mp4"%(key, filename))
            torchvision.io.write_video(path, grid, fps=save_fps, video_codec='h264', options={'crf': '10'})
            
            ## save frame sheet
            img = value
            video_frames = rearrange(img, 'b c t h w -> (b t) c h w')
            t = img.shape[2]
            grid = torchvision.utils.make_grid(video_frames, nrow=t, padding=0)
            path = os.path.join(save_dir, "%s-%s.jpg"%(key, filename))
            #save_img_grid(grid, path, rescale)
        elif isinstance(value, torch.Tensor) and value.dim() == 4:
            ## save image grids
            img = value
            ## only save grayscale or rgb mode
            if img.shape[1] != 1 and img.shape[1] != 3:
                continue
            n = img.shape[0]
            grid = torchvision.utils.make_grid(img, nrow=1, padding=0)
            path = os.path.join(save_dir, "%s-%s.jpg"%(key, filename))
            save_img_grid(grid, path, rescale)
        else:
            pass

def prepare_to_log(batch_logs, max_images=100000, clamp=True):
    if batch_logs is None:
        return None
    # process
    for key in batch_logs:
        N = batch_logs[key].shape[0] if hasattr(batch_logs[key], 'shape') else len(batch_logs[key])
        N = min(N, max_images)
        batch_logs[key] = batch_logs[key][:N]
        ## in batch_logs: images <batched tensor> & caption <text list>
        if isinstance(batch_logs[key], torch.Tensor):
            batch_logs[key] = batch_logs[key].detach().cpu()
            if clamp:
                try:
                    batch_logs[key] = torch.clamp(batch_logs[key].float(), -1., 1.)
                except RuntimeError:
                    print("clamp_scalar_cpu not implemented for Half")
    return batch_logs

# ----------------------------------------------------------------------------------------------

def fill_with_black_squares(video, desired_len: int) -> Tensor:
    if len(video) >= desired_len:
        return video

    return torch.cat([
        video,
        torch.zeros_like(video[0]).unsqueeze(0).repeat(desired_len - len(video), 1, 1, 1),
    ], dim=0)

# ----------------------------------------------------------------------------------------------
def load_num_videos(data_path, num_videos):
    # first argument can be either data_path of np array 
    if isinstance(data_path, str):
        videos = np.load(data_path)['arr_0'] # NTHWC
    elif isinstance(data_path, np.ndarray):
        videos = data_path
    else:
        raise Exception

    if num_videos is not None:
        videos = videos[:num_videos, :, :, :, :]
    return videos

def npz_to_video_grid(data_path, out_path, num_frames, fps, num_videos=None, nrow=None, verbose=True):
    # videos = torch.tensor(np.load(data_path)['arr_0']).permute(0,1,4,2,3).div_(255).mul_(2) - 1.0 # NTHWC->NTCHW, np int -> torch tensor 0-1
    if isinstance(data_path, str):
        videos = load_num_videos(data_path, num_videos)
    elif isinstance(data_path, np.ndarray):
        videos = data_path
    else:
        raise Exception
    n,t,h,w,c = videos.shape
    videos_th = []
    for i in range(n):
        video = videos[i, :,:,:,:]
        images = [video[j, :,:,:] for j in range(t)]
        images = [to_tensor(img) for img in images]
        video = torch.stack(images)
        videos_th.append(video)
    if verbose:
        videos = [fill_with_black_squares(v, num_frames) for v in tqdm(videos_th, desc='Adding empty frames')] # NTCHW
    else:
        videos = [fill_with_black_squares(v, num_frames) for v in videos_th] # NTCHW

    frame_grids = torch.stack(videos).permute(1, 0, 2, 3, 4) # [T, N, C, H, W]
    if nrow is None:
        nrow = int(np.ceil(np.sqrt(n)))
    if verbose:
        frame_grids = [make_grid(fs, nrow=nrow) for fs in tqdm(frame_grids, desc='Making grids')]
    else:
        frame_grids = [make_grid(fs, nrow=nrow) for fs in frame_grids]

    if os.path.dirname(out_path) != "":
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frame_grids = (torch.stack(frame_grids) * 255).to(torch.uint8).permute(0, 2, 3, 1) # [T, H, W, C]
    torchvision.io.write_video(out_path, frame_grids, fps=fps, video_codec='h264', options={'crf': '10'})
