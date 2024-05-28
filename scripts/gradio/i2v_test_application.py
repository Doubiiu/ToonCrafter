import os
import time
from omegaconf import OmegaConf
import torch
from scripts.evaluation.funcs import load_model_checkpoint, save_videos, batch_ddim_sampling, get_latent_z
from utils.utils import instantiate_from_config
from huggingface_hub import hf_hub_download
from einops import repeat
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from einops import rearrange

class Image2Video():
    def __init__(self,result_dir='./tmp/',gpu_num=1,resolution='256_256') -> None:
        self.resolution = (int(resolution.split('_')[0]), int(resolution.split('_')[1])) #hw
        self.download_model()
        
        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        ckpt_path='checkpoints/tooncrafter_'+resolution.split('_')[1]+'_interp_v1/model.ckpt'
        config_file='configs/inference_'+resolution.split('_')[1]+'_v1.0.yaml'
        config = OmegaConf.load(config_file)
        model_config = config.pop("model", OmegaConf.create())
        model_config['params']['unet_config']['params']['use_checkpoint']=False   
        model_list = []
        for gpu_id in range(gpu_num):
            model = instantiate_from_config(model_config)
            # model = model.cuda(gpu_id)
            print(ckpt_path)
            assert os.path.exists(ckpt_path), "Error: checkpoint Not Found!"
            model = load_model_checkpoint(model, ckpt_path)
            model.eval()
            model_list.append(model)
        self.model_list = model_list
        self.save_fps = 8

    def get_image(self, image, prompt, steps=50, cfg_scale=7.5, eta=1.0, fs=3, seed=123, image2=None):
        seed_everything(seed)
        transform = transforms.Compose([
            transforms.Resize(min(self.resolution)),
            transforms.CenterCrop(self.resolution),
            ])
        torch.cuda.empty_cache()
        print('start:', prompt, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        start = time.time()
        gpu_id=0
        if steps > 60:
            steps = 60 
        model = self.model_list[gpu_id]
        model = model.cuda()
        batch_size=1
        channels = model.model.diffusion_model.out_channels
        frames = model.temporal_length
        h, w = self.resolution[0] // 8, self.resolution[1] // 8
        noise_shape = [batch_size, channels, frames, h, w]

        # text cond
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_emb = model.get_learned_conditioning([prompt])

            # img cond
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(model.device)
            img_tensor = (img_tensor / 255. - 0.5) * 2

            image_tensor_resized = transform(img_tensor) #3,h,w
            videos = image_tensor_resized.unsqueeze(0).unsqueeze(2) # bc1hw
            
            # z = get_latent_z(model, videos) #bc,1,hw
            videos = repeat(videos, 'b c t h w -> b c (repeat t) h w', repeat=frames//2)
            
            


            img_tensor2 = torch.from_numpy(image2).permute(2, 0, 1).float().to(model.device)
            img_tensor2 = (img_tensor2 / 255. - 0.5) * 2
            image_tensor_resized2 = transform(img_tensor2) #3,h,w
            videos2 = image_tensor_resized2.unsqueeze(0).unsqueeze(2) # bchw
            videos2 = repeat(videos2, 'b c t h w -> b c (repeat t) h w', repeat=frames//2)
            
            
            videos = torch.cat([videos, videos2], dim=2)
            z, hs = self.get_latent_z_with_hidden_states(model, videos)

            img_tensor_repeat = torch.zeros_like(z)

            img_tensor_repeat[:,:,:1,:,:] = z[:,:,:1,:,:]
            img_tensor_repeat[:,:,-1:,:,:] = z[:,:,-1:,:,:]


            cond_images = model.embedder(img_tensor.unsqueeze(0)) ## blc
            img_emb = model.image_proj_model(cond_images)

            imtext_cond = torch.cat([text_emb, img_emb], dim=1)

            fs = torch.tensor([fs], dtype=torch.long, device=model.device)
            cond = {"c_crossattn": [imtext_cond], "fs": fs, "c_concat": [img_tensor_repeat]}
            
            ## inference
            batch_samples = batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=steps, ddim_eta=eta, cfg_scale=cfg_scale, hs=hs)

            ## remove the last frame
            if image2 is None:
                batch_samples = batch_samples[:,:,:,:-1,...]
            ## b,samples,c,t,h,w
            prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
            prompt_str = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
            prompt_str=prompt_str[:40]
            if len(prompt_str) == 0:
                prompt_str = 'empty_prompt'

        save_videos(batch_samples, self.result_dir, filenames=[prompt_str], fps=self.save_fps)
        print(f"Saved in {prompt_str}. Time used: {(time.time() - start):.2f} seconds")
        model = model.cpu()
        return os.path.join(self.result_dir, f"{prompt_str}.mp4")
    
    def download_model(self):
        REPO_ID = 'Doubiiu/ToonCrafter'
        filename_list = ['model.ckpt']
        if not os.path.exists('./checkpoints/tooncrafter_'+str(self.resolution[1])+'_interp_v1/'):
            os.makedirs('./checkpoints/tooncrafter_'+str(self.resolution[1])+'_interp_v1/')
        for filename in filename_list:
            local_file = os.path.join('./checkpoints/tooncrafter_'+str(self.resolution[1])+'_interp_v1/', filename)
            if not os.path.exists(local_file):
                hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir='./checkpoints/tooncrafter_'+str(self.resolution[1])+'_interp_v1/', local_dir_use_symlinks=False)
    
    def get_latent_z_with_hidden_states(self, model, videos):
        b, c, t, h, w = videos.shape
        x = rearrange(videos, 'b c t h w -> (b t) c h w')
        encoder_posterior, hidden_states = model.first_stage_model.encode(x, return_hidden_states=True)

        hidden_states_first_last = []
        ### use only the first and last hidden states
        for hid in hidden_states:
            hid = rearrange(hid, '(b t) c h w -> b c t h w', t=t)
            hid_new = torch.cat([hid[:, :, 0:1], hid[:, :, -1:]], dim=2)
            hidden_states_first_last.append(hid_new)

        z = model.get_first_stage_encoding(encoder_posterior).detach()
        z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
        return z, hidden_states_first_last
if __name__ == '__main__':
    i2v = Image2Video()
    video_path = i2v.get_image('prompts/art.png','man fishing in a boat at sunset')
    print('done', video_path)