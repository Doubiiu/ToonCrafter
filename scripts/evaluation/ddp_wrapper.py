import datetime
import argparse, importlib
from pytorch_lightning import seed_everything

import torch
import torch.distributed as dist

def setup_dist(local_rank):
    if dist.is_initialized():
        return
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl', init_method='env://')


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, help="module name", default="inference")
    parser.add_argument("--local_rank", type=int, nargs="?", help="for ddp", default=0)
    args, unknown = parser.parse_known_args()
    inference_api = importlib.import_module(args.module, package=None)

    inference_parser = inference_api.get_parser()
    inference_args, unknown = inference_parser.parse_known_args()

    seed_everything(inference_args.seed)
    setup_dist(args.local_rank)
    torch.backends.cudnn.benchmark = True
    rank, gpu_num = get_dist_info()

    # inference_args.savedir = inference_args.savedir+str('_seed')+str(inference_args.seed)
    print("@DynamiCrafter Inference [rank%d]: %s"%(rank, now))
    inference_api.run_inference(inference_args, gpu_num, rank)