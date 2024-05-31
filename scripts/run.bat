@echo off

set ckpt=checkpoints\tooncrafter_512_interp_v1\model.ckpt
set config=configs\inference_512_v1.0.yaml

set prompt_dir=prompts\512_interp\
set res_dir=results

set FS=10
rem This model adopts FPS=5, range recommended: 5-30 (smaller value -> larger motion)

set seed=123
set name=tooncrafter_512_interp_seed%seed%

set CUDA_VISIBLE_DEVICES=0

python scripts\evaluation\inference.py ^
--seed %seed% ^
--ckpt_path %ckpt% ^
--config %config% ^
--savedir %res_dir%\%name% ^
--n_samples 1 ^
--bs 1 ^
--height 320 ^
--width 512 ^
--unconditional_guidance_scale 7.5 ^
--ddim_steps 50 ^
--ddim_eta 1.0 ^
--prompt_dir %prompt_dir% ^
--text_input ^
--video_length 16 ^
--frame_stride %FS% ^
--timestep_spacing uniform_trailing ^
--guidance_rescale 0.7 ^
--perframe_ae ^
--interp
