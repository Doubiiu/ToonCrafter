## ___***ToonCrafter: Generative Cartoon Interpolation***___
<!-- ![](./assets/logo_long.png#gh-light-mode-only){: width="50%"} -->
<!-- ![](./assets/logo_long_dark.png#gh-dark-mode-only=100x20) -->
<div align="center">



</div>
 
## 🔆 Introduction

🤗 ToonCrafter can interpolate two cartoon images by leveraging the pre-trained image-to-video diffusion priors. Please check our project page and paper for more information. <br>







<!-- ### 1 Showcases (512x320)
<table class="center">
  <tr>
  <td>
    <img src=assets/showcase/bloom2.gif width="340">
  </td>
  <td>
    <img src=assets/showcase/train_anime02.gif width="340">
  </td>
  </tr>

  <tr>
  <td>
    <img src=assets/showcase/pour_honey.gif width="340">
  </td>
  <td>
    <img src=assets/showcase/lighthouse.gif width="340">
  </td>
  </tr>
</table>




### 2. Applications
#### 2.1 Cartoon Sketch Interpolation (see project page for more details)
<table class="center">
  <tr>
    <td colspan="4"><img src=assets/application/storytellingvideo.gif width="250"></td>
  </tr>
</table >

#### 2.2 Reference-based Sketch Colorization

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Input starting frame</td>
        <td>Input ending frame</td>
        <td>Generated video</td>
    </tr>
  <tr>
  <td>
    <img src=assets/application/gkxX0kb8mE8_input_start.png width="250">
  </td>
  <td>
    <img src=assets/application/gkxX0kb8mE8_input_end.png width="250">
  </td>
  <td>
    <img src=assets/application/gkxX0kb8mE8.gif width="250">
  </td>
  </tr>


   <tr>
  <td>
    <img src=assets/application/smile_start.png width="250">
  </td>
  <td>
    <img src=assets/application/smile_end.png width="250">
  </td>
  <td>
    <img src=assets/application/smile.gif width="250">
  </td>
  </tr>
  <tr>
  <td>
    <img src=assets/application/stone01_start.png width="250">
  </td>
  <td>
    <img src=assets/application/stone01_end.png width="250">
  </td>
  <td>
    <img src=assets/application/stone01.gif width="250">
  </td>
  </tr> 
</table > -->






## 📝 Changelog
- __[2024.05.29]__: 🔥🔥 Release code and model weights.
- __[2024.05.28]__: Launch the project page and update the arXiv preprint.
<br>


## 🧰 Models

|Model|Resolution|GPU Mem. & Inference Time (A100, ddim 50steps)|Checkpoint|
|:---------|:---------|:--------|:--------|
|ToonCrafter_512|320x512|12.8GB & 20s (`perframe_ae=True`)|[Hugging Face](https://huggingface.co/Doubiiu/ToonCrafter/blob/main/model.ckpt)|


Currently, our ToonCrafter can support generating videos of up to 16 frames with a resolution of 512x320. The inference time can be reduced by using fewer DDIM steps.



## ⚙️ Setup

### Install Environment via Anaconda (Recommended)
```bash
conda create -n tooncrafter python=3.8.5
conda activate tooncrafter
pip install -r requirements.txt
```


## 💫 Inference
### 1. Command line

Download pretrained ToonCrafter_512 and put the `model.ckpt` in `checkpoints/tooncrafter_512_interp_v1/model.ckpt`.
```bash
  sh scripts/run_application.sh # Generate frame interpolation
```


### 2. Local Gradio demo

Download the pretrained model and put it in the corresponding directory according to the previous guidelines.
```bash
  python gradio_app.py 
```






<!-- ## 🤝 Community Support -->




## 📢 Disclaimer
This project strives to impact the domain of AI-driven video generation positively. Users are granted the freedom to create videos using this tool, but they are expected to comply with local laws and utilize it responsibly. The developers do not assume any responsibility for potential misuse by users.
****