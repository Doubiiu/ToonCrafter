## ___***ToonCrafter: Generative Cartoon Interpolation***___
<!-- ![](./assets/logo_long.png#gh-light-mode-only){: width="50%"} -->
<!-- ![](./assets/logo_long_dark.png#gh-dark-mode-only=100x20) -->
<div align="center">



</div>
 
## 🔆 Introduction

⚠️ Please check our [disclaimer](#disc) first.

🤗 ToonCrafter can interpolate two cartoon images by leveraging the pre-trained image-to-video diffusion priors. Please check our project page and paper for more information. <br>







### 1.1 Showcases (512x320)
<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Input starting frame</td>
        <td>Input ending frame</td>
        <td>Generated video</td>
    </tr>
  <tr>
  <td>
    <img src=assets/72109_125.mp4_00-00.png width="250">
  </td>
  <td>
    <img src=assets/72109_125.mp4_00-01.png width="250">
  </td>
  <td>
    <img src=assets/00.gif width="250">
  </td>
  </tr>


   <tr>
  <td>
    <img src=assets/Japan_v2_2_062266_s2_frame1.png width="250">
  </td>
  <td>
    <img src=assets/Japan_v2_2_062266_s2_frame3.png width="250">
  </td>
  <td>
    <img src=assets/03.gif width="250">
  </td>
  </tr>
  <tr>
  <td>
    <img src=assets/Japan_v2_1_070321_s3_frame1.png width="250">
  </td>
  <td>
    <img src=assets/Japan_v2_1_070321_s3_frame3.png width="250">
  </td>
  <td>
    <img src=assets/02.gif width="250">
  </td>
  </tr> 
  <tr>
  <td>
    <img src=assets/74302_1349_frame1.png width="250">
  </td>
  <td>
    <img src=assets/74302_1349_frame3.png width="250">
  </td>
  <td>
    <img src=assets/01.gif width="250">
  </td>
  </tr>
</table>

### 1.2 Sparse sketch guidance
<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Input starting frame</td>
        <td>Input ending frame</td>
        <td>Input sketch guidance</td>
        <td>Generated video</td>
    </tr>
  <tr>
  <td>
    <img src=assets/72105_388.mp4_00-00.png width="200">
  </td>
  <td>
    <img src=assets/72105_388.mp4_00-01.png width="200">
  </td>
  <td>
    <img src=assets/06.gif width="200">
  </td>
   <td>
    <img src=assets/07.gif width="200">
  </td>
  </tr>

  <tr>
  <td>
    <img src=assets/72110_255.mp4_00-00.png width="200">
  </td>
  <td>
    <img src=assets/72110_255.mp4_00-01.png width="200">
  </td>
  <td>
    <img src=assets/12.gif width="200">
  </td>
   <td>
    <img src=assets/13.gif width="200">
  </td>
  </tr>


</table>


### 2. Applications
#### 2.1 Cartoon Sketch Interpolation (see project page for more details)
<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Input starting frame</td>
        <td>Input ending frame</td>
        <td>Generated video</td>
    </tr>

  <tr>
  <td>
    <img src=assets/frame0001_10.png width="250">
  </td>
  <td>
    <img src=assets/frame0016_10.png width="250">
  </td>
  <td>
    <img src=assets/10.gif width="250">
  </td>
  </tr>


   <tr>
  <td>
    <img src=assets/frame0001_11.png width="250">
  </td>
  <td>
    <img src=assets/frame0016_11.png width="250">
  </td>
  <td>
    <img src=assets/11.gif width="250">
  </td>
  </tr>

</table>


#### 2.2 Reference-based Sketch Colorization
<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Input sketch</td>
        <td>Input reference</td>
        <td>Colorization results</td>
    </tr>
    
  <tr>
  <td>
    <img src=assets/04.gif width="250">
  </td>
  <td>
    <img src=assets/frame0001_05.png width="250">
  </td>
  <td>
    <img src=assets/05.gif width="250">
  </td>
  </tr>


   <tr>
  <td>
    <img src=assets/08.gif width="250">
  </td>
  <td>
    <img src=assets/frame0001_09.png width="250">
  </td>
  <td>
    <img src=assets/09.gif width="250">
  </td>
  </tr>

</table>







## 📝 Changelog
- [ ] Add sketch control and colorization function.
- __[2024.05.29]__: 🔥🔥 Release code and model weights.
- __[2024.05.28]__: Launch the project page and update the arXiv preprint.
<br>


## 🧰 Models

|Model|Resolution|GPU Mem. & Inference Time (A100, ddim 50steps)|Checkpoint|
|:---------|:---------|:--------|:--------|
|ToonCrafter_512|320x512| TBD (`perframe_ae=True`)|[Hugging Face](https://huggingface.co/Doubiiu/ToonCrafter/blob/main/model.ckpt)|


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
  sh scripts/run.sh
```


### 2. Local Gradio demo

Download the pretrained model and put it in the corresponding directory according to the previous guidelines.
```bash
  python gradio_app.py 
```






<!-- ## 🤝 Community Support -->



<a name="disc"></a>
## 📢 Disclaimer
Calm down. Our framework opens up the era of generative cartoon interpolation, but due to the variaity of generative video prior, the success rate is not guaranteed.

⚠️This is an open-source research exploration, instead of commercial products. It can't meet all your expectations.

This project strives to impact the domain of AI-driven video generation positively. Users are granted the freedom to create videos using this tool, but they are expected to comply with local laws and utilize it responsibly. The developers do not assume any responsibility for potential misuse by users.
****