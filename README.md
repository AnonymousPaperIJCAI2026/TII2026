<div align="left"> 
<h1> 📌 TACO </h1>
<h3>TACO: Token Calibration with Prompt Anchors for Cross-Domain Zero/Few-Shot Anomaly Detection and Localization</h3>
</div>


<div align="center"> <img src="figs/fig1.png" width="60%"> </div>

<div align="justify">

## ⭐ Abstract 
Industrial anomaly detection and localization increasingly rely on large vision--language models (VLMs) for their strong priors and promising zero-shot transfer. However, directly deploying VLMs across unseen industrial domains remains challenging due to (i) semantic inconsistency caused by prompt sensitivity and domain shifts, and (ii) insufficient, task-mismatched local evidence for reliable pixel-level localization. To address these issues, we propose TACO, a parameter-efficient framework that couples prompt anchors with token calibration for zero- and few-shot cross-domain anomaly detection and localization. TACO learns stable normal/ abnormal textual anchors from a prompt ensemble (TWPP), restores anomaly-sensitive local cues via lightweight token adapters with enhanced aggregation (DTA-EA), and performs low-rank cross-modal calibration that injects anchor semantics into visual tokens to reduce representational drift and improve token separability (PTC). The framework further supports few-shot target adaptation by building an inference-time memory bank from a few normal samples. Extensive experiments on diverse 2D benchmarks (VisA, BTAD, DAGM, DTD-Synthetic) and the 3D MVTec-3D dataset demonstrate superior transferability. In zero-shot 2D cross-domain evaluation, TACO establishes a new state-of-the-art with 94.2\% Image-AUROC and 95.8\% AP, surpassing the strongest prior baseline by +1.4\%. For pixel-level localization, it achieves 95.7\% Pixel-AUROC and 87.4\% PRO. Furthermore, our few-shot adaptation on MVTec-3D significantly outperforms existing methods, achieving 81.8\% AUROC in the 1-shot setting---surpassing the previous state-of-the-art by +2.6\%.
</div>

📴**Keywords**: Cross-Domain, Zero/Few-Shot, Anomaly Detection and Segmentation, Large Vision Language Model

<div align="center"> <img src="figs/fig2.png " width="100%"> </div>


## 🚀 Get Started

⚙️ Environment
- python >= 3.8.5
- pytorch >= 1.10.0
- torchvision >= 0.11.1
- numpy >= 1.19.2
- scipy >= 1.5.2
- kornia >= 0.6.1
- pandas >= 1.1.3
- opencv-python >= 4.5.4
- pillow
- tqdm
- ftfy
- regex

### Device
Single NVIDIA A40 GPU

## 📦 Pretrained model
- CLIP: ##################################################################

    👉 Download and put it under `CLIP/ckpt` folder



## 🏥🏭 Medical and Industrial Anomaly Detection Benchmark(2D\3D)

1. We will provide the pre-processed benchmark. Please download the following dataset

    

2. Place it within the master directory `data` and unzip the dataset.

    ```
    
    ```


## 📂 File Structure
After the preparation work, the whole project should have the following structure:

```
code
├─ ckpt
│  └─ zero-shot
├─ CLIP
│  ├─ bpe_simple_vocab_16e6.txt.gz
│  ├─ ckpt
│  │  └─ ViT-L-14-336px.pt
│  ├─ clip.py
│  ├─ model.py
│  ├─ models.py
│  ├─ model_configs
│  │  └─ ViT-L-14-336.json
│  ├─ modified_resnet.py
│  ├─ openai.py
│  ├─ tokenizer.py
│  └─ transformer.py
├─ data
│  ├─ Mvtec3D
│  │  ├─ valid
│  │  └─ test
│  ├─ BTAD
│  │  ├─ valid
│  │  └─ test
│  ├─ Mvtec
│  │  ├─ valid
│  │  └─ test
│  ├─ ...
│  └─ Visa
│     ├─ valid
│     └─ test
├─ dataset
│  ├─ fewshot_seed
│  │  └─ Mvtec3D
│  ├─ medical_few.py
│  └─ medical_zero.py
├─ loss.py
├─ readme.md
├─ train_few.py
├─ train_zero.py
├─ test.py
└─ utils.py

```


## ⚡ Quick Start

`python test.py`

For example, to test on the Visa , simply run:

`###########################`

### Training

`python train_zero.py`




## 🖼️ Visualization
<center><img src="figs/fig3.png "width="70%"></center>
<center><img src="figs/fig4.png "width="70%"></center>


## 🙏 Acknowledgement
We borrow some codes from [OpenCLIP](https://github.com/mlfoundations/open_clip), and [April-GAN](https://github.com/ByChelsea/VAND-APRIL-GAN).

## 📬 Contact

If you have any problem with this code, please feel free to contact **** and ****.

