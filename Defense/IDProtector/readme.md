<div align="center">
<h1>IDProtector: An Adversarial Noise Encoder to Protect Against ID-Preserving Image Generation</h1>
</div>

<div align="center">
    <a href="https://scholar.google.com/citations?user=L2YS0jgAAAAJ&hl=en">Yiren Song</a><sup>&#42;</sup>&nbsp;, <a href="https://scholar.google.com/citations?user=eBvav_0AAAAJ">Pei Yang</a><sup>&#42;</sup>&nbsp;, <a href="https://scholar.google.com/citations?user=GMrjppAAAAAJ&hl=en">Hai Ci</a><sup>&#x2709</sup>, and <a href="https://sites.google.com/view/showlab">Mike Zheng Shou</a><sup>&#x2709</sup>

</div>

<div align="center">
    <sup>&#42;</sup>Equal contribution &nbsp; <sup>&#x2709</sup>Corresponding author
</div>

<div align="center">
    <a href='https://sites.google.com/view/showlab/home?authuser=0' target='_blank'>Show Lab</a>, National University of Singapore
    <p>
</div>

<div align="center">
    <a href="https://arxiv.org/abs/2412.11638">
        <img src="https://img.shields.io/badge/arXiv-2412.11638-b31b1b.svg?logo=arXiv" alt="arXiv">
    </a>
    <p>
</div>

<br>

## ⭐ About

<img src="assets/teaser.jpg" width="1024">

IDProtector is a model that adds protective noise to images, such that if the protected image is used in encoder-based identity-preserving generation, the resulting face will no longer resemble the original, thus achieving privacy protection.

## 🚀 Getting Started

### Pre-Requisite
 - Python 3.9.18: `conda create -n IDProtector python=3.9.18`
 - CUDA 11.6

### Install dependencies
```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install transformers huggingface-hub spaces numpy accelerate safetensors omegaconf peft==0.12.0 gradio insightface==0.7.3 jupyter matplotlib wandb kornia diffusers["torch"]==0.29.2 transformers==4.37.2 tokenizers==0.15.2 onnxruntime onnx2torch einops timm tensorboard mxnet-cu116 scikit-image scikit-learn opencv-python brisque pyiqa torch-dct

conda install -c conda-forge nccl
```

### Clone this repo
```bash
git clone https://github.com/yangpei-comp/IDProtector_Preview.git
git submodule init
git submodule update
```

### Download models
1. Under the root directory of this repo, run `python -m scripts.setup.model_setup`
2. If your Hugging Face model cache path is not the default path, please go to `modules/model_instances.py` and change the variables starting with `PATH` to your corresponding paths
3. Download the antelopev2 model from [this link](https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view) and place it in `generation_methods/InstantID/models/antelopev2/*.onnx`
4. Download InstantID models:
```bash
cp scripts/setup/model_setup_instantid.py generation_methods/InstantID/
cd generation_methods/InstantID
python model_setup_instantid.py
```
5. Download the IPAdapter models:
```bash
cd generation_methods/IP_Adapter
git lfs install
git clone https://huggingface.co/h94/IP-Adapter
mv IP-Adapter/models models
mv IP-Adapter/sdxl_models sdxl_models
```
6. Download the insightface model from [this link](https://drive.google.com/file/d/17fEWczMzTUDzRTv9qN3hFwVbkqRD7HE7/view?usp=sharing) and place it in `utils/FaceImageQuality/insightface/model`

## ⚡ Inference

### Run Complete Experiment

1. Place the images to be protected (jpg or png) in a folder and record the path to this folder.
     - Each image must contain a face that can be detected by ArcFace. If multiple faces exist, protection will be applied to the largest face.
     - Each image must be 512×512 pixels. If you want to protect images of arbitrary sizes, please follow [Run Protection Only](#run-protection-only).
2. Run protection and generation experiments:
```bash
bash scripts/evaluation/run_eval.sh \
$cuda_index \
/path/to/ref_imgs \
/path/to/data_dir \
/path/to/gen_dir
```
3. You can find the protected images in `data_dir/IDProtector/clean`. The experimental results and various metrics are stored as multiple CSV files in the corresponding folders under `gen_dir`. Please refer to [File Organization](docs/file_organisation.md).

### Run Protection Only

If you only need to use IDProtector to protect your images, place jpg/png images of any size containing faces in `/path/to/input_dir`, then run:

```bash
CUDA_VISIBLE_DEVICES=$cuda_index python -m scripts.evaluation.perturb_get_metric \
--in_channels 5 \
--epsilon ${epsilon} \
--clip_epsilon ${clip_epsilon} \
--clean_data_dir /path/to/input_dir \
--save_dir /path/to/output_dir \
--path_to_state_dict ${path_to_state_dict} \
--metrics_save_path /path/to/output_dir/metrics.csv \
--batch_size 1 \
--cuda 0
```

## 📄 Citation

```
@inproceedings{song2025idprotector,
    title={IDProtector: An Adversarial Noise Encoder to Protect Against ID-Preserving Image Generation},
    author={Song, Yiren and Yang, Pei and Ci, Hai and Shou, Mike Zheng},
    booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
    pages={3019--3028},
    year={2025}
}
```