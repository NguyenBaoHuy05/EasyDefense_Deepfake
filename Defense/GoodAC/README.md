##### Table of contents
1. [Environment setup](#environment-setup)
2. [Dataset](#dataset)
3. [How to run](#how-to-run)
4. [Contacts](#contacts)
5. [Acknowledgement](#acknowledgement)
6. [Citation](#citation)

# Harnessing Global-Local Collaborative Adversarial Perturbation for Anti-Customization (CVPR'25)
This repository provides the official PyTorch implementation of the following paper: 
<div align="center">
  <a href="" target="_blank">Long&nbsp;Xu</a><sup>1</sup> &emsp;
  <a href="" target="_blank">Jiakai&nbsp;Wang</a><sup>2*</sup> &emsp;
  <a href="" target="_blank">Haojie&nbsp;Hao</a><sup>1</sup> &emsp;
  <a href="" target="_blank">Haotong&nbsp;Qin</a><sup>3</sup>&emsp;
  <a href="" target="_blank">Jiejie&nbsp;Zhao</a><sup>2</sup>&emsp;
  <a href="" target="_blank">Xianglong&nbsp;Liu</a><sup>1,2</sup>&emsp;
 <br>{xulonguuid, haojiehao, xlliu}@buaa.edu.cn, {wangjk, zhaojiejie}@zgclab.edu.cn, haotong.qin@pbl.ee.ethz.ch<br>
</div>
<br>


## Environment setup
Install dependencies:
```shell
cd GoodAC
conda env create -f environment.yml
conda activate GoodAC
```

Pretrained checkpoints of different Stable Diffusion versions can be **downloaded** from provided links in the table below:
<table style="width:100%">
  <tr>
    <th>Version</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>2.1</td>
    <td><a href="https://huggingface.co/stabilityai/stable-diffusion-2-1-base">stable-diffusion-2-1-base</a></td>
  </tr>
  <tr>
    <td>1.5</td>
    <td><a href="https://huggingface.co/runwayml/stable-diffusion-v1-5">stable-diffusion-v1-5</a></td>
  </tr>
  <tr>
    <td>1.4</td>
    <td><a href="https://huggingface.co/CompVis/stable-diffusion-v1-4">stable-diffusion-v1-4</a></td>
  </tr>
</table>

Please download the pretrain weights and define "$MODEL_PATH" in the script. Note: Stable Diffusion version 2.1 is the default version in all of our experiments.

> GPU allocation: All experiments are performed on a single NVIDIA 48GB L2 GPU.

## Dataset 
Thanks for Anti-Dreambooth's great efforts, there are two datasets: VGGFace2 and CelebA-HQ which are provided at [here](https://drive.google.com/drive/folders/1vlpmoKPZVgZZp-ANBzg915hOWPlCYv95?usp=sharing).

For convenient testing, we have provided a split set of one subject in CelebA-HQ at `./data/CelebA-HQ/103` as the Anti-dreambooth does.

## How to run

To defense Stable Diffusion version 2.1 (default) with ASPL, you can run
```
bash scripts/attack_aspl.sh
```

To defense Stable Diffusion version 2.1 (default) with GoodAC, you can run
```
bash scripts/attack_goodac.sh
```


If you want to train a DreamBooth model from your own data, whether it is clean or perturbed, you may run the following script:
```
bash scripts/train_dreambooth_alone.sh
```

Inference: generates examples with multiple-prompts
```
python infer.py --model_path <path to DREAMBOOTH model>/checkpoint-1000 --output_dir $<path to DREAMBOOTH model>/checkpoint-1000-test-infer
```

## Contacts
If you have any problems, please open an issue in this repository or send an email to [xulonguuid@163.com](mailto:xulonguuid@163.com)

## Acknowledgement
This repo is heavil based on [Anti-DB](https://github.com/VinAIResearch/Anti-DreamBooth). Thanks for their impressive works!

## Citation
Details of algorithms and experimental results can be found in [our following paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Harnessing_Global-Local_Collaborative_Adversarial_Perturbation_for_Anti-Customization_CVPR_2025_paper.pdf):
```bibtex
@inproceedings{xu2025harnessing,
  title={Harnessing Global-Local Collaborative Adversarial Perturbation for Anti-Customization},
  author={Xu, Long and Wang, Jiakai and Hao, Haojie and Qin, Haotong and Zhao, Jiejie and Liu, Xianglong},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={13414--13423},
  year={2025}
}
```
**Please CITE** our paper if you find this work useful for your research.
