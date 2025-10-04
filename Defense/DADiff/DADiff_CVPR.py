import os

import torch
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

cpu_num = 10
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

import argparse
import copy
import hashlib
import itertools
import logging
from pathlib import Path

import datasets
import diffusers
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from attention_loader_CVPR import AttentionLoader

logger = get_logger(__name__)


class DreamBoothDatasetFromTensor(Dataset):
    """Just like DreamBoothDataset, but take instance_images_tensor instead of path"""

    def __init__(
            self,
            instance_images_tensor,
            instance_prompt,
            tokenizer,
            class_data_root=None,
            class_prompt=None,
            size=512,
            center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_images_tensor = instance_images_tensor
        self.num_instance_images = len(self.instance_images_tensor)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.instance_images_tensor[index % self.num_instance_images]
        example["instance_images"] = instance_image
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5", # "CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1-base"
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.com/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir_for_train",
        type=str,
        default="./datasets/vggface2/000050/set_A",
        # required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_data_dir_for_adversarial",
        type=str,
        default="./datasets/vggface2/000050/set_B",
        # required=True,
        help="A folder containing the images to add adversarial noise",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default="./datasets/class-person",
        # required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="a photo of sks person",
        # required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="a photo of person",
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=True,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=200,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="attack_outputs/DADiff_sks-v1.5/vggface2_000050_ADVERSARIAL",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,  # 512
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=True,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        default=True,
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=50,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--max_f_train_steps",
        type=int,
        default=3,
        help="Total number of sub-steps to train surogate model.",
    )
    parser.add_argument(
        "--max_adv_train_steps",
        type=int,
        default=6,
        help="Total number of sub-steps to train adversarial noise.",
    )
    parser.add_argument(
        "--checkpointing_iterations",
        type=int,
        default=1,
        help=("Save a checkpoint of the training state every X iterations."),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-7,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16", "fp32"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        default=True,
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--pgd_alpha",
        type=float,
        default=5e-3,
        help="The step size for pgd.",
    )
    parser.add_argument(
        "--pgd_eps",
        type=float,
        default=0.05,
        help="The noise budget for pgd.",
    )
    parser.add_argument(
        "--target_image_path",
        default=None,
        help="target image for attacking",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

def load_data(data_dir, size=512, center_crop=True) -> torch.Tensor:
    image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    images = [image_transforms(Image.open(i).convert("RGB")) for i in list(Path(data_dir).iterdir())]
    images = torch.stack(images)
    return images


def train_one_epoch(
        args,
        models,
        tokenizer,
        noise_scheduler,
        vae,
        data_tensor: torch.Tensor,
        num_steps=20,
):
    # Load the tokenizer

    unet, text_encoder = copy.deepcopy(models[0]), copy.deepcopy(models[1])
    params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters())

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    train_dataset = DreamBoothDatasetFromTensor(
        data_tensor,
        args.instance_prompt,
        tokenizer,
        args.class_data_dir,
        args.class_prompt,
        args.resolution,
        args.center_crop,
    )

    weight_dtype = torch.bfloat16
    device = torch.device("cuda")

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    for step in range(num_steps):
        unet.train()
        text_encoder.train()

        step_data = train_dataset[step % len(train_dataset)]
        pixel_values = torch.stack([step_data["instance_images"], step_data["class_images"]]).to(
            device, dtype=weight_dtype
        )
        input_ids = torch.cat([step_data["instance_prompt_ids"], step_data["class_prompt_ids"]], dim=0).to(device)

        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(input_ids)[0]

        # Predict the noise residual
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        # with prior preservation loss
        if args.with_prior_preservation:
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)  # 对新图片的预测+对原始图片的预测
            target, target_prior = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

            # Add the prior loss to the instance loss.
            loss = instance_loss + args.prior_loss_weight * prior_loss

        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad()
        print(
            f"Step #{step}, loss: {loss.detach().item()}, prior_loss: {prior_loss.detach().item()}, instance_loss: {instance_loss.detach().item()}"
        )

    return [unet, text_encoder]


def pgd_attack(
        args,
        models,
        tokenizer,
        noise_scheduler,
        vae,
        data_tensor: torch.Tensor,
        original_images: torch.Tensor,
        num_steps: int,
        long_term_momentum: torch.Tensor,
        delta: torch.Tensor,
        context_adv
):
    """Return new perturbed data"""

    unet, text_encoder = models
    unet.eval()
    text_encoder.eval()
    weight_dtype = torch.bfloat16
    device = torch.device("cuda")

    instance_ids = tokenizer(
        args.instance_prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.repeat(len(data_tensor), 1)

    attn_loader = AttentionLoader(device=device, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, unet=unet, scheduler=noise_scheduler)
    attn_loader.init_(noise_scheduler.timesteps, pnp_attn_t=1.0)
    vae = attn_loader.vae
    unet = attn_loader.unet
    text_encoder = attn_loader.text_encoder

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    perturbed_images = torch.clamp(original_images + delta, -1, 1)
    eps = args.pgd_eps

    import torch.nn.functional as F
    batchsize = len(original_images)
    up_attn_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}

    print("2. Get the adversarial prompt tensor...")

    _ = vae.requires_grad_(False)
    _ = text_encoder.requires_grad_(False)
    _ = unet.requires_grad_(False)

    ori_latents = (vae.encode(
        original_images.to(device, dtype=weight_dtype)).latent_dist.sample()) * vae.config.scaling_factor

    if context_adv == None:
        context_adv = text_encoder(instance_ids[0:1].to(device))[0]
        context_adv.requires_grad_(True)
        latents = (vae.encode(perturbed_images.to(device, dtype=weight_dtype)).latent_dist.sample()) * vae.config.scaling_factor
        '''随机t生成APV'''
        for j in range(500):
            t = torch.randint(low=0, high=999, device=device, size=(1,)).long()
            noise = torch.randn_like(latents, dtype=weight_dtype).to(device)
            current_adv_latents = noise_scheduler.add_noise(latents, noise, t)
            adv = unet(current_adv_latents, t, encoder_hidden_states=context_adv.repeat(len(data_tensor), 1, 1)).sample
            loss = -F.mse_loss(adv, noise, reduction="mean")
            loss.backward()
            grad = context_adv.grad
            grad = grad / (torch.mean(torch.abs(grad), (1, 2), keepdim=True) + 1e-12)

            context_adv.data = context_adv.data - 0.005 * grad  # .sign()
            if j % 100 == 0:
                print(-loss)
        print("Get the adversarial prompt!")
        context_adv = context_adv.repeat(len(data_tensor), 1, 1)
    else:
        print("Already got the APV, keep running.")

    print("3. Start attacking unet...")
    how_many_steps = 25
    noise_scheduler.set_timesteps(how_many_steps)

    score_unet_saver = []
    score_attn_saver = []
    score_sum_saver = []
    loss_sum_saver = []
    loss_unet_saver = []
    loss_attn_saver = []

    for step in range(num_steps):
        print("Start Step %d: " % step)
        rand_interval = np.random.randint(0, 1000 // how_many_steps, size=how_many_steps)
        rand_interval = torch.tensor(rand_interval)
        current_ts = noise_scheduler.timesteps + rand_interval

        _ = vae.requires_grad_(False)
        _ = text_encoder.requires_grad_(False)
        _ = unet.requires_grad_(False)

        context_instance = text_encoder(instance_ids.to(device))[0]

        adv_unet_grads = torch.zeros(len(current_ts), *delta.size()).cpu()
        adv_attn_grads = torch.zeros_like(adv_unet_grads).cpu()

        sum_loss_unet = torch.tensor(0.0).to(device)
        sum_loss_attn2 = torch.tensor(0.0).to(device)
        sum_loss_attn1 = torch.tensor(0.0).to(device)
        for mode in range(2):
            for i, actual_t in enumerate(tqdm(current_ts)):

                delta.requires_grad = True
                perturbed_images = torch.clamp(original_images + delta, -1, 1)
                adv_latents = vae.encode(perturbed_images.to(device, dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(adv_latents, dtype=weight_dtype).to(device)

                if mode == 0:
                    contexts_cat = torch.cat([context_instance, context_adv], dim=0)
                    latents_cat = torch.cat([adv_latents, ori_latents], dim=0)
                    noise_cat = torch.cat([noise, noise], dim=0)
                    fea_outs_t = []
                    with torch.no_grad():
                        current_latents = noise_scheduler.add_noise(ori_latents, noise, torch.LongTensor([actual_t]))
                        _ = unet(current_latents, actual_t, encoder_hidden_states=context_instance).sample

                        # 上采样层**********
                        for res in up_attn_dict:
                            for block in up_attn_dict[res]:
                                module = unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1  # 交叉注意力层
                                attn_ins = module.attn
                                fea_outs_t.append(attn_ins.clone())
                        # 上采样层**********
                else:
                    contexts_cat = context_instance
                    latents_cat = adv_latents
                    noise_cat = noise

                current_adv_latents = noise_scheduler.add_noise(latents_cat, noise_cat, torch.LongTensor([actual_t]))
                outs = unet(current_adv_latents, actual_t, encoder_hidden_states=contexts_cat).sample

                if mode == 0:
                    '''Self-Attention'''
                    count_attn1 = 0
                    loss_attn1 = torch.tensor(0).to(device, dtype=torch.float)

                    # 上采样层**********
                    for res in up_attn_dict:
                        for block in up_attn_dict[res]:
                            module = unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1  # 自注意力层
                            adv_attn_ins, _ = module.attn.chunk(2)
                            module.attn = None
                            loss_attn1 += F.mse_loss(adv_attn_ins.float(), fea_outs_t[count_attn1].float(), reduction="mean")
                            count_attn1 += 1
                    # 上采样层**********
                    '''Self-Attention'''

                    '''Cross-Attention'''
                    loss_attn2 = torch.tensor(0).to(device, dtype=torch.float)
                    count_attn2 = 0

                    # 上采样层**********
                    for res in up_attn_dict:
                        for block in up_attn_dict[res]:
                            module = unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2  # 自注意力层
                            attn_ins, attn_adv = module.attn.chunk(2)
                            count_attn2 += 1
                            loss_attn2 += F.cosine_similarity(attn_ins.float().view(batchsize, -1),
                                                              attn_adv.float().view(batchsize, -1)).mean()
                    # 上采样层**********
                    '''Cross-Attention'''

                    loss = 0.5 * loss_attn2 / count_attn2 + 0.5 * loss_attn1 / count_attn1
                    sum_loss_attn1 += loss_attn1 / count_attn1
                    sum_loss_attn2 += loss_attn2 / count_attn2
                else:
                    out_ins = outs
                    loss_unet = F.mse_loss(out_ins.float(), noise.float(), reduction="mean")
                    loss = loss_unet
                    sum_loss_unet += loss_unet

                unet.zero_grad()
                text_encoder.zero_grad()

                loss.backward()
                grad = delta.grad.data
                if mode == 0:
                    adv_attn_grads[i] += grad
                else:
                    adv_unet_grads[i] += grad

                delta = delta.detach()

        print("Step:", step,
              " Avg Loss_unet:", sum_loss_unet.item() / len(current_ts),
              " Avg Loss_attn2:", sum_loss_attn2.item() / len(current_ts),
              " Avg Loss_attn1:", sum_loss_attn1.item() / len(current_ts),
              )

        sum_unet_grads = torch.sum(adv_unet_grads, dim=0)
        score_unet = torch.sum(torch.abs(sum_unet_grads.data)).item()
        sum_unet_grads = sum_unet_grads / (torch.mean(torch.abs(sum_unet_grads), (1, 2, 3), keepdim=True) + 1e-12)

        sum_attn_grads = torch.sum(adv_attn_grads, dim=0)
        score_attn = torch.sum(torch.abs(sum_attn_grads.data)).item()
        sum_attn_grads = sum_attn_grads / (torch.mean(torch.abs(sum_attn_grads), (1, 2, 3), keepdim=True) + 1e-12)

        sum_grads = sum_unet_grads + 0.4 * sum_attn_grads
        score_sum = torch.sum(torch.abs(sum_grads.data)).item()

        score_unet_saver.append(score_unet)
        score_attn_saver.append(score_attn)
        score_sum_saver.append(score_sum)

        sum_loss = (sum_loss_unet.item() + sum_loss_attn2.item() + sum_loss_attn1.item()) / len(current_ts)
        loss_sum_saver.append(sum_loss)
        loss_attn_saver.append((sum_loss_attn2.item() + sum_loss_attn1.item()) / len(current_ts))
        loss_unet_saver.append(sum_loss_unet.item() / len(current_ts))

        # sum_grads.data = sum_grads.data + 0.6 * long_term_momentum.data
        # long_term_momentum.data = sum_grads.data

        delta.data = delta.data + args.pgd_alpha * torch.sign(sum_grads.data)
        delta.data = torch.clamp(delta.data, min=-eps, max=+eps)
        delta = delta.detach_()

    print(delta.max(), delta.min(), delta.size())
    perturbed_images = torch.clamp(original_images + delta, min=-1, max=+1).detach_()
    return perturbed_images, long_term_momentum, delta, context_adv, loss_sum_saver, loss_unet_saver, loss_attn_saver, score_sum_saver, score_unet_saver, score_attn_saver


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if args.mixed_precision == "fp32":
                torch_dtype = torch.float32
            elif args.mixed_precision == "fp16":
                torch_dtype = torch.float16
            elif args.mixed_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision, local_files_only=True
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                    sample_dataloader,
                    desc="Generating class images",
                    disable=not accelerator.is_local_main_process,
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    ).cuda()

    vae.requires_grad_(False)

    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    clean_data = load_data(
        args.instance_data_dir_for_train,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    original_data = load_data(
        args.instance_data_dir_for_adversarial,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    delta = torch.tensor(
        np.random.uniform(-args.pgd_eps, args.pgd_eps, original_data.shape).astype('float32')).to(
        original_data.device)
    perturbed_data = torch.clamp(original_data + delta, -1, 1)

    original_data.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    target_latent_tensor = None
    if args.target_image_path is not None:
        target_image_path = Path(args.target_image_path)
        assert target_image_path.is_file(), f"Target image path {target_image_path} does not exist"

        target_image = Image.open(target_image_path).convert("RGB").resize((args.resolution, args.resolution))
        target_image = np.array(target_image)[None].transpose(0, 3, 1, 2)

        target_image_tensor = torch.from_numpy(target_image).to("cuda", dtype=torch.float32) / 127.5 - 1.0
        target_latent_tensor = (
                vae.encode(target_image_tensor).latent_dist.sample().to(dtype=torch.bfloat16) * vae.config.scaling_factor
        )
        target_latent_tensor = target_latent_tensor.repeat(len(original_data), 1, 1, 1).cuda()

    f = [unet, text_encoder]

    long_term_momentum = torch.zeros_like(delta)
    context_adv = None

    all_loss_sum = []
    all_loss_unet = []
    all_loss_attn = []
    all_score_sum = []
    all_score_unet = []
    all_score_attn = []

    for i in range(args.max_train_steps):
        # 1. f' = f.clone()
        f_sur = copy.deepcopy(f)

        f_sur = train_one_epoch(
            args,
            f_sur,
            tokenizer,
            noise_scheduler,
            vae,
            clean_data,
            args.max_f_train_steps,
        )
        perturbed_data, long_term_momentum, delta, context_adv, loss_sum, loss_unet, loss_attn, score_sum, score_unet, score_attn = pgd_attack(
            args,
            f_sur,
            tokenizer,
            noise_scheduler,
            vae,
            perturbed_data,
            original_data,
            args.max_adv_train_steps,
            long_term_momentum,
            delta,
            context_adv
        )

        all_loss_sum.extend(loss_sum)
        all_loss_unet.extend(loss_unet)
        all_loss_attn.extend(loss_attn)
        all_score_sum.extend(score_sum)
        all_score_unet.extend(score_unet)
        all_score_attn.extend(score_attn)

        f = train_one_epoch(
            args,
            f,
            tokenizer,
            noise_scheduler,
            vae,
            perturbed_data,
            args.max_f_train_steps,
        )

        if (i + 1) % args.checkpointing_iterations == 0:
            save_folder = f"{args.output_dir}/noise-ckpt/{i + 1}"
            os.makedirs(save_folder, exist_ok=True)

            noise_save_folder = f"{args.output_dir}/noise-only-ckpt/{i + 1}"
            os.makedirs(noise_save_folder, exist_ok=True)

            noised_imgs = perturbed_data.detach()
            img_names = [
                str(instance_path).split("/")[-1]
                for instance_path in list(Path(args.instance_data_dir_for_adversarial).iterdir())
            ]
            for img_pixel, img_name, ori_img in zip(noised_imgs, img_names, original_data):
                if '.jpg' in img_name:
                    img_name = img_name.replace(".jpg", ".png")
                if '.JPEG' in img_name:
                    img_name = img_name.replace(".JPEG", ".png")
                save_path = os.path.join(save_folder, f"{i + 1}_noise_{img_name}")
                Image.fromarray(
                    (img_pixel * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                ).save(save_path)

                noise_save_path = os.path.join(noise_save_folder, f"{i + 1}_zaoyin_noise_{img_name}")
                noise = img_pixel - ori_img
                noise = (noise + torch.abs(torch.min(noise))) / (torch.abs(torch.max(noise)) + torch.abs(torch.min(noise))) * 255.0
                Image.fromarray(
                    noise.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                ).save(noise_save_path)

            print(f"Saved noise at step {i + 1} to {save_folder}")

    # np.save('ensemble_timestep_loss_sum.npy', all_loss_sum)
    # np.save('ensemble_timestep_loss_unet.npy', all_loss_unet)
    # np.save('ensemble_timestep_loss_attn.npy', all_loss_attn)
    # np.save('ensemble_timestep_score_sum.npy', all_score_sum)
    # np.save('ensemble_timestep_score_unet.npy', all_score_unet)
    # np.save('ensemble_timestep_score_attn.npy', all_score_attn)

if __name__ == "__main__":
    args = parse_args()
    main(args)
