import torch.nn as nn
import torch
import os
import random
import numpy as np

def register_time(model, t):
    # 下采样层**********
    down_attn_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}  # ，没有第四层了
    for res in down_attn_dict:
        for block in down_attn_dict[res]:
            module1 = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1  # 自注意力层
            module2 = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2  # 交叉注意力层
            setattr(module1, 't', t)
            setattr(module2, 't', t)
    # 下采样层**********

    # 中间层***********
    module1 = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1  # 自注意力层
    module2 = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2  # 交叉注意力层
    setattr(module1, 't', t)
    setattr(module2, 't', t)
    # 中间层***********

    # 上采样层**********
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module1 = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module2 = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module1, 't', t)
            setattr(module2, 't', t)
    # 上采样层**********


def load_source_latents_t(t, latents_path):
    latents_t_path = os.path.join(latents_path, f'noisy_latents_{t}.pt')
    assert os.path.exists(latents_t_path), f'Missing latents at t {t} path {latents_t_path}'
    latents = torch.load(latents_t_path)
    return latents


def register_attention_control_efficient(model, injection_schedule):
    def sa_forward(self):

        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        self.attn = None
        self.keyword_loss = 0

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x

            if is_cross:  # cross-attention
                # print("Do CA")
                q = self.to_q(x)
                q = self.head_to_batch_dim(q)

                k = self.to_k(encoder_hidden_states)
                v = self.to_v(encoder_hidden_states)

                k = self.head_to_batch_dim(k)
                v = self.head_to_batch_dim(v)

                sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
                if attention_mask is not None:
                    attention_mask = attention_mask.reshape(batch_size, -1)
                    max_neg_value = -torch.finfo(sim.dtype).max
                    attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                    sim.masked_fill_(~attention_mask, max_neg_value)
                # attention, what we cannot get enough of
                attn = sim.softmax(dim=-1)

                out = torch.einsum("b i j, b j d -> b i d", attn, v)
                out = self.batch_to_head_dim(out)
                out = to_out(out)
                self.attn = out

            else:  # self-attention
                # print("Do SA")
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                v = self.to_v(encoder_hidden_states)

                q_base = self.head_to_batch_dim(q)
                k_base = self.head_to_batch_dim(k)
                v_base = self.head_to_batch_dim(v)
                sim_base = torch.einsum("b i d, b j d -> b i j", q_base, k_base) * self.scale

                if attention_mask is not None:
                    attention_mask = attention_mask.reshape(batch_size, -1)
                    max_neg_value = -torch.finfo(sim_base.dtype).max
                    attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                    sim_base.masked_fill_(~attention_mask, max_neg_value)
                attn_base = sim_base.softmax(dim=-1)
                now_attn = attn_base

                out = torch.einsum("b i j, b j d -> b i d", now_attn, v_base)
                out = self.batch_to_head_dim(out)
                out = to_out(out)
                self.attn = out

            return out

        return forward

    # 下采样层**********
    down_attn_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    for res in down_attn_dict:
        for block in down_attn_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)
            module2 = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            module2.forward = sa_forward(module2)
            setattr(module2, 'injection_schedule', injection_schedule)
    # 下采样层**********

    # 中间层***********
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    module.forward = sa_forward(module)
    setattr(module, 'injection_schedule', injection_schedule)
    module2 = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    module2.forward = sa_forward(module2)
    setattr(module2, 'injection_schedule', injection_schedule)
    # 中间层***********

    # 上采样层********
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)
            module2 = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            module2.forward = sa_forward(module2)
            setattr(module2, 'injection_schedule', injection_schedule)
    # 上采样层********


def register_conv_control_efficient(model, injection_schedule):
    def conv_forward(self):
        self.conv_feature = None
        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)
            hidden_states = self.conv1(hidden_states)
            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = self.nonlinearity(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)
            self.conv_feature = hidden_states
            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    down_layers_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1], 3: [0, 1]}
    up_layers_dict = {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}


    # 下采样层**********
    for res in down_layers_dict:
        for block in down_layers_dict[res]:
            module = model.unet.down_blocks[res].resnets[block]
            module.forward = conv_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)
    # 下采样层**********
    # 上采样层**********
    for res in up_layers_dict:
        for block in up_layers_dict[res]:
            module = model.unet.up_blocks[res].resnets[block]
            module.forward = conv_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)
    # 上采样层**********


class AttentionLoader(nn.Module):
    def __init__(self, device, vae, tokenizer, text_encoder, unet, scheduler):
        super().__init__()
        # self.config = config
        self.device = device

        # Create SD models
        # print('Loading SD model')
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.scheduler = scheduler
        # print('SD model loaded')

    # @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    def init_(self, current_timesteps, pnp_attn_t):
        idx = torch.randperm(current_timesteps.nelement())
        rand_t = current_timesteps.view(-1)[idx].view(current_timesteps.size())
        qk_injection_t = int(len(rand_t) * pnp_attn_t)
        self.qk_injection_timesteps = rand_t[:qk_injection_t] if qk_injection_t >= 0 else []

        register_attention_control_efficient(self, self.qk_injection_timesteps)
        register_conv_control_efficient(self, self.qk_injection_timesteps)
