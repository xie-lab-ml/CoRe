
import torch
import torch.nn as nn
import os
import json
import torch.nn.functional as F
import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from glob import glob
import math
from PIL import Image
device = torch.device('cuda')
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.utils import logging
from diffusers.models.embeddings import PatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.normalization import AdaLayerNormContinuous
from torchvision import transforms
    
def add_hook_to_module(model, module_name):
    outputs = []
    def hook(module, input, output):
        outputs.append(output)
    module = dict(model.named_modules()).get(module_name)
    if module is None:
        raise ValueError(f"can't find module {module_name}")
    hook_handle = module.register_forward_hook(hook)
    return hook_handle, outputs

class PromptSD35Net(nn.Module):

    def __init__(self,         
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 8,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        out_channels: int = 16,
        pos_embed_max_size: int = 192
        ):
        super().__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.out_channels = out_channels
        self.pos_embed_max_size = pos_embed_max_size
        self.inner_dim = self.num_attention_heads * self.attention_head_dim
        
        self.pos_embed = PatchEmbed(
            height=self.sample_size,
            width=self.sample_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size
        )
        
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                    ff_inner_dim=2*self.inner_dim   # mult should be 4 by default
                )
                for i in range(self.num_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)
        
        self.noise_shape = (1, 16, 128, 128) # (667, 4096)
        self.pre8_linear = nn.Sequential(nn.Linear(4096, 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, 1536))
        self.pre16_linear = nn.Sequential(nn.Linear(4096, 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, 1536))
        self.pre24_linear = nn.Sequential(nn.Linear(4096, 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, 1536))

        self.pre8_linear2 = nn.Sequential(nn.Linear(4096, 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, 1536))
        self.pre16_linear2 = nn.Sequential(nn.Linear(4096, 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, 1536))
        self.pre24_linear2 = nn.Sequential(nn.Linear(4096, 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, 1536))
        
        self.last_linear = nn.Sequential(nn.Linear(4096, 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, 1536))
        # self.last_linear2 = nn.Sequential(nn.Linear(667, 32))
        self.skip_connection2 = nn.Linear(4096, 1, bias=False)
        self.skip_connection = nn.Linear(667, 32, bias=False)
        self.trans_linear = nn.Linear(666+1+4096, 1536, bias=False)
        nn.init.constant_(self.skip_connection.weight.data, 0)
        nn.init.constant_(self.trans_linear.weight.data, 0)
        nn.init.constant_(self.trans_linear.weight.data, 0)
        nn.init.constant_(self.pre8_linear[-1].weight.data, 0)
        nn.init.constant_(self.pre16_linear[-1].weight.data, 0)
        nn.init.constant_(self.pre24_linear[-1].weight.data, 0)
        nn.init.constant_(self.pre8_linear2[-1].weight.data, 0)
        nn.init.constant_(self.pre16_linear2[-1].weight.data, 0)
        nn.init.constant_(self.pre24_linear2[-1].weight.data, 0)

    def forward(self, noise: torch.Tensor, _s, _v, _d, _pool_embedding) -> torch.Tensor:
        
        assert noise is not None
        _ori_v = _v.clone()
        _v = torch.stack([torch.diag(_v[jj]) for jj in range(_v.shape[0])], dim=0)
        positive_embedding = _s.permute(0, 2, 1) @ _v @ _d # [2, 64, 666] [2, 64] [2, 64, 4096]
        pool_embedding = _pool_embedding[:, None, :]
        embedding = torch.cat([positive_embedding, pool_embedding], dim=1)
        bs = noise.shape[0]
        height, width = noise.shape[-2:]
        embed_8 = embedding
        embed_16 = embedding
        embed_24 = embedding
        scale_8 = self.pre8_linear2(embed_8).mean(1)
        scale_16 = self.pre16_linear2(embed_16).mean(1)
        scale_24 = self.pre24_linear2(embed_24).mean(1)
        embed_8 = self.pre8_linear(embed_8).mean(1)
        embed_16 = self.pre16_linear(embed_16).mean(1)
        embed_24 = self.pre24_linear(embed_24).mean(1)
        embed_last = self.last_linear(embedding).mean(1)
        embed_trans = self.trans_linear(torch.cat([_s, _ori_v[...,None], _d], dim=2)).mean(1)
        skip_embedding = self.skip_connection(self.skip_connection2(embedding).permute(0,2,1)).permute(0,2,1)
        scale_skip, embed_skip = skip_embedding.chunk(2,dim=1)
        
        ori_noise = noise * (scale_skip[...,None]) + embed_skip[...,None]
        noise = self.pos_embed(noise)        
        noise = noise * (1 + scale_8[:, None, :] + embed_trans[:, None, :]) + embed_8[:, None, :]
        scale_list = [scale_16, scale_24]
        embed_list = [embed_16, embed_24]
        for _ii, block in enumerate(self.transformer_blocks):
            noise = block(noise)  
            if len(scale_list)!=0 and len(embed_list)!=0:
                noise = noise * (1 + scale_list[int(_ii//4)][:, None, :] + embed_trans[:, None, :]) + embed_list[int(_ii//4)][:, None, :]
        
        hidden_states = noise
        hidden_states = self.norm_out(hidden_states, embed_last)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.patch_size
        height = height // patch_size
        width = width // patch_size
    
        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )
        return output + ori_noise

    def weak_load_state_dict(self, state_dict: os.Mapping[str, torch.any], strict: bool = True, assign: bool = False):
        return load_filtered_state_dict(self, state_dict)

class PromptSDXLNet(nn.Module):

    def __init__(self,         
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 4,
        num_layers: int = 4,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        out_channels: int = 4,
        pos_embed_max_size: int = 192
        ):
        super().__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.out_channels = out_channels
        self.pos_embed_max_size = pos_embed_max_size
        self.inner_dim = self.num_attention_heads * self.attention_head_dim
        
        self.pos_embed = PatchEmbed(
            height=self.sample_size,
            width=self.sample_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size
        )
        
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                    ff_inner_dim=2*self.inner_dim   # mult should be 4 by default
                )
                for i in range(self.num_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)
        
        self.noise_shape = (1, 4, 128, 128) 
        self.pre8_linear = nn.Sequential(nn.Linear(2048, 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, 1536))
        self.pre16_linear = nn.Sequential(nn.Linear(2048, 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, 1536))
        self.pre24_linear = nn.Sequential(nn.Linear(2048, 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, 1536))

        self.pre8_linear2 = nn.Sequential(nn.Linear(2048, 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, 1536))
        self.pre16_linear2 = nn.Sequential(nn.Linear(2048, 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, 1536))
        self.pre24_linear2 = nn.Sequential(nn.Linear(2048, 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, 1536))
        
        self.last_linear = nn.Sequential(nn.Linear(2048, 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, 1536))
        # self.last_linear2 = nn.Sequential(nn.Linear(667, 32))
        self.skip_connection2 = nn.Linear(2048, 1, bias=False)
        self.skip_connection = nn.Linear(154+1, 8, bias=False)
        self.trans_linear = nn.Linear(154+1+2048, 1536, bias=False)
        self.pool_prompt_linear = nn.Linear(2560, 2048, bias=False)
        nn.init.constant_(self.skip_connection.weight.data, 0)
        nn.init.constant_(self.trans_linear.weight.data, 0)
        nn.init.constant_(self.trans_linear.weight.data, 0)
        nn.init.constant_(self.pre8_linear[-1].weight.data, 0)
        nn.init.constant_(self.pre16_linear[-1].weight.data, 0)
        nn.init.constant_(self.pre24_linear[-1].weight.data, 0)
        nn.init.constant_(self.pre8_linear2[-1].weight.data, 0)
        nn.init.constant_(self.pre16_linear2[-1].weight.data, 0)
        nn.init.constant_(self.pre24_linear2[-1].weight.data, 0)

    def forward(self, noise: torch.Tensor, _s, _v, _d, _pool_embedding) -> torch.Tensor:
        
        assert noise is not None
        _ori_v = _v.clone()
        _v = torch.stack([torch.diag(_v[jj]) for jj in range(_v.shape[0])], dim=0)
        positive_embedding = _s.permute(0, 2, 1) @ _v @ _d # [2, 64, 154] [2, 64] [2, 64, 2048]
        pool_embedding = self.pool_prompt_linear(_pool_embedding[:, None, :])
        embedding = torch.cat([positive_embedding, pool_embedding], dim=1)
        bs = noise.shape[0]
        height, width = noise.shape[-2:]
        embed_8 = embedding
        embed_16 = embedding
        embed_24 = embedding
        scale_8 = self.pre8_linear2(embed_8).mean(1)
        scale_16 = self.pre16_linear2(embed_16).mean(1)
        scale_24 = self.pre24_linear2(embed_24).mean(1)
        embed_8 = self.pre8_linear(embed_8).mean(1)
        embed_16 = self.pre16_linear(embed_16).mean(1)
        embed_24 = self.pre24_linear(embed_24).mean(1)
        embed_last = self.last_linear(embedding).mean(1)
        embed_trans = self.trans_linear(torch.cat([_s, _ori_v[...,None], _d], dim=2)).mean(1)
        skip_embedding = self.skip_connection(self.skip_connection2(embedding).permute(0,2,1)).permute(0,2,1)
        scale_skip, embed_skip = skip_embedding.chunk(2,dim=1)
        
        ori_noise = noise * (scale_skip[...,None]) + embed_skip[...,None]
        noise = self.pos_embed(noise)        
        noise = noise * (1 + scale_8[:, None, :] + embed_trans[:, None, :]) + embed_8[:, None, :]
        scale_list = [scale_16, scale_24]
        embed_list = [embed_16, embed_24]
        for _ii, block in enumerate(self.transformer_blocks):
            noise = block(noise)  
            if len(scale_list)!=0 and len(embed_list)!=0:
                noise = noise * (1 + scale_list[int(_ii//4)][:, None, :] + embed_trans[:, None, :]) + embed_list[int(_ii//4)][:, None, :]
        
        hidden_states = noise
        hidden_states = self.norm_out(hidden_states, embed_last)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.patch_size
        height = height // patch_size
        width = width // patch_size
    
        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )
        return output + ori_noise

    def weak_load_state_dict(self, state_dict: os.Mapping[str, torch.any], strict: bool = True, assign: bool = False):
        return load_filtered_state_dict(self, state_dict)


class NoisePromptDataset(Dataset):
    def __init__(self, if_weight=False):
        
        self.if_weight = if_weight
        json_list = glob('/home/xiedian/total_datacollect/json/*.json')
        self.original_score = []
        self.optim_score = []
        self.prompt = []
        self.noise_paths = []
        self.mask_conditions = []
        self.embeddings = []
        counter = 0
        for i in range(len(json_list)):
            with open('//home/xiedian/total_datacollect/json/new{:06d}.json'.format(i), 'r') as f:
                data = json.load(f)
            self.original_score.append(data['original_score_list'])
            self.optim_score.append(data['optimized_score_list'])
            if data['optimized_score_list']>data['original_score_list']:
                counter += 1
            self.prompt.append(data['caption'])
            self.noise_paths.append('/home/xiedian/total_datacollect/latents/{:06d}.pt'.format(i))
            self.embeddings.append('/data/xiedian/hg_reward/newdatacollect/total_datacollect_step_1_1/embedding/embeds_{:06d}.pt'.format(i))
            z = [0, 1] * ((512+77+77) // 2)
            self.mask_conditions.append(data['mid_token_ids'] if 'mid_token_ids' in data else z)
        # while counter * 2 > len(self.prompt):
        #     p = random.randint(0,len(self.prompt)-1)
        #     if self.original_score[p] >  self.optim_score[p]:
        #         self.optim_score.append(self.optim_score[p])
        #         self.original_score.append(self.original_score[p])
        #         self.mask_conditions.append(self.mask_conditions[p])
        #         self.noise_paths.append(self.noise_paths[p])
        #         self.prompt.append(self.prompt[p])
        
        # while counter * 2 < len(self.prompt):
        #     p = random.randint(0,len(self.prompt)-1)
        #     if self.original_score[p] >  self.optim_score[p]:
        #         self.optim_score.append(self.optim_score[p])
        #         self.original_score.append(self.original_score[p])
        #         self.mask_conditions.append(self.mask_conditions[p])
        #         self.noise_paths.append(self.noise_paths[p])
        #         self.prompt.append(self.prompt[p])
        
        self.original_score = torch.Tensor(self.original_score)
        self.optim_score = torch.Tensor(self.optim_score)
                
    def __len__(self):
        return len(self.prompt)

    def __getitem__(self, index):
        try:
            noise = torch.load(self.noise_paths[index], map_location='cpu').squeeze(0).float()
            noise_pred_uncond, mid_noise_pred, noise_pred_text = noise.chunk(3,dim=0)
            prompt = self.prompt[index]
            original_score = self.original_score[index]
            optim_score = self.optim_score[index]
            embedding = torch.load(self.embeddings[index], map_location='cpu')
            _s, _v, _d, _pool_embedding = embedding['_s'], embedding['_v'], embedding['_d'], embedding['pooled_prompt_embeds']
            _s = _s.detach().float()
            _v = _v.detach().float()
            _d = _d.detach().float()
            _pool_embedding = _pool_embedding.detach().float()
            if original_score > optim_score:
                noise_pred = noise_pred_uncond + 4.5 * (noise_pred_text - noise_pred_uncond)
            else:
                guidance_scale = 4.5 * 1.6
                diff_text = torch.norm(noise_pred_text - noise_pred_uncond)
                mid_guidance_scale = (diff_text / torch.norm(noise_pred_text - mid_noise_pred)).item()
                guidance_scale_mid = guidance_scale / (2.4 + 1)
                guidance_scale_all = guidance_scale * 2.4 / (2.4 + 1)
                all_mid = (noise_pred_text - mid_noise_pred) * mid_guidance_scale
                all_null = noise_pred_text - noise_pred_uncond
                noise_pred = all_mid * guidance_scale_mid + all_null * guidance_scale_all + (mid_noise_pred + noise_pred_uncond) / 2
        except:
            print("error", index)
            return self.__getitem__((index+1)%len(self.prompt))
        if self.if_weight:
            return noise_pred_text, prompt, noise_pred, 2 / (1+ math.exp((-abs(original_score) + 26.5)/1.7)), _s, _v, _d, _pool_embedding
        else:
            return noise_pred_text, prompt, noise_pred, _s, _v, _d, _pool_embedding

class NoisePromptDataset_2_0(Dataset):
    def __init__(self, if_weight=False):
        
        self.if_weight = if_weight
        json_list = glob('/data/xiedian/hg_reward/newdatacollect/total_datacollect_step_1_1/json/*.json')
        self.original_score = []
        self.quick_score = []
        self.slow_score = []
        self.prompt = []
        self.noise_paths = []
        self.mask_conditions = []
        self.img_list = []
        self.embeddings = []
        
        counter = 0
        for i in range(len(json_list)):
            with open('/data/xiedian/hg_reward/newdatacollect/total_datacollect_step_1_1/json/new{:06d}.json'.format(i), 'r') as f:
                data = json.load(f)
            if (not os.path.exists('/data/xiedian/hg_reward/newdatacollect/total_datacollect_step_1_1/latents/{:06d}.pt'.format(i))) or \
                max(data['original_score_list'], data['quick_score_list'], data['slow_score_list']) != data['original_score_list']:
                continue
            self.original_score.append(data['original_score_list'])
            self.quick_score.append(data['quick_score_list'])
            self.slow_score.append(data['slow_score_list'])
            self.prompt.append(data['caption'])
            self.noise_paths.append('/data/xiedian/hg_reward/newdatacollect/total_datacollect_step_1_1/latents/{:06d}.pt'.format(i))
            z = [0, 1] * ((512+77+77) // 2)
            self.mask_conditions.append(data['mid_token_ids'] if 'mid_token_ids' in data else z)
            if data['original_score_list'] >= max(data['quick_score_list'], data['slow_score_list']):
                self.img_list.append('/data/xiedian/hg_reward/newdatacollect/total_datacollect_step_1_1/optim/original{:06d}.png'.format(i))
            elif data['quick_score_list'] >= max(data['original_score_list'], data['slow_score_list']):
                self.img_list.append('/data/xiedian/hg_reward/newdatacollect/total_datacollect_step_1_1/optim/quick{:06d}.png'.format(i))
            else:
                self.img_list.append('/data/xiedian/hg_reward/newdatacollect/total_datacollect_step_1_1/optim/slow{:06d}.png'.format(i))
            self.embeddings.append('/home/xiedian/total_datacollect/embedding/embeds_{:06d}.pt'.format(i))
        self.original_score = torch.Tensor(self.original_score)
        self.quick_score = torch.Tensor(self.quick_score)
        self.slow_score = torch.Tensor(self.slow_score)
            
    def __len__(self):
        return len(self.prompt)

    def __getitem__(self, index):
        try:
            original_score = self.original_score[index]
            quick_score = self.quick_score[index]
            slow_score = self.slow_score[index]
            original_score = max(max(quick_score, slow_score), original_score)
            embedding = torch.load(self.embeddings[index], map_location='cpu')
            _s, _v, _d, _pool_embedding = embedding['_s'], embedding['_v'], embedding['_d'], embedding['pooled_prompt_embeds']
            _s = _s.detach().float()
            _v = _v.detach().float()
            _d = _d.detach().float()
            _pool_embedding = _pool_embedding.detach().float()
            noise = torch.load(self.noise_paths[index], map_location='cpu').squeeze(0).float()
            noise_pred_text, noise_pred = noise.chunk(2,dim=0)
            prompt = self.prompt[index]
        except:
            print("error", index)
            return self.__getitem__((index+1)%len(self.prompt))
        if self.if_weight:
            return noise_pred_text, prompt, noise_pred, 2 / (1+ math.exp((-abs(original_score) + 26.5)/1.7)), _s, _v, _d, _pool_embedding
        else:
            return noise_pred_text, prompt, noise_pred, _s, _v, _d, _pool_embedding

class NoisePromptDataset_3_0(Dataset):
    def __init__(self, if_weight=False):
        
        self.if_weight = if_weight
        json_list = glob('/data/xiedian/hg_reward/CFG_TOTAL/total_datacollect/json/*.json')
        self.score = []
        self.prompt = []
        self.noise_paths = []
        self.mask_conditions = []
        self.img_list = []
        self.embeddings = []
        
        print(len(json_list))
        
        for i in range(len(json_list)):
            if (not os.path.exists('/data/xiedian/hg_reward/CFG_TOTAL/total_datacollect/embedding/{:06d}.pt'.format(i))):
                continue
            
            with open('/data/xiedian/hg_reward/CFG_TOTAL/total_datacollect/json/new{:06d}.json'.format(i), 'r') as f:
                data = json.load(f)
            if data['original_score_list'] > data['optimized_score_list']:
                tag = 0
                if (not os.path.exists('/data/xiedian/hg_reward/CFG_TOTAL/total_datacollect/latents/original{:06d}.pt'.format(i))):
                    continue
            else:
                tag = 1
                if (not os.path.exists('/data/xiedian/hg_reward/CFG_TOTAL/total_datacollect/latents/new{:06d}.pt'.format(i))):
                    continue
            
            if tag == 1:
                self.score.append(data['optimized_score_list'])
                self.noise_paths.append('/data/xiedian/hg_reward/CFG_TOTAL/total_datacollect/latents/new{:06d}.pt'.format(i))
            else:
                self.score.append(data['original_score_list'])
                self.noise_paths.append('/data/xiedian/hg_reward/CFG_TOTAL/total_datacollect/latents/original{:06d}.pt'.format(i))
            self.prompt.append(data['caption'])
            self.embeddings.append('/data/xiedian/hg_reward/CFG_TOTAL/total_datacollect/embedding/{:06d}.pt'.format(i))
        self.score = torch.Tensor(self.score)
            
    def __len__(self):
        return len(self.prompt)

    def __getitem__(self, index):
        try:
            embedding = torch.load(self.embeddings[index], map_location='cpu')
            _s, _v, _d, _pool_embedding = embedding['_s'], embedding['_v'], embedding['_d'], embedding['_pooled_prompt_embeds']
            _s = _s.detach().float()
            _v = _v.detach().float()
            _d = _d.detach().float()
            _pool_embedding = _pool_embedding.detach().float()
            noise = torch.load(self.noise_paths[index], map_location='cpu').float() # [2XT, 16, 128, 128]
            prompt = self.prompt[index] # [ori, target, ori]
            score = self.score[index]
        except:
            print("error", index)
            return self.__getitem__((index+1)%len(self.prompt))
        if self.if_weight:
            return noise, prompt, 2 / (1+ math.exp((-abs(score) + 26.5)/1.7)), _s, _v, _d, _pool_embedding
        else:
            return noise, prompt, _s, _v, _d, _pool_embedding


def load_filtered_state_dict(model, state_dict):
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            if model_state_dict[k].size() == v.size():
                filtered_state_dict[k] = v
            else:
                print(f"Skipping {k}: shape mismatch ({model_state_dict[k].size()} vs {v.size()})")
        else:
            print(f"Skipping {k}: not found in model's state_dict.")
    model.load_state_dict(filtered_state_dict, strict=False)
    return model

def custom_collate_fn_2_0(batch):
    noise_pred_texts, prompts, noise_preds, max_scores = zip(*batch)
    
    noise_pred_texts = torch.stack(noise_pred_texts)
    noise_preds = torch.stack(noise_preds)
    max_scores = torch.stack(max_scores)
    
    return noise_pred_texts, prompts, noise_preds, max_scores


if __name__ == "__main__":
    dataset = NoisePromptDataset(if_weight=True)
    weights = []
    for i, (noise, prompt, gt, weight) in enumerate(dataset):
        weights.append(weight)
    weights = torch.from_numpy(np.array(weights)).cuda()
    print(weights.mean(), weights.std(dim=0))