import json
import numpy as np
import math
import csv
import random
import argparse
import torch
import os
import torch.distributed as dist

from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

from accelerate.utils import set_seed

from diffusion_pipeline.sd35_pipeline import StableDiffusion3Pipeline, FlowMatchEulerInverseScheduler
from diffusion_pipeline.sdxl_pipeline import StableDiffusionXLPipeline
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import FlowMatchEulerDiscreteScheduler, DDIMInverseScheduler, DDIMScheduler

device = torch.device('cuda')

def get_args():
      # pick: test_unique_caption_zh.csv       draw: drawbench.csv
      parser = argparse.ArgumentParser()
      parser.add_argument("--model", default='sd35', choices=['sdxl', 'sd35'], type=str)
      parser.add_argument("--inference-step", default=30, type=int)
      parser.add_argument("--size", default=1024, type=int)
      parser.add_argument("--seed", default=33, type=int)
      parser.add_argument("--cfg", default=3.5, type=float)

      # hyperparameters for Z-Sampling
      parser.add_argument("--inv-cfg", default=0.5, type=float)

      # hyperparameters for Z-Core^2
      parser.add_argument("--w2s-guidance", default=1.5, type=float)
      parser.add_argument("--end_timesteps", default=28, type=int) # equal to inference step - 2 or inference step


      parser.add_argument("--prompt", default='Mickey Mouse painting by Frank Frazetta.', type=str)

      parser.add_argument("--method", default='standard', choices=['standard', 'core', 'zigzag', 'z-core'], type=str)

      args =  parser.parse_args()
      return args


if __name__ == '__main__':
      torch.cuda.empty_cache()
      dtype = torch.float16
      args = get_args()
      print("args.seed: ", args.seed)
      set_seed(args.seed)

      # TODO: load pipeline
      if args.model == 'sd35':
            nf4_config = BitsAndBytesConfig(
                  load_in_4bit=True,
                  bnb_4bit_quant_type="nf4",
                  bnb_4bit_compute_dtype=torch.bfloat16
            )
            model_nf4 = SD3Transformer2DModel.from_pretrained(
                  "stabilityai/stable-diffusion-3.5-large",
                  subfolder="transformer",
                  quantization_config=nf4_config,
                  torch_dtype=torch.bfloat16
            )

            pipe = StableDiffusion3Pipeline.from_pretrained(
                  "stabilityai/stable-diffusion-3.5-large", 
                  transformer=model_nf4,
                  torch_dtype=torch.bfloat16,
            )

            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
            inverse_scheduler = FlowMatchEulerInverseScheduler.from_pretrained("stabilityai/stable-diffusion-3.5-large",
                                                                  subfolder='scheduler')
            pipe.inv_scheduler = inverse_scheduler

      elif args.model == "sdxl":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                  "stabilityai/stable-diffusion-xl-base-1.0", 
                  torch_dtype=torch.float16, 
                  variant="fp16",
                  use_safetensors=True
            ).to("cuda")

            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            inverse_scheduler = DDIMInverseScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                                  subfolder='scheduler')
            pipe.inv_scheduler = inverse_scheduler

      pipe.to(device)
      pipe.enable_model_cpu_offload()

      # TODO: load noise model
      if args.method == 'core' or args.method == 'z-core':
            from diffusion_pipeline.refine_model import PromptSD35Net, PromptSDXLNet
            from diffusion_pipeline.lora import replace_linear_with_lora, lora_true

            if args.model == 'sd35':
                  refine_model = PromptSD35Net()
                  replace_linear_with_lora(refine_model, rank=64, alpha=1.0, number_of_lora=28)
                  lora_true(refine_model, lora_idx=0)

                  checkpoint = torch.load('./weights/sd35_ckpt_v9.pth', map_location='cpu')
                  refine_model.load_state_dict(checkpoint)
            elif args.model == 'sdxl':
                  refine_model = PromptSDXLNet()
                  replace_linear_with_lora(refine_model, rank=48, alpha=1.0, number_of_lora=50)
                  lora_true(refine_model, lora_idx=0)

                  checkpoint = torch.load('./weights/sdxl_ckpt_v9.pth', map_location='cpu')
                  refine_model.load_state_dict(checkpoint)

            print("Load Lora Success")
            refine_model = refine_model.to(device)
            refine_model = refine_model.to(torch.bfloat16)

      
      # TODO: load hyperparameters
      size = args.size
      if args.model == 'sdxl':
            shape = (1, 4, size // 8, size // 8)
      else:
            shape = (1, 16, size // 8, size // 8)

      num_steps = args.inference_step
      end_timesteps = args.end_timesteps
      guidance_scale = args.cfg
      w2s_guidance = args.w2s_guidance
      inv_cfg = args.inv_cfg
      prompt = args.prompt

      print("pass this prompt: ", prompt)
    
      start_latents = torch.randn(shape, dtype=dtype).to(device)
        
      if args.model == 'sdxl':
            if args.method == 'core':
                  output = pipe.core(
                        prompt=prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_steps,
                        latents=start_latents,
                        return_dict=False,
                        refine_model=refine_model,
                        lora_true=lora_true,
                        end_timesteps=end_timesteps,
                        w2s_guidance=w2s_guidance)[0][0]  
            
            elif args.method == 'zigzag':
                  output = pipe.zigzag(
                        prompt=prompt,
                        guidance_scale=guidance_scale,
                        latents=start_latents,
                        return_dict=False,
                        num_inference_steps=num_steps,
                        inv_cfg=inv_cfg)[0][0] 

            elif args.method == 'z-core':
                  output = pipe.z_core(
                        prompt=prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_steps,
                        latents=start_latents,
                        return_dict=False,
                        refine_model=refine_model,
                        lora_true=lora_true,
                        end_timesteps=end_timesteps,
                        w2s_guidance=w2s_guidance,
                        inv_cfg=inv_cfg)[0][0]  
                  
            elif args.method == 'standard':
                  output = pipe(
                        prompt=prompt,
                        guidance_scale=guidance_scale,
                        latents=start_latents,
                        return_dict=False,
                        num_inference_steps=num_steps)[0][0] 
            else:
                  raise ValueError("Invalid method")
            
            output.save(f'{args.model}_{args.method}.png')  


      else: 
            if args.method == 'core':
                  output = pipe.core(
                        prompt=prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_steps,
                        latents=start_latents,
                        max_sequence_length=512,
                        return_dict=False,
                        refine_model=refine_model,
                        lora_true=lora_true,
                        end_timesteps=end_timesteps,
                        w2s_guidance=w2s_guidance)[0][0]  
            
            elif args.method == 'zigzag':
                  output = pipe.zigzag(
                        prompt=prompt,
                        max_sequence_length=512,
                        guidance_scale=guidance_scale,
                        latents=start_latents,
                        return_dict=False,
                        num_inference_steps=num_steps,
                        inv_cfg=inv_cfg)[0][0] 

            elif args.method == 'z-core':
                  output = pipe.z_core(
                        prompt=prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_steps,
                        latents=start_latents,
                        return_dict=False,
                        max_sequence_length=512,
                        refine_model=refine_model,
                        lora_true=lora_true,
                        end_timesteps=end_timesteps,
                        w2s_guidance=w2s_guidance)[0][0]   
                  
            elif args.method == 'standard':
                  output = pipe(
                        prompt=prompt,
                        guidance_scale=guidance_scale,
                        latents=start_latents,
                        return_dict=False,
                        max_sequence_length=512,
                        num_inference_steps=num_steps)[0][0] 
            else:
                  raise ValueError("Invalid method")
            
            output.save(f'{args.model}_{args.method}.png')  

    