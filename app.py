import json
import numpy as np
import math
import csv
import random
import argparse
import torch
import os
import torch.distributed as dist
import gradio as gr
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
import spaces
from accelerate.utils import set_seed

from diffusion_pipeline.sd35_pipeline import StableDiffusion3Pipeline, FlowMatchEulerInverseScheduler
from diffusion_pipeline.sdxl_pipeline import StableDiffusionXLPipeline
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import FlowMatchEulerDiscreteScheduler, DDIMInverseScheduler, DDIMScheduler
from huggingface_hub import login
import os
login(token=os.getenv('HF_TOKEN'))
device = torch.device('cuda')



# Load models outside the function to avoid reloading every time
def load_models():
    # Load sd35 model
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
    pipe_sd35 = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", 
        transformer=model_nf4,
        torch_dtype=torch.bfloat16,
    )
    pipe_sd35.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe_sd35.scheduler.config)
    inverse_scheduler_sd35 = FlowMatchEulerInverseScheduler.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        subfolder='scheduler'
    )
    pipe_sd35.inv_scheduler = inverse_scheduler_sd35

    # Load sdxl model
    pipe_sdxl = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    pipe_sdxl.scheduler = DDIMScheduler.from_config(pipe_sdxl.scheduler.config)
    inverse_scheduler_sdxl = DDIMInverseScheduler.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder='scheduler'
    )
    pipe_sdxl.inv_scheduler = inverse_scheduler_sdxl

    return pipe_sd35, pipe_sdxl

pipe_sd35, pipe_sdxl = load_models()

@spaces.GPU
def generate_image(
    model_name,
    seed,
    num_steps,
    guidance_scale,
    inv_cfg,
    w2s_guidance,
    end_timesteps,
    prompt,
    method,
    size,
):
    try:
        # 根据传入的参数生成图像
        torch.cuda.empty_cache()
        dtype = torch.float16
        set_seed(seed)

        # Select the appropriate pipeline
        if model_name == 'sd35':
            pipe = pipe_sd35
        elif model_name == 'sdxl':
            pipe = pipe_sdxl
        else:
            raise ValueError("Invalid model name")

        pipe.to(device)
        pipe.enable_model_cpu_offload()
        os.system('huggingface-cli download sst12345/CoRe2 weights/sd35_noise_model.pth weights/sdxl_noise_model.pth --local-dir ./weights')
        # TODO: load noise model
        if method == 'core' or method == 'z-core':
            from diffusion_pipeline.refine_model import PromptSD35Net, PromptSDXLNet
            from diffusion_pipeline.lora import replace_linear_with_lora, lora_true

            if model_name == 'sd35':
                refine_model = PromptSD35Net()
                replace_linear_with_lora(refine_model, rank=64, alpha=1.0, number_of_lora=28)
                lora_true(refine_model, lora_idx=0)
                checkpoint = torch.load('./weights/weights/sd35_noise_model.pth', map_location='cpu')
                refine_model.load_state_dict(checkpoint)
            elif model_name == 'sdxl':
                refine_model = PromptSDXLNet()
                replace_linear_with_lora(refine_model, rank=48, alpha=1.0, number_of_lora=50)
                lora_true(refine_model, lora_idx=0)
                checkpoint = torch.load('./weights/weights/sdxl_noise_model.pth', map_location='cpu')
                refine_model.load_state_dict(checkpoint)

            refine_model = refine_model.to(torch.bfloat16)
            refine_model = refine_model.to(device)
            print("Load Lora Success")
        # 根据模型类型设置形状
        if model_name == 'sdxl':
            shape = (1, 4, size // 8, size // 8)
        else:
            shape = (1, 16, size // 8, size // 8)
        
        start_latents = torch.randn(shape, dtype=dtype).to(device)
        
        # 根据方法选择生成图像
        if model_name == 'sdxl':
            if method == 'core':
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
            elif method == 'zigzag':
                output = pipe.zigzag(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    latents=start_latents,
                    return_dict=False,
                    num_inference_steps=num_steps,
                    inv_cfg=inv_cfg)[0][0]
            elif method == 'z-core':
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
            elif method == 'standard':
                output = pipe(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    latents=start_latents,
                    return_dict=False,
                    num_inference_steps=num_steps)[0][0]
            else:
                raise ValueError("Invalid method")
        else:
            if method == 'core':
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
            elif method == 'zigzag':
                output = pipe.zigzag(
                    prompt=prompt,
                    max_sequence_length=512,
                    guidance_scale=guidance_scale,
                    latents=start_latents,
                    return_dict=False,
                    num_inference_steps=num_steps,
                    inv_cfg=inv_cfg)[0][0]
            elif method == 'z-core':
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
            elif method == 'standard':
                output = pipe(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    latents=start_latents,
                    return_dict=False,
                    max_sequence_length=512,
                    num_inference_steps=num_steps)[0][0]
            else:
                raise ValueError("Invalid method")

        # 将生成的图像保存为临时文件并返回
        output_path = f'{model_name}_{method}.png'
        output.save(output_path)
        return output_path

    except Exception as e:
        print(f"An error occurred: {e}")
        return None



if __name__ == '__main__':
      # 创建Gradio接口
      iface = gr.Interface(
      fn=generate_image,
      inputs=[
            gr.Dropdown(choices=['sdxl', 'sd35'], value='sdxl', label="Model"),  # 设置默认模型为 'sdxl'
            gr.Slider(minimum=1, maximum=1000000, value=1, label="seed"),  # 设置默认种子为 1
            gr.Slider(minimum=1, maximum=100, value=50, label="Inference Steps"),  # 设置默认推理步数为 50
            gr.Slider(minimum=1, maximum=10, value=5.5, label="CFG"),  # 设置默认CFG为 5.5
            gr.Slider(minimum=-10, maximum=10, value=-1, label="Inverse CFG"),  # 设置默认逆CFG为 -1
            gr.Slider(minimum=1, maximum=3.5, value=2.5, label="W2S Guidance"),  # 设置默认W2S指导为 2.5
            gr.Slider(minimum=1, maximum=100, value=50, label="End Timesteps"),  # 设置默认结束时间步为 50
            gr.Textbox(label="Prompt"),  # 文本框没有默认值
            gr.Dropdown(choices=['standard', 'core', 'zigzag', 'z-core'], value='core', label="Method"),  # 设置默认方法为 'core'
            gr.Slider(minimum=1024, maximum=2048, value=1024, label="Size")  # 设置默认大小为 1024
      ],
      outputs=gr.Image(type="filepath"),  # 修改了type参数
      title="Image Generation with CoRe^2",
      )
      iface.launch(share=True)
    