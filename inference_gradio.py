import argparse
import gradio as gr
from diffusers import DiffusionPipeline
import torch
import os
from utils import insert_sd_klora_to_unet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="",
        help="Pretrained model path",
    )
    parser.add_argument(
        "--lora_name_or_path_content",
        type=str,
        help="LoRA path",
        default="",
    )
    parser.add_argument(
        "--lora_name_or_path_style",
        type=str,
        help="LoRA path",
        default="",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Pattern for the image generation",
        default="s",
    )
    return parser.parse_args()


args = parse_args()
pattern = args.pattern
if pattern == "s*":
    alpha = 1.5
    beta = alpha * 0.85
else:
    alpha = 1.5
    beta = 0.5
    
sum_timesteps = 28000

args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path)
pipe.unet = insert_sd_klora_to_unet(
    pipe.unet, args.lora_name_or_path_content, args.lora_name_or_path_style, alpha, beta, sum_timesteps 
)
pipe.to(device, dtype=torch.float16)


def run(prompt: str):
    generator = torch.Generator(device=device).manual_seed(42)
    image = pipe(prompt=prompt, generator=generator).images[0]
    return image


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt = gr.Text(label="prompt", value="a sbu cat in szn style")
            bttn = gr.Button(value="Run")
        with gr.Column():
            out = gr.Image(label="out")
    prompt.submit(fn=run, inputs=[prompt], outputs=[out])
    bttn.click(fn=run, inputs=[prompt], outputs=[out])
    demo.launch()
