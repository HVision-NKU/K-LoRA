import argparse
from diffusers import DiffusionPipeline
import torch
import os
from utils import insert_sd_klora_to_unet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/ubuntu/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/",
        help="Pretrained model path",
    )
    parser.add_argument(
        "--lora_name_or_path_content",
        type=str,
        help="LoRA path",
        default="loraDataset/content_6/pytorch_lora_weights.safetensors",
    )
    parser.add_argument(
        "--lora_name_or_path_style",
        type=str,
        help="LoRA path",
        default="loraDataset/style_9/pytorch_lora_weights.safetensors",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Output folder path",
        default="output",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for the image generation",
        default="a sbu cat in szn style",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Pattern for the image generation",
        default="s*",
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


device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path)
pipe.unet = insert_sd_klora_to_unet(
    pipe.unet, args.lora_name_or_path_content, args.lora_name_or_path_style, alpha, beta, sum_timesteps, pattern
)
pipe.to(device, dtype=torch.float16)

def run():
    seeds = list(range(40))
    seeds = [see for see in seeds]

    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(prompt=args.prompt, generator=generator).images[0]
        output_path = os.path.join(args.output_folder, f"output_image_{index}.png")
        image.save(output_path)
        print(output_path)


if __name__ == "__main__":
    run()
