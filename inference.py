import argparse
from diffusers import DiffusionPipeline
import torch
import os
from utils import insert_klora_to_unet


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
        "--output_folder",
        type=str,
        help="Output folder path",
        default="Output/",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for the image generation",
        default="",
    )
    return parser.parse_args()


args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path)
pipe.unet = insert_klora_to_unet(
    pipe.unet, args.lora_name_or_path_content, args.lora_name_or_path_style
)
pipe.to(device, dtype=torch.float16)

def run():
    seeds = list(range(40))

    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(prompt=args.prompt, generator=generator).images[0]
        output_path = os.path.join(args.output_folder, f"output_image_{index}.png")
        image.save(output_path)


if __name__ == "__main__":
    run()
