import os
from typing import Optional, Dict
from huggingface_hub import hf_hub_download

import torch
from safetensors import safe_open
from diffusers import UNet2DConditionModel
from diffusers.loaders.lora import LORA_WEIGHT_NAME_SAFE
from klora import KLoRALinearLayer, KLoRALinearLayerInference


def get_lora_weights(
    lora_name_or_path: str, subfolder: Optional[str] = None, **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Args:
        lora_name_or_path (str): huggingface repo id or folder path of lora weights
        subfolder (Optional[str], optional): sub folder. Defaults to None.
    """
    if os.path.exists(lora_name_or_path):
        if subfolder is not None:
            lora_name_or_path = os.path.join(lora_name_or_path, subfolder)
        if os.path.isdir(lora_name_or_path):
            lora_name_or_path = os.path.join(lora_name_or_path, LORA_WEIGHT_NAME_SAFE)
    else:
        lora_name_or_path = hf_hub_download(
            repo_id=lora_name_or_path,
            filename=LORA_WEIGHT_NAME_SAFE,
            subfolder=subfolder,
            **kwargs,
        )
    assert lora_name_or_path.endswith(
        ".safetensors"
    ), "Currently only safetensors is supported"
    tensors = {}
    with safe_open(lora_name_or_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def merge_lora_weights(
    tensors: torch.Tensor, key: str, prefix: str = "unet.unet."
) -> Dict[str, torch.Tensor]:
    """
    Args:
        tensors (torch.Tensor): state dict of lora weights
        key (str): target attn layer's key
        prefix (str, optional): prefix for state dict. Defaults to "unet.unet.".
    """
    target_key = prefix + key
    out1 = {}
    out2 = {}
    for part in ["to_q", "to_k", "to_v", "to_out.0"]:
        down_key = target_key + f".{part}.lora.down.weight"
        up_key = target_key + f".{part}.lora.up.weight"
        out1[part] = tensors[up_key]
        out2[part] = tensors[down_key]
    return out1, out2


def initialize_klora_layer(
    average_ratio,
    state_dict_1_a,
    state_dict_1_b,
    state_dict_2_a,
    state_dict_2_b,
    part,
    **model_kwargs,
):
    klora_layer = KLoRALinearLayer(
        average_ratio=average_ratio,
        weight_1_a=state_dict_1_a[part],
        weight_1_b=state_dict_1_b[part],
        weight_2_a=state_dict_2_a[part],
        weight_2_b=state_dict_2_b[part],
        **model_kwargs,
    )
    return klora_layer


def get_ratio_between_content_and_style(lora_weights_content, lora_weights_style):
    comparison_results = []
    layer_names = list(lora_weights_content.keys())

    for i in range(0, len(layer_names), 2):
        layer_name_up = layer_names[i + 1]
        layer_name_down = layer_names[i]

        tensor_content_up = lora_weights_content[layer_name_up]
        tensor_content_down = lora_weights_content[layer_name_down]
        tensor_style_up = lora_weights_style[layer_name_up]
        tensor_style_down = lora_weights_style[layer_name_down]

        tensor_content_product = tensor_content_up @ tensor_content_down
        tensor_style_product = tensor_style_up @ tensor_style_down

        tensor_content_product = torch.abs(tensor_content_product)
        tensor_style_product = torch.abs(tensor_style_product)
        max_x_sum_content = tensor_content_product.sum().item()
        max_x_sum_style = tensor_style_product.sum().item()

        if max_x_sum_style != 0:
            ratio = max_x_sum_content / max_x_sum_style
        else:
            ratio = float("inf")

        comparison_results.append(ratio)
        average_ratio = (
            sum(comparison_results) / len(comparison_results)
            if len(comparison_results) > 0
            else float("inf")
        )
    return average_ratio


def insert_klora_to_unet(unet, lora_weights_content_path, lora_weights_style_path):
    lora_weights_content = get_lora_weights(lora_weights_content_path)
    lora_weights_style = get_lora_weights(lora_weights_style_path)
    
    average_ratio = get_ratio_between_content_and_style(
        lora_weights_content, lora_weights_style
    )
    

    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)
        attn_name = ".".join(attn_processor_name.split(".")[:-1])
        merged_lora_weights_dict_1_a, merged_lora_weights_dict_1_b = merge_lora_weights(
            lora_weights_content, attn_name
        )
        merged_lora_weights_dict_2_a, merged_lora_weights_dict_2_b = merge_lora_weights(
            lora_weights_style, attn_name
        )
        kwargs = {
            "average_ratio": average_ratio,
            "state_dict_1_a": merged_lora_weights_dict_1_a,
            "state_dict_1_b": merged_lora_weights_dict_1_b,
            "state_dict_2_a": merged_lora_weights_dict_2_a,
            "state_dict_2_b": merged_lora_weights_dict_2_b,
        }
        # Set the `lora_layer` attribute of the attention-related matrices.
        attn_module.to_q.set_lora_layer(
            initialize_klora_layer(
                **kwargs,
                part="to_q",
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
            )
        )
        attn_module.to_k.set_lora_layer(
            initialize_klora_layer(
                **kwargs,
                part="to_k",
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
            )
        )
        attn_module.to_v.set_lora_layer(
            initialize_klora_layer(
                **kwargs,
                part="to_v",
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
            )
        )
        attn_module.to_out[0].set_lora_layer(
            initialize_klora_layer(
                **kwargs,
                part="to_out.0",
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
            )
        )
    return unet

