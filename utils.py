from klora import KLoRALinearLayer, KLoRALinearLayerInference
import torch.nn as nn
import os
import re
from typing import Optional, Dict
from huggingface_hub import hf_hub_download
import torch
from safetensors import safe_open
from diffusers.models.lora import LoRACompatibleLinear

LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"


def get_lora_weights(
    lora_name_or_path: str,
    subfolder: Optional[str] = None,
    sub_lora_weights_name: str = None,
    **kwargs,
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
            filename=(
                sub_lora_weights_name
                if sub_lora_weights_name is not None
                else LORA_WEIGHT_NAME_SAFE
            ),
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
    tensors: torch.Tensor, key: str, prefix: str = "unet."
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


def merge_sd_lora_weights(
    tensors: torch.Tensor, key: str, prefix: str = "unet.unet."
) -> Dict[str, torch.Tensor]:
    """
    Args:
        tensors (torch.Tensor): state dict of lora weights
        key (str): target attn layer's key
        prefix (str, optional): prefix for state dict.
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


def merge_community_flux_lora_weights(
    tensors: torch.Tensor, key: str, prefix: str = "transformer.", layer_num: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Args:
        tensors (torch.Tensor): state dict of lora weights
        key (str): target attn layer's key
        prefix (str, optional): prefix for state dict. Defaults to "unet.unet.".
    """
    """
    target_key = prefix + key + "."

    """
    target_key = prefix + key + "."

    out1 = {}
    out2 = {}
    if layer_num == 0:
        raise ValueError("LoRA was not loaded successfully.")
    elif layer_num == 190:
        target_key = prefix + key + "."
        if "single" in key:
            for part in ["to_k", "to_v", "to_q"]:
                down_key = target_key + f"{part}.lora_A.weight"
                up_key = target_key + f"{part}.lora_B.weight"
                out1[part] = tensors[up_key]
                out2[part] = tensors[down_key]
        else:
            for part in ["to_k", "to_v", "to_q", "to_out.0"]:
                down_key = target_key + f"{part}.lora_A.weight"
                up_key = target_key + f"{part}.lora_B.weight"
                out1[part] = tensors[up_key]
                out2[part] = tensors[down_key]
        return out1, out2


def initialize_klora_layer(
    alpha,
    beta,
    sum_timesteps,
    average_ratio,
    state_dict_1_a,
    state_dict_1_b,
    state_dict_2_a,
    state_dict_2_b,
    pattern,
    part,
    **model_kwargs,
):
    klora_layer = KLoRALinearLayer(
        alpha=alpha,
        beta=beta,
        sum_timesteps=sum_timesteps,
        average_ratio=average_ratio,
        pattern=pattern,
        weight_1_a=state_dict_1_a[part],
        weight_1_b=state_dict_1_b[part],
        weight_2_a=state_dict_2_a[part],
        weight_2_b=state_dict_2_b[part],
        **model_kwargs,
    )
    return klora_layer


def get_ratio_between_content_and_style(lora_weights_content, lora_weights_style):
    if len(lora_weights_content) != len(lora_weights_style):
        raise ValueError("The number of layers in content and style must be the same.")

    comparison_results = []

    layer_content_names = list(
        filter(lambda name: "alpha" not in name, lora_weights_content.keys())
    )
    layer_style_names = list(
        filter(lambda name: "alpha" not in name, lora_weights_style.keys())
    )

    for i in range(0, len(layer_content_names), 2):
        layer_content_name_up = layer_content_names[i + 1]
        layer_content_name_down = layer_content_names[i]
        layer_style_name_up = layer_style_names[i + 1]
        layer_style_name_down = layer_style_names[i]

        tensor_content_up = lora_weights_content[layer_content_name_up]
        tensor_content_down = lora_weights_content[layer_content_name_down]
        tensor_style_up = lora_weights_style[layer_style_name_up]
        tensor_style_down = lora_weights_style[layer_style_name_down]

        tensor_content_up = tensor_content_up.to("cuda", dtype=torch.float32)
        tensor_content_down = tensor_content_down.to("cuda", dtype=torch.float32)
        tensor_style_up = tensor_style_up.to("cuda", dtype=torch.float32)
        tensor_style_down = tensor_style_down.to("cuda", dtype=torch.float32)

        tensor_content_product = tensor_content_up @ tensor_content_down
        tensor_style_product = tensor_style_up @ tensor_style_down

        tensor_content_product_abs = torch.abs(tensor_content_product).sum()
        tensor_style_product_abs = torch.abs(tensor_style_product).sum()

        max_x_sum_content = tensor_content_product_abs.item()
        max_x_sum_style = tensor_style_product_abs.item()

        ratio = max_x_sum_content / max_x_sum_style
        comparison_results.append(ratio)
    average_ratio = 1.0
    if comparison_results:
        average_ratio = sum(comparison_results) / len(comparison_results)
    else:
        raise ValueError("No layers found in content or style.")
    comparison_results = [result for result in comparison_results if result < 3 * average_ratio]

    average_ratio = (
        sum(comparison_results) / len(comparison_results)
        if len(comparison_results) > 0
        else float("inf")
    )
    return average_ratio


def insert_sd_klora_to_unet(
    unet, lora_weights_content_path, lora_weights_style_path, alpha, beta, sum_timesteps, pattern
):
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
            "alpha": alpha,
            "beta": beta,
            "sum_timesteps": sum_timesteps,
            "average_ratio": average_ratio,
            "state_dict_1_a": merged_lora_weights_dict_1_a,
            "state_dict_1_b": merged_lora_weights_dict_1_b,
            "state_dict_2_a": merged_lora_weights_dict_2_a,
            "state_dict_2_b": merged_lora_weights_dict_2_b,
            "pattern": pattern,
        }
        # Set the `lora_layer` attribute of the attention-related matrices.
        has_bias_q = attn_module.to_q.bias is not None
        new_module_q = LoRACompatibleLinear(
            in_features=attn_module.to_q.in_features,
            out_features=attn_module.to_q.out_features,
            bias=has_bias_q,
            device=attn_module.to_q.weight.device,
            dtype=attn_module.to_q.weight.dtype,
        )
        new_module_q.set_lora_layer(
            initialize_klora_layer(
                **kwargs,
                part="to_q",
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
            )
        )
        new_module_q.weight.data.copy_(attn_module.to_q.weight.data)
        if has_bias_q:
            new_module_q.bias.data.copy_(attn_module.to_q.bias.data)
        attn_module.to_q = new_module_q
        
        has_bias_k = attn_module.to_k.bias is not None
        new_module_k = LoRACompatibleLinear(
            in_features=attn_module.to_k.in_features,
            out_features=attn_module.to_k.out_features,
            bias=has_bias_k,
            device=attn_module.to_k.weight.device,
            dtype=attn_module.to_k.weight.dtype,
        )
        new_module_k.set_lora_layer(
            initialize_klora_layer(
                **kwargs,
                part="to_k",
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
            )
        )
        new_module_k.weight.data.copy_(attn_module.to_k.weight.data)
        if has_bias_k:
            new_module_k.bias.data.copy_(attn_module.to_k.bias.data)
        attn_module.to_k = new_module_k
        
        has_bias_v = attn_module.to_v.bias is not None
        new_module_v = LoRACompatibleLinear(
            in_features=attn_module.to_v.in_features,
            out_features=attn_module.to_v.out_features,
            bias=has_bias_v,
            device=attn_module.to_v.weight.device,
            dtype=attn_module.to_v.weight.dtype,
        )
        new_module_v.set_lora_layer(
            initialize_klora_layer(
                **kwargs,
                part="to_v",
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
            )
        )
        new_module_v.weight.data.copy_(attn_module.to_v.weight.data)
        if has_bias_v:
            new_module_v.bias.data.copy_(attn_module.to_v.bias.data)
        attn_module.to_v = new_module_v
        
        has_bias_out = attn_module.to_out[0].bias is not None
        new_module_out = LoRACompatibleLinear(
            in_features=attn_module.to_out[0].in_features,
            out_features=attn_module.to_out[0].out_features,
            bias=has_bias_out,
            device=attn_module.to_out[0].weight.device,
            dtype=attn_module.to_out[0].weight.dtype,
        )
        new_module_out.set_lora_layer(
            initialize_klora_layer(
                **kwargs,
                part="to_out.0",
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
            )
        )
        new_module_out.weight.data.copy_(attn_module.to_out[0].weight.data)
        if has_bias_out:
            new_module_out.bias.data.copy_(attn_module.to_out[0].bias.data)
        attn_module.to_out[0] = new_module_out
    return unet


def insert_community_sd_lora_to_unet(
    unet, lora_weights_content_path, lora_weights_style_path, alpha, beta, sum_timesteps
):
    lora_weights_content = get_lora_weights(lora_weights_content_path)
    lora_weights_style = get_lora_weights(lora_weights_style_path)

    average_ratio = get_ratio_between_content_and_style(
        lora_weights_content, lora_weights_style
    )
    lora_weights_content = rename_safetensors_layer_name(lora_weights_content)
    lora_weights_style = rename_safetensors_layer_name(lora_weights_style)

    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)
        attn_name = ".".join(attn_processor_name.split(".")[:-1])
        merged_lora_weights_dict_1_a, merged_lora_weights_dict_1_b = (
            merge_sd_lora_weights(lora_weights_content, attn_name)
        )
        merged_lora_weights_dict_2_a, merged_lora_weights_dict_2_b = (
            merge_sd_lora_weights(lora_weights_style, attn_name)
        )
        kwargs = {
            "alpha": alpha,
            "beta": beta,
            "sum_timesteps": sum_timesteps,
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


def copy_and_assign_klora_weights(
    prefix,
    attn_module,
    sub_module_names,
    lora_weights_content,
    lora_weights_style,
    alpha,
    beta,
    sumtimesteps,
    average_ratio,
    patten,
):
    original_modules = {
        sub_module_name: (
            getattr(attn_module, sub_module_name)
            if not isinstance(attn_module, nn.Linear)
            else attn_module
        )
        for sub_module_name in sub_module_names
    }
    lora_weights_A = {
        name: lora_weights_content[prefix + name + ".lora_A.weight"]
        for name in sub_module_names
    }
    lora_weights_B = {
        name: lora_weights_content[prefix + name + ".lora_B.weight"]
        for name in sub_module_names
    }
    lora_weights_style_A = {
        name: lora_weights_style[prefix + name + ".lora_A.weight"]
        for name in sub_module_names
    }
    lora_weights_style_B = {
        name: lora_weights_style[prefix + name + ".lora_B.weight"]
        for name in sub_module_names
    }

    for sub_module_name in sub_module_names:
        original_module = original_modules[sub_module_name]
        new_module = LoRACompatibleLinear(
            in_features=original_module.in_features,
            out_features=original_module.out_features,
            bias=True,
            device=original_module.weight.device,
            dtype=original_module.weight.dtype,
        )

        klora_layer = KLoRALinearLayer(
            alpha=alpha,
            beta=beta,
            sum_timesteps=sumtimesteps,
            in_features=lora_weights_A[sub_module_name].shape[1],
            out_features=lora_weights_B[sub_module_name].shape[0],
            weight_1_a=lora_weights_B[sub_module_name],
            weight_1_b=lora_weights_A[sub_module_name],
            weight_2_a=lora_weights_style_B[sub_module_name],
            weight_2_b=lora_weights_style_A[sub_module_name],
            average_ratio=average_ratio,
            pattern=patten,
        )
        new_module.set_lora_layer(klora_layer)

        new_module.weight.data.copy_(original_module.weight.data)
        new_module.bias.data.copy_(original_module.bias.data)
        setattr(attn_module, sub_module_name, new_module)

    return attn_module


def insert_community_flux_lora_to_unet(
    unet,
    lora_weights_content_path,
    lora_weights_style_path,
    alpha,
    beta,
    diffuse_step,
    content_lora_weight_name: str = None,
    style_lora_weight_name: str = None,
    patten: str = "s*",
):
    lora_weights_content = get_lora_weights(
        lora_name_or_path=lora_weights_content_path,
        sub_lora_weights_name=content_lora_weight_name,
    )
    lora_weights_style = get_lora_weights(
        lora_name_or_path=lora_weights_style_path,
        sub_lora_weights_name=style_lora_weight_name,
    )
    average_ratio = get_ratio_between_content_and_style(
        lora_weights_content, lora_weights_style
    )
    content_layer_nums = len(lora_weights_content) // 2
    style_layer_nums = len(lora_weights_style) // 2
    sum_timesteps = diffuse_step * content_layer_nums

    if content_layer_nums != style_layer_nums:
        raise ValueError("The number of layers in content and style must be the same.")

    if content_layer_nums == 190:
        unet = unet.transformer
        for attn_processor_name, attn_processor in unet.attn_processors.items():
            attn_module = unet
            for n in attn_processor_name.split(".")[:-1]:
                attn_module = getattr(attn_module, n)
            attn_name = ".".join(attn_processor_name.split(".")[:-1])
            merged_lora_weights_dict_1_a, merged_lora_weights_dict_1_b = (
                merge_community_flux_lora_weights(
                    tensors=lora_weights_content,
                    key=attn_name,
                    layer_num=content_layer_nums,
                )
            )
            merged_lora_weights_dict_2_a, merged_lora_weights_dict_2_b = (
                merge_community_flux_lora_weights(
                    tensors=lora_weights_style,
                    key=attn_name,
                    layer_num=style_layer_nums,
                )
            )
            kwargs = {
                "alpha": alpha,
                "beta": beta,
                "sum_timesteps": sum_timesteps,
                "average_ratio": average_ratio,
                "patten": patten,
                "state_dict_1_a": merged_lora_weights_dict_1_a,
                "state_dict_1_b": merged_lora_weights_dict_1_b,
                "state_dict_2_a": merged_lora_weights_dict_2_a,
                "state_dict_2_b": merged_lora_weights_dict_2_b,
            }
            # Set the `lora_layer` attribute of the attention-related matrices.
            copy_and_assign_klora_weights(attn_module, "to_q")
            copy_and_assign_klora_weights(attn_module, "to_k")
            copy_and_assign_klora_weights(attn_module, "to_v")

            to_k = LoRACompatibleLinear(
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
                bias=True,
                device=attn_module.to_k.weight.device,
                dtype=attn_module.to_k.weight.dtype,
            )
            to_k.weight.data = attn_module.to_k.weight.data.clone()
            to_k.bias.data = attn_module.to_k.bias.data.clone()
            attn_module.to_k = to_k

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

            if not ("single" in attn_name):
                attn_module.to_out[0].set_lora_layer(
                    initialize_klora_layer(
                        **kwargs,
                        part="to_out.0",
                        in_features=attn_module.to_out[0].in_features,
                        out_features=attn_module.to_out[0].out_features,
                    )
                )

    elif content_layer_nums == 494:
        sum_timesteps = 11704
        unet = unet.transformer
        # load single_transformer_blocks_lora
        for index, layer_name in enumerate(unet.single_transformer_blocks):
            # proj
            prefix = "transformer.single_transformer_blocks." + str(index) + "."
            copy_and_assign_klora_weights(
                prefix=prefix,
                attn_module=layer_name,
                sub_module_names=["proj_mlp", "proj_out"],
                lora_weights_content=lora_weights_content,
                lora_weights_style=lora_weights_style,
                alpha=alpha,
                beta=beta,
                sumtimesteps=sum_timesteps,
                average_ratio=average_ratio,
                patten=patten,
            )
            # attn
            temp_layer = layer_name.attn
            copy_and_assign_klora_weights(
                prefix=prefix + "attn.",
                attn_module=temp_layer,
                sub_module_names=["to_q", "to_k", "to_v"],
                lora_weights_content=lora_weights_content,
                lora_weights_style=lora_weights_style,
                alpha=alpha,
                beta=beta,
                sumtimesteps=sum_timesteps,
                average_ratio=average_ratio,
                patten=patten,
            )
            # norm
            temp_layer = layer_name.norm
            copy_and_assign_klora_weights(
                prefix=prefix + "norm.",
                attn_module=temp_layer,
                sub_module_names=["linear"],
                lora_weights_content=lora_weights_content,
                lora_weights_style=lora_weights_style,
                alpha=alpha,
                beta=beta,
                sumtimesteps=sum_timesteps,
                average_ratio=average_ratio,
                patten=patten,
            )

        # load transformer_blocks_lora
        for index, layer_name in enumerate(unet.transformer_blocks):
            prefix = "transformer.transformer_blocks." + str(index) + "."
            # attn
            temp_layer = layer_name.attn
            copy_and_assign_klora_weights(
                prefix=prefix + "attn.",
                attn_module=temp_layer,
                sub_module_names=[  
                    "add_k_proj",
                    "add_q_proj",
                    "add_v_proj",
                    "to_add_out",
                    "to_k",
                    "to_q",
                    "to_v",
                ],
                lora_weights_content=lora_weights_content,
                lora_weights_style=lora_weights_style,
                alpha=alpha,
                beta=beta,
                sumtimesteps=sum_timesteps,
                average_ratio=average_ratio,
                patten=patten,
            )
            temp_layer = layer_name.attn.to_out[0]
            copy_and_assign_klora_weights(
                prefix=prefix + "attn.to_out.0",
                attn_module=temp_layer,
                sub_module_names=[""],
                lora_weights_content=lora_weights_content,
                lora_weights_style=lora_weights_style,
                alpha=alpha,
                beta=beta,
                sumtimesteps=sum_timesteps,
                average_ratio=average_ratio,
                patten=patten,
            )
            # norm1
            temp_layer = layer_name.norm1
            copy_and_assign_klora_weights(
                prefix=prefix + "norm1.",
                attn_module=temp_layer,
                sub_module_names=["linear"],
                lora_weights_content=lora_weights_content,
                lora_weights_style=lora_weights_style,
                alpha=alpha,
                beta=beta,
                sumtimesteps=sum_timesteps,
                average_ratio=average_ratio,
                patten=patten,
            )
            # norm1_context
            temp_layer = layer_name.norm1
            copy_and_assign_klora_weights(
                prefix=prefix + "norm1.",
                attn_module=temp_layer,
                sub_module_names=["linear"],
                lora_weights_content=lora_weights_content,
                lora_weights_style=lora_weights_style,
                alpha=alpha,
                beta=beta,
                sumtimesteps=sum_timesteps,
                average_ratio=average_ratio,
                patten=patten,
            )
            # ff
            for index, sub_layer_name in enumerate(layer_name.ff.net):
                if index == 0:
                    copy_and_assign_klora_weights(
                        prefix=prefix + "ff.net." + str(index) + ".",
                        attn_module=sub_layer_name,
                        sub_module_names=["proj"],
                        lora_weights_content=lora_weights_content,
                        lora_weights_style=lora_weights_style,
                        alpha=alpha,
                        beta=beta,
                        sumtimesteps=sum_timesteps,
                        average_ratio=average_ratio,
                        patten=patten,
                    )
                if index == 2:
                    copy_and_assign_klora_weights(
                        prefix=prefix + "ff.net." + str(index),
                        attn_module=sub_layer_name,
                        sub_module_names=[""],
                        lora_weights_content=lora_weights_content,
                        lora_weights_style=lora_weights_style,
                        alpha=alpha,  
                        beta=beta,
                        sumtimesteps=sum_timesteps,
                        average_ratio=average_ratio,
                        patten=patten,
                    )
            # ff_context
            for index, sub_layer_name in enumerate(layer_name.ff_context.net):
                if index == 0:
                    copy_and_assign_klora_weights(
                        prefix=prefix + "ff_context.net." + str(index) + ".",
                        attn_module=sub_layer_name,
                        sub_module_names=["proj"],
                        lora_weights_content=lora_weights_content,
                        lora_weights_style=lora_weights_style,
                        alpha=alpha,
                        beta=beta,
                        sumtimesteps=sum_timesteps,
                        average_ratio=average_ratio,
                        patten=patten,
                    )
                if index == 2:
                    copy_and_assign_klora_weights(
                        prefix=prefix + "ff_context.net." + str(index),
                        attn_module=sub_layer_name,
                        sub_module_names=[""],
                        lora_weights_content=lora_weights_content,
                        lora_weights_style=lora_weights_style,
                        alpha=alpha,
                        beta=beta,
                        sumtimesteps=sum_timesteps,
                        average_ratio=average_ratio, 
                        patten=patten,
                    )

    else:
        raise ValueError("This kind of LoRA is not supported now, it is recommended to use dreambooth-trained LoRA")
    return unet


def rename_safetensors_layer_name(tensor):
    def rename_key(key):
        patten = r"(\w+\_blocks\.\d+\.attentions\.\d+\.+\w+\.\d+\.attn\d+)"
        match = re.search(patten, key)
        new_key = "unet.unet."
        if match:
            new_key += match.group(1)
            new_key += "."

        patten = r"(\w+\_block\.attentions\.\d+\.+\w+\.\d+\.attn\d+)"
        match = re.search(patten, key)
        if match:
            new_key += match.group(1)
            new_key += "."

        key = key.replace(".", "_")
        if "to_q" in key:
            new_key += "to_q"
        elif "to_k" in key:
            new_key += "to_k"
        elif "to_v" in key:
            new_key += "to_v"
        elif "to_out" in key:
            new_key += "to_out.0"
        if "down_weight" in key:
            new_key += ".lora.down.weight"
        if "up_weight" in key:
            new_key += ".lora.up.weight"
        return new_key

    renamed_tensor = {rename_key(key): value for key, value in tensor.items()}

    return renamed_tensor
