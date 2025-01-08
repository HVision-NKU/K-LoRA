from typing import Optional, Union
import torch
from torch import nn

glo_count = 0


class KLoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        average_ratio: float,
        weight_1_a: torch.Tensor,
        weight_1_b: torch.Tensor,
        weight_2_a: torch.Tensor,
        weight_2_b: torch.Tensor,
        rank: int = 8,
        device: Optional[Union[torch.device, str]] = "cuda",
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.device = device
        self.weight_1_a = weight_1_a.to(device)
        self.weight_1_b = weight_1_b.to(device)
        self.weight_2_a = weight_2_a.to(device)
        self.weight_2_b = weight_2_b.to(device)
        self.average_ratio = average_ratio
        self.rank = rank

        self.out_features = out_features
        self.in_features = in_features
        self.forward_type = "merge"

    # select topk weights
    def get_klora_weight(self, timestep):
        sum_timesteps = 28000
        k = 64
        alpha = 1.5
        beta = 0.5
        gamma = self.average_ratio
        
        # compute the sum of top k values
        time_ratio = timestep % sum_timesteps
        matrix1 = self.weight_1_a @ self.weight_1_b
        abs_matrix = torch.abs(matrix1)
        top_k_values, _ = torch.topk(abs_matrix.flatten(), k)
        top_k_sum1 = top_k_values.sum()
        matrix2 = self.weight_2_a @ self.weight_2_b
        abs_matrix = torch.abs(matrix2)
        top_k_values, _ = torch.topk(abs_matrix.flatten(), k)
        top_k_sum2 = top_k_values.sum()
        
        
        scale = alpha * time_ratio / sum_timesteps + beta
        
        # apply scaling factor to the sum of top k values
        top_k_sum1 = top_k_sum1 / gamma
        top_k_sum2 = top_k_sum2 * scale
        
        temp_ratio = top_k_sum1 / top_k_sum2
        if temp_ratio > 1:
            return matrix1
        else:
            return matrix2

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        global glo_count
        orig_dtype = hidden_states.dtype
        dtype = self.weight_1_a.dtype

        if self.forward_type == "merge":
            glo_count += 1
            weight = self.get_klora_weight(glo_count)
        elif self.forward_type == "weight_1":
            weight = self.weight_1_a @ self.weight_1_b
        elif self.forward_type == "weight_2":
            weight = self.weight_2_a @ self.weight_2_b
        else:
            raise ValueError(self.forward_type)
        hidden_states = nn.functional.linear(hidden_states.to(dtype), weight=weight)
        return hidden_states.to(orig_dtype)


class KLoRALinearLayerInference(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.zeros((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.weight.dtype
        hidden_states = nn.functional.linear(
            hidden_states.to(dtype), weight=self.weight
        )
        return hidden_states.to(orig_dtype)
