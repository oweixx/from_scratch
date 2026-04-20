# References: https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
# I'm not sure referenced from "pytorch-image-models".

from typing import Callable, Optional
from torch import Tensor, nn

"""
It's just a simple Multi-Layer Perceptron (MLP) composed of two layers.
"""
class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None,
                 act_layer: Callable[..., nn.Module] = nn.GELU, drop: float = 0.0, bias: bool = True,) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # Simple Linear Layer (in_features -> hidden_features -> out_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)
    
    # simple forwarding
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x