import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from models.aggregator import Aggregator
from heads.camera_head import CameraHead
from heads.dpt_head import DPTHead
from heads.track_head import TrackHead

"""
VGGT Class
"""
class VGGT(nn.Module, PyTorchModelHubMixin):
    def init(self, img_size=518, patch_size=14, embed_dim=1024,
             enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()
        
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        
        self.camera_head = CameraHead(dim_in=2*embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2*embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2*embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2*embed_dim, patch_size=patch_size) if enable_track else None
    
    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward Pass of the VGGT Model
            images: Input images with shape [S, 3, H, W] or [B, S, 3, H, W] in range [0, 1]
            B: batch size, S: sequence length, 3: channels, H: height, W: weight
        """
        
        if len(images.shape) == 4 :         # if images no batch dim
            images = images.unsqueeze(0)
            
        # skip the query_points
        
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        
        predictions = {}
        
        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list
            
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
                
            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf
                
        # skip track_head
        
        # store original images for visualization during inference
        if not self.training:
            predictions["images"] = images
        
        return predictions