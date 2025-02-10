import torch
from torch import nn
from typing import Literal
from torch.nn import functional as F


RADIO_MODELS = ["radio_v2.5-l",]
DINO_V2_MODELS = ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]
BOQ_MODELS = ["get_trained_boq",]
DINO_FACETS = Literal["query", "key", "value", "token", None]


class ViTBaseFeatureExtractor(nn.Module):
    """
        Extract features from an intermediate layer in Dino-v2
    """
    def __init__(
            self, 
            vit_type, 
            layer: int, 
            facet: DINO_FACETS="token", 
            use_cls=False, 
            norm_descs=True, 
            device: str = "cpu",
        ) -> None:
        """
            Parameters:
            - vit_type:   The DINO-v2 or Radio model to use
            - layer:        The layer to extract features from
            - facet:    "query", "key", or "value" for the attention
                        facets. "token" for the output of the layer.
            - use_cls:  If True, the CLS token (first item) is also
                        included in the returned list of descriptors.
                        Otherwise, only patch descriptors are used.
            - norm_descs:   If True, the descriptors are normalized
            - device:   PyTorch device to use
        """
        super().__init__()
        self.vit_type: str = vit_type
        self.model = self._load_model(vit_type)
        self.device = torch.device(device)
        self.model = self.model.eval().to(self.device)
        self.layer: int = layer
        self.facet = facet
        if self.facet == "token":
            self.fh_handle = self.model.blocks[self.layer].\
                    register_forward_hook(
                            self._generate_forward_hook())
        elif self.facet is None:
            self.fh_handle = None
        else:
            self.fh_handle = self.model.blocks[self.layer].\
                    attn.qkv.register_forward_hook(
                            self._generate_forward_hook())
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        # Hook data
        self._hook_out = None
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
            Parameters:
            - img:   The input image
        """
        with torch.no_grad():
            res = self.model(img)
            if self.fh_handle is not None:
                if self.use_cls:
                    res = self._hook_out
                else:
                    res = self._hook_out[:, 1:, ...]
                if self.facet in ["query", "key", "value"]:
                    d_len = res.shape[2] // 3
                    if self.facet == "query":
                        res = res[:, :, :d_len]
                    elif self.facet == "key":
                        res = res[:, :, d_len:2*d_len]
                    else:
                        res = res[:, :, 2*d_len:]
                if self.norm_descs:
                    res = F.normalize(res, dim=-1)
        self._hook_out = None   # Reset the hook
        return res
    
    def _load_model(self, vit_type) -> nn.Module:
        if vit_type in DINO_V2_MODELS:
            return torch.hub.load('facebookresearch/dinov2', vit_type)
        elif vit_type in RADIO_MODELS:
            return torch.hub.load('NVlabs/RADIO', 'radio_model', version=vit_type)
        elif vit_type in BOQ_MODELS:
            return torch.hub.load("amaralibey/bag-of-queries", vit_type, backbone_name="dinov2", output_dim=12288)
        else:
            raise ValueError(f"Invalid model version: {vit_type}")
    
    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            self._hook_out = output
        return _forward_hook
    
    def __del__(self):
        self.fh_handle.remove()
