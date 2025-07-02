"""Base meta-models for Place Recognition."""
import os
from typing import Dict, Optional

import torch
from loguru import logger
from torch import Tensor, nn
import numpy as np

from opr.modules import Concat
from opr.modules.temporal import TemporalAveragePooling
from opr.optional_deps import lazy

ME = lazy("MinkowskiEngine", feature="sparse convolutions")
polygraphy = lazy("polygraphy", feature="TensorRT")
torch_tensorrt = lazy("torch_tensorrt", feature="TensorRT")
onnxruntime = lazy("onnxruntime", feature="ONNX")


class ImageModel(nn.Module):
    """Meta-model for image-based Place Recognition. Combines feature extraction backbone and head modules."""

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        fusion: Optional[nn.Module] = None,
        forward_type: Optional[str] = "fp32",
        onnx_model_path: Optional[str] = None,
        engine_path: Optional[str] = None
    ) -> None:
        """Meta-model for image-based Place Recognition.

        Args:
            backbone (ImageFeatureExtractor): Image feature extraction backbone.
            head (ImageHead): Image head module.
            fusion (FusionModule, optional): Module to fuse descriptors for multiple images in batch.
                Defaults to None.
            forward_type (str, optional): One of fp32 | onnx_fp32 | trt_fp32 | trt_int8.
                Defaults to fp32.
            onnx_model_path (str, optional): Path to ResNet18FPN_ImageFeatureExtractor.onnx.
                Defaults to None.
            engine_path (str, optional): Path to ResNet18FPN_ImageFeatureExtractor_int8.engine.
                Defaults to None.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.fusion = fusion
        self.forward_type = forward_type

        if forward_type.startswith("onnx"):
            print(f"WARNING - {forward_type} mode is only for inference on cuda!")
            so = onnxruntime.SessionOptions()
            exproviders = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            if self.backbone.__class__.__name__ == "ResNet18FPNFeatureExtractor":
                self.ort_session = onnxruntime.InferenceSession(
                    onnx_model_path, so, providers=exproviders)
            else:
                raise NotImplementedError
        elif forward_type.startswith("trt_fp32"):
            print(f"WARNING - {forward_type} mode is only for inference on cuda!")
            self.trt_model = None
        elif forward_type.startswith("trt_int8"):
            print(f"WARNING - {forward_type} mode is only for inference on cuda!")
            with open(engine_path, "rb") as bf:
                self.engine = polygraphy.backend.trt.engine_from_bytes(bf.read())
            self.runner = polygraphy.backend.trt.TrtRunner(self.engine)
            self.runner.__enter__()

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:  # noqa: D102
        img_descriptors = {}
        for key, value in batch.items():
            if key.startswith("images_"):
                if self.forward_type == "fp32":
                    features = self.backbone(value)
                elif self.forward_type == "onnx_fp32":
                    input_name = self.ort_session.get_inputs()[0].name
                    output_name = self.ort_session.get_outputs()[0].name
                    io_binding = self.ort_session.io_binding()
                    value = value.contiguous()

                    io_binding.bind_input(
                        name=input_name,
                        device_type='cuda',
                        device_id=0,
                        element_type=np.float32,
                        shape=tuple(value.shape),
                        buffer_ptr=value.data_ptr(),
                    )

                    features = torch.empty((value.shape[0], 256, 12, 20), dtype=torch.float32, device='cuda:0').contiguous()
                    io_binding.bind_output(
                        name=output_name,
                        device_type='cuda',
                        device_id=0,
                        element_type=np.float32,
                        shape=tuple(features.shape),
                        buffer_ptr=features.data_ptr(),
                    )
                    self.ort_session.run_with_iobinding(io_binding)
                elif self.forward_type == "trt_fp32":
                    if not self.trt_model:
                        # Enabled precision for TensorRT optimization
                        enabled_precisions = {torch.float32}
                        # Whether to print verbose logs
                        debug = False
                        # Workspace size for TensorRT
                        workspace_size = 20 << 30
                        # Maximum number of TRT Engines
                        # (Lower value allows more graph segmentation)
                        min_block_size = 7
                        # Operations to Run in Torch, regardless of converter support
                        torch_executed_ops = {}

                        # Build and compile the model with torch.compile, using Torch-TensorRT backend
                        self.trt_model = torch_tensorrt.compile(
                            self.backbone,
                            ir="torch_compile",
                            inputs=[value.contiguous()],
                            enabled_precisions=enabled_precisions,
                            debug=debug,
                            workspace_size=workspace_size,
                            min_block_size=min_block_size,
                            torch_executed_ops=torch_executed_ops,
                        )
                    features = self.trt_model(value.contiguous())
                elif self.forward_type == "trt_int8":
                    features = self.runner.infer({"input": value.contiguous()}, copy_outputs_to_host=False)["output"]
                else:
                    raise NotImplementedError("Unknown forward_type for ImageModel")
                img_descriptors[key] = self.head(features)
        if len(img_descriptors) > 1:
            if self.fusion is None:
                raise ValueError("Fusion module is not defined but multiple images are provided")
            descriptor = self.fusion(img_descriptors)
        else:
            if self.fusion is not None:
                raise ValueError("Fusion module is defined but only one image is provided")
            descriptor = list(img_descriptors.values())[0]
        out_dict: Dict[str, Tensor] = {"final_descriptor": descriptor}
        return out_dict


class SemanticModel(ImageModel):
    """Meta-model for semantic-based Place Recognition. Combines feature extraction backbone and head modules."""

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        fusion: Optional[nn.Module] = None,
        forward_type: Optional[str] = "fp32",
        onnx_model_path: Optional[str] = None,
        engine_path: Optional[str] = None
    ) -> None:
        """Meta-model for semantic-based Place Recognition.

        Args:
            backbone (ImageFeatureExtractor): Semantic feature extraction backbone.
            head (ImageHead): Image head module.
            fusion (FusionModule, optional): Module to fuse descriptors for multiple images in batch.
                Defaults to None.
            forward_type (str, optional): One of fp32 | onnx_fp32 | trt_fp32 | trt_int8.
                Defaults to fp32.
            onnx_model_path (str, optional): Path to ResNet18FPN_ImageFeatureExtractor.onnx.
                Defaults to None.
            engine_path (str, optional): Path to ResNet18FPN_ImageFeatureExtractor_int8.engine.
                Defaults to None.
        """
        super().__init__(
            backbone=backbone,
            head=head,
            fusion=fusion,
            forward_type=forward_type,
            onnx_model_path=onnx_model_path,
            engine_path=engine_path
        )

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:  # noqa: D102
        mask_descriptors = {}
        for key, value in batch.items():
            if key.startswith("masks_"):
                if self.forward_type == "fp32":
                    features = self.backbone(value)
                elif self.forward_type == "onnx_fp32":
                    input_name = self.ort_session.get_inputs()[0].name
                    output_name = self.ort_session.get_outputs()[0].name
                    io_binding = self.ort_session.io_binding()
                    value = value.contiguous()

                    io_binding.bind_input(
                        name=input_name,
                        device_type='cuda',
                        device_id=0,
                        element_type=np.float32,
                        shape=tuple(value.shape),
                        buffer_ptr=value.data_ptr(),
                    )

                    features = torch.empty((value.shape[0], 256, 12, 20), dtype=torch.float32, device='cuda:0').contiguous()
                    io_binding.bind_output(
                        name=output_name,
                        device_type='cuda',
                        device_id=0,
                        element_type=np.float32,
                        shape=tuple(features.shape),
                        buffer_ptr=features.data_ptr(),
                    )
                    self.ort_session.run_with_iobinding(io_binding)
                elif self.forward_type == "trt_fp32":
                    if not self.trt_model:
                        # Enabled precision for TensorRT optimization
                        enabled_precisions = {torch.float32}
                        # Whether to print verbose logs
                        debug = False
                        # Workspace size for TensorRT
                        workspace_size = 20 << 30
                        # Maximum number of TRT Engines
                        # (Lower value allows more graph segmentation)
                        min_block_size = 7
                        # Operations to Run in Torch, regardless of converter support
                        torch_executed_ops = {}

                        # Build and compile the model with torch.compile, using Torch-TensorRT backend
                        self.trt_model = torch_tensorrt.compile(
                            self.backbone,
                            ir="torch_compile",
                            inputs=[value.contiguous()],
                            enabled_precisions=enabled_precisions,
                            debug=debug,
                            workspace_size=workspace_size,
                            min_block_size=min_block_size,
                            torch_executed_ops=torch_executed_ops,
                        )
                    features = self.trt_model(value.contiguous())
                elif self.forward_type == "trt_int8":
                    features = self.runner.infer({"input": value.contiguous()}, copy_outputs_to_host=False)["output"]
                else:
                    raise NotImplementedError("Unknown forward_type for ImageModel")
                mask_descriptors[key] = self.head(features)
        if len(mask_descriptors) > 1:
            if self.fusion is None:
                raise ValueError("Fusion module is not defined but multiple masks are provided")
            descriptor = self.fusion(mask_descriptors)
        else:
            if self.fusion is not None:
                raise ValueError("Fusion module is defined but only one mask is provided")
            descriptor = list(mask_descriptors.values())[0]
        out_dict: Dict[str, Tensor] = {"final_descriptor": descriptor}
        return out_dict


class CloudModel(nn.Module):
    """Meta-model for lidar-based Place Recognition. Combines feature extraction backbone and head modules."""

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
    ) -> None:
        """Meta-model for lidar-based Place Recognition.

        Args:
            backbone (CloudFeatureExtractor): Cloud feature extraction backbone.
            head (CloudHead): Cloud head module.

        Raises:
            RuntimeError: MinkowskiEngine is not installed. CloudModel requires MinkowskiEngine.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:  # noqa: D102
        x = ME.SparseTensor(
            features=batch["pointclouds_lidar_feats"], coordinates=batch["pointclouds_lidar_coords"]
        )
        x = self.backbone(x)
        x = self.head(x)
        out_dict: Dict[str, Tensor] = {"final_descriptor": x}
        return out_dict


class LateFusionModel(nn.Module):
    """Meta-model for multimodal Place Recognition architectures with late fusion."""

    def __init__(
        self,
        image_module: Optional[ImageModel] = None,
        semantic_module: Optional[SemanticModel] = None,
        cloud_module: Optional[CloudModel] = None,
        soc_module: Optional[nn.Module] = None,
        fusion_module: Optional[nn.Module] = None,
    ) -> None:
        """Meta-model for multimodal Place Recognition architectures with late fusion.

        Args:
            image_module (ImageModule, optional): Image modality branch. Defaults to None.
            semantic_module (SemanticModel, optional): Semantic modality branch. Defaults to None.
            cloud_module (CloudModule, optional): Cloud modality branch. Defaults to None.
            soc_module (nn.Module, optional): Module to fuse different modalities.
            fusion_module (FusionModule, optional): Module to fuse different modalities.
                If None, will be set to opr.modules.Concat(). Defaults to None.
        """
        super().__init__()

        self.image_module = image_module
        self.semantic_module = semantic_module
        self.cloud_module = cloud_module
        self.soc_module = soc_module
        if fusion_module:
            self.fusion_module = fusion_module
        else:
            self.fusion_module = Concat()

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:  # noqa: D102
        out_dict: Dict[str, Tensor] = {}

        if self.image_module is not None:
            out_dict["image"] = self.image_module(batch)["final_descriptor"]

        if self.semantic_module is not None:
            out_dict["semantic"] = self.semantic_module(batch)["final_descriptor"]

        if self.cloud_module is not None:
            out_dict["cloud"] = self.cloud_module(batch)["final_descriptor"]

        if self.soc_module is not None:
            out_dict["soc"] = self.soc_module(batch["soc"])["final_descriptor"]

        out_dict["final_descriptor"] = self.fusion_module(out_dict)

        return out_dict

class SequenceLateFusionModel(nn.Module):
    """Meta-model for sequence-based multimodal Place Recognition with late fusion."""

    def __init__(
        self,
        late_fusion_model: LateFusionModel,
        temporal_fusion_module: nn.Module | None = None,
    ) -> None:
        """Meta-model for sequence-based multimodal Place Recognition with late fusion.

        Args:
            late_fusion_model (LateFusionModel): Base model for processing individual frames.
            temporal_fusion_module (nn.Module, optional): Module to fuse features across time.
                If None, defaults to a module that takes the average across the sequence.
        """
        super().__init__()
        self.late_fusion_model = late_fusion_model
        self.temporal_fusion_module = temporal_fusion_module or TemporalAveragePooling()

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Process a sequence of frames efficiently by reshaping to batch processing.

        Args:
            batch: Dictionary containing sequence data with shape [B, S, ...]
                  where B is batch size and S is sequence length

        Returns:
            Dictionary with the final descriptor after temporal fusion
        """
        batch_size, seq_len = self._get_batch_and_seq_dims(batch)
        flat_batch = self._reshape_batch_for_processing(batch, batch_size, seq_len)
        flat_output = self.late_fusion_model(flat_batch)
        descriptors = flat_output["final_descriptor"].view(batch_size, seq_len, -1)
        final_descriptor = self.temporal_fusion_module(descriptors)
        return {"final_descriptor": final_descriptor}

    def _get_batch_and_seq_dims(self, batch: Dict[str, Tensor]) -> tuple[int, int]:
        """Extract batch size and sequence length from batch data."""
        for key, value in batch.items():
            if key.startswith("images_"):
                return value.shape[0], value.shape[1]  # B, S from [B, S, C, H, W]
            elif key == "pointclouds_lidar_coords":
                return value.shape[0], value.shape[1]  # B, S from [B, S, N, 3]

        raise ValueError("Could not determine batch size and sequence length from batch")

    def _reshape_batch_for_processing(self, batch: Dict[str, Tensor], batch_size: int, seq_len: int) -> Dict[str, Tensor]:
        """Reshape batch from [B, S, ...] to [B*S, ...] for efficient processing."""
        flat_batch = {}

        for key, value in batch.items():
            if key.startswith("images_"):
                # Reshape image data: [B, S, C, H, W] -> [B*S, C, H, W]
                flat_batch[key] = value.reshape(batch_size * seq_len, *value.shape[2:])
            elif key == "pointclouds_lidar_coords":
                # Reshape point coordinates: [B, S, N, 3] -> [B*S, N, 3]
                flat_batch[key] = value.reshape(batch_size * seq_len, *value.shape[2:])
            elif key == "pointclouds_lidar_feats":
                # Reshape point features: [B, S, N, 1] -> [B*S, N, 1]
                flat_batch[key] = value.reshape(batch_size * seq_len, *value.shape[2:])

        return flat_batch
