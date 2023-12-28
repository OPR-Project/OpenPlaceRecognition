"""Semantic-Object-Context modality model."""
from typing import Dict, Optional

import torch.nn.functional as F
from mlp_mixer_pytorch import MLPMixer
from torch import Tensor, nn


class SOCModel(nn.Module):
    """Semantic-Object-Context modality base model class."""

    def __init__(self, num_classes: int, num_objects: int, embeddings_size: Optional[int] = 256) -> None:
        """Semantic-Object-Context modality model.

        Args:
            num_classes (int): number of classes
            num_objects (int): number of objects
            embeddings_size (int): size of output embeddings

        Returns:
            None
        """
        super().__init__()

        # Input shape (batch_size, num_classes, num_objects, 3 (coords))
        self.num_classes = num_classes
        self.num_objects = num_objects

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            batch (Dict[str, Tensor]): input batch

        Returns:
            Dict[str, Tensor]: output dictionary
        """
        raise NotImplementedError


class SOCMLP(SOCModel):
    """Semantic-Object-Context modality model."""

    def __init__(self, num_classes: int, num_objects: int, embeddings_size: Optional[int] = 256) -> None:
        """Semantic-Object-Context modality model.

        Args:
            num_classes (int): number of classes
            num_objects (int): number of objects
            embeddings_size (int): size of embeddings

        Returns:
            None
        """
        super().__init__()

        # Input shape (batch_size, num_classes, num_objects, 3 (coords))
        self.num_classes = num_classes
        self.num_objects = num_objects

        self.fc1 = nn.Linear(num_classes * num_objects * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, embeddings_size)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            batch (Dict[str, Tensor]): input batch

        Returns:
            torch.Tensor: output tensor of shape (batch_size, embeddings_size)
        """
        x = batch["soc"]
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        descriptor = self.fc3(x)
        out_dict: Dict[str, Tensor] = {"final_descriptor": descriptor}
        return out_dict


class SOCMLPMixer(SOCModel):
    """Semantic-Object-Context modality model based on MLP Mixer .

    Kind of Attention-layer build on top of MLPs.
    Original paper: https://arxiv.org/abs/2105.01601
    implementation: https://github.com/lucidrains/mlp-mixer-pytorch
    """

    def __init__(
        self,
        num_classes: int,
        num_objects: int,
        patch_size: int = 1,
        hidden_dim: int = 64,
        depth: int = 3,
        embeddings_size: int = 256,
    ) -> None:
        """Semantic-Object-Context modality model based on MLP Mixer .

        Kind of Attention-layer build on top of MLPs.
        Original paper: https://arxiv.org/abs/2105.01601
        implementation: https://github.com/lucidrains/mlp-mixer-pytorch

        Args:
            num_classes (int): number of classes
            num_objects (int): number of objects
            patch_size (int): patch size
            hidden_dim (int): hidden dimension
            depth (int): depth
            embeddings_size (int): size of embeddings

        Returns:
            None
        """
        super(SOCMLPMixer, self).__init__(num_classes, num_objects)

        self.mlp_mixer = MLPMixer(
            image_size=(num_classes, 1),
            channels=num_objects * 3,  # Assuming each of the K triplets is a "channel"
            patch_size=patch_size,  # Should be divider of N
            dim=hidden_dim,
            depth=depth,
            num_classes=embeddings_size,  # This will be projected down to 256 by the custom network
        )
        # Define a fully connected layer that takes the output of the MLP Mixer and
        # projects it down to the desired embedding size (256 in this case)

        self.fc = nn.Linear(embeddings_size, embeddings_size)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            batch (Dict[str, Tensor]): input batch

        Returns:
            Dict[str, Tensor] : output dictionary with "final_descriptor" key containing the output tensor
        """
        # Reshape input to be compatible with the MLP Mixer, which expects an "image" tensor
        # Assuming the input x is of shape (batch_size, N, K, 3)
        x = batch["soc"]
        batch_size = x.shape[0]

        # Flatten the last two dimensions and treat them as channels (K*3)
        x_reshaped = x.view(batch_size, self.num_classes, self.num_objects * 3)
        x_permuted = x_reshaped.permute(0, 2, 1)

        x_permuted = x_permuted.unsqueeze(3)  # Add a height dimension
        # Pass the reshaped input through the MLP Mixer
        x_mixed = self.mlp_mixer(x_permuted)

        # Flatten the output to pass through the fully connected layer
        x_flat = x_mixed.view(batch_size, -1)
        descriptor = self.fc(x_flat)

        out_dict: Dict[str, Tensor] = {"final_descriptor": descriptor}
        return out_dict
