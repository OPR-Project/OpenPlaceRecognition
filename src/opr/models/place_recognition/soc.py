"""Semantic-Object-Context modality model."""
from typing import Dict, Optional

import torch.nn.functional as F
from torch import Tensor, nn


class SOCMLP(nn.Module):
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
