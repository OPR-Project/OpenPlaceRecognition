import torch
from torch import nn, Tensor


class TemporalAveragePooling(nn.Module):
    """Simple module that averages features across sequence dimension."""

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Average features across sequence dimension.

        Args:
            x: Tensor of shape [B, S, D] where S is sequence length

        Returns:
            Tuple containing:
                - Tensor of shape [B, D] with averaged features
                - Original input tensor x for further processing if needed
        """
        return torch.mean(x, dim=1), x


class TemporalMaxPooling(nn.Module):
    """Simple module that takes max value for each feature across sequence."""

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Take maximum value for each feature across sequence dimension.

        Args:
            x: Tensor of shape [B, S, D] where S is sequence length

        Returns:
            Tuple containing:
                - Tensor of shape [B, D] with max pooled features
                - Original input tensor x for further processing if needed
        """
        return torch.max(x, dim=1)[0], x


class TemporalAttentionFusion(nn.Module):
    """Attention-based fusion of sequence features."""

    def __init__(self, feature_dim: int):
        """Initialize with feature dimension.

        Args:
            feature_dim: Dimension of input features
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply attention-based fusion across sequence dimension.

        Args:
            x: Tensor of shape [B, S, D]

        Returns:
            Tuple containing:
                - Tensor of shape [B, D] with fused features
                - Tensor of shape [B, S, D] with attended features
        """
        # Calculate attention scores [B, S, 1]
        attn_scores = self.attention(x)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Apply attention weights
        return torch.sum(x * attn_weights, dim=1), x * attn_weights


class TemporalSelfAttentionFusion(nn.Module):
    """Multi-head self-attention based fusion of sequence features."""

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        pooling: nn.Module | None = None,
    ):
        """Initialize with feature dimension and attention parameters.

        Args:
            feature_dim: Dimension of input features
            num_heads: Number of attention heads
            dropout: Dropout probability
            pooling: Module for pooling sequence features to a single vector.
                     If None, defaults to TemporalAttentionFusion.
        """
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        # Multi-head self-attention components
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)

        # Pooling module to combine sequence features
        self.pooling = pooling or TemporalAttentionFusion(feature_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply multi-head self-attention and pooling across sequence dimension.

        Args:
            x: Tensor of shape [B, S, D]

        Returns:
            Tuple containing:
                - Tensor of shape [B, D] with pooled features
                - Tensor of shape [B, S, D] with attention weights applied
        """
        B, S, D = x.shape

        # Project queries, keys, values
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D/H]
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D/H]
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D/H]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # [B, H, S, S]
        attn_weights = torch.softmax(scores, dim=-1)  # [B, H, S, S]
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        attended = torch.matmul(attn_weights, v)  # [B, H, S, D/H]
        attended = attended.transpose(1, 2).contiguous().view(B, S, D)  # [B, S, D]
        attended = self.out_proj(attended)  # [B, S, D]

        # Apply pooling to get final descriptor
        return self.pooling(attended), attended


class TemporalLSTMFusion(nn.Module):
    """LSTM-based fusion of sequence features."""

    def __init__(self, feature_dim: int, hidden_dim: int = None):
        """Initialize with feature dimensions.

        Args:
            feature_dim: Dimension of input features
            hidden_dim: Dimension of LSTM hidden state (defaults to feature_dim)
        """
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Process sequence through LSTM and return final descriptor.

        Args:
            x: Tensor of shape [B, S, D]

        Returns:
            Tuple containing:
                - Tensor of shape [B, D] with final descriptor
                - Tensor of shape [B, S, D] with LSTM outputs
        """
        # Run LSTM on sequence
        outputs, (final_hidden, _) = self.lstm(x)
        # Use final hidden state
        return self.output_projection(final_hidden.squeeze(0)), outputs
