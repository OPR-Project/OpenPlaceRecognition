"""Feature extraction modules."""
from .convnext import ConvNeXtTinyFeatureExtractor
from .mink_resnet import MinkResNetFPNFeatureExtractor
from .resnet import (
    ResNet18FeatureExtractor,
    ResNet18FPNFeatureExtractor,
    ResNet50FeatureExtractor,
    ResNet50FPNFeatureExtractor,
)
