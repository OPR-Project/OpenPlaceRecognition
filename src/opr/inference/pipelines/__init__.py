"""Inference pipelines.

Exports top-k Place Recognition pipeline that uses the new Index module.
"""

from .place_recognition import PlaceRecognitionPipeline
from .registration import RansacPointCloudRegistrationPipeline

__all__ = [
    "PlaceRecognitionPipeline",
    "RansacPointCloudRegistrationPipeline",
]
