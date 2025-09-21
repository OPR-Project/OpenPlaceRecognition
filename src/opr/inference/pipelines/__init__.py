"""Inference pipelines.

Exports top-k Place Recognition pipeline that uses the new Index module.
"""

from .place_recognition import PlaceRecognitionPipeline

__all__ = [
    "PlaceRecognitionPipeline",
]
