"""Module for Registration models."""
import logging

logger = logging.getLogger(__name__)

try:
    from .geotransformer import GeoTransformer
except ImportError as err:
    logger.warning(f"Cannot import GeoTransformer: {err}")
