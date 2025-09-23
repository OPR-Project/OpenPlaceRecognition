"""OpenPlaceRecognition runtime inference package.

Provides:
- index: FAISS Flat backend built at load from descriptors.npy (no FAISS files persisted)
- pipelines: top‑k Place Recognition (raw distances), Registration, Localization
  (per-candidate estimated_pose and registration_confidence)
- preprocessing: composable modality transforms used by pipelines

On-disk layout used by index: descriptors.npy, meta.parquet (idx + pose), schema.json.
"""
