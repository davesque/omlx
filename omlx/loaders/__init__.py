"""
Side-channel model loaders for formats that need a non-default load path.

Each module here exposes a ``try_load_*`` helper that returns the model
on success or ``None`` when the loader is not applicable (so the calling
engine can fall through to its default path). This keeps the engine
classes simple — they don't need to know how to dispatch between
mlx-lm, mlx-vlm, JANG, and any future quantization format.
"""
