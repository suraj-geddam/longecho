# LongEcho: Long-form audio generation with Echo-TTS
"""
LongEcho package for generating long-form audio using Echo-TTS.

This package provides:
- Text segmentation for chunking long text
- Voice management with caching
- Audio generation with continuation for seamless long-form output
- FastAPI server for web interface
"""

# Eager imports: modules with no external dependencies
from .text_segmenter import segment_text
from .text_normalizer import TextNormalizer

__all__ = [
    "segment_text",
    "TextNormalizer",
    "VoiceManager",
    "AudioGenerator",
]

# Lazy imports for modules that depend on external 'inference' package (PEP 562)
_lazy_imports = {
    "VoiceManager": ".voice_manager",
    "AudioGenerator": ".audio_generator",
}


def __getattr__(name: str):
    """Lazily import VoiceManager and AudioGenerator to defer 'inference' dependency."""
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(_lazy_imports[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
