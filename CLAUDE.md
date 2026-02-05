# LongEcho - Agent Guidelines

## Running Commands

Always use `uv run` prefix for all Python commands:
```bash
uv run python -m longecho.main      # Run server
uv run pytest                        # Run tests
uv run uvicorn longecho.main:app    # Run with uvicorn
```

Do NOT use bare `python` or `pytest` - they won't have the venv dependencies.

## Vendored Echo-TTS

Echo-TTS inference code is vendored at `src/longecho/_vendor/echo_tts/`.

**Imports:**
```python
# Correct
from longecho._vendor.echo_tts import load_model_from_hf, ae_decode

# Wrong - there is no external echo-tts package
from inference import load_model_from_hf
from echo_tts import load_model_from_hf
```

**Internal imports** within `_vendor/echo_tts/` use relative imports (`.inference`, `.model`, etc.).

## Dependencies

- PyTorch is installed from the cu128 index (CUDA 12.8) - configured in `pyproject.toml`
- `torchcodec` on Windows requires FFmpeg shared libraries in PATH (system dependency)
- Run `uv sync` to install/update dependencies

## Licensing

- Code: MIT (most files) and Apache-2.0 (autoencoder.py)
- Model weights: CC-BY-NC-SA-4.0 (non-commercial only)
- Generated audio inherits CC-BY-NC-SA-4.0 license

## Project Structure

```
src/longecho/
├── main.py              # FastAPI server, model loading
├── audio_generator.py   # Chunk generation with context
├── voice_manager.py     # Voice preprocessing & caching
├── text_normalizer.py   # Currency, abbreviations, etc.
├── text_segmenter.py    # Text chunking logic
└── _vendor/echo_tts/    # Vendored inference code (don't modify unless necessary)
```

## Testing

- Tests are in `tests/` directory
- Use `uv run pytest -v` for verbose output
- Tests mock the heavy ML components - they don't require GPU

## Voice Library

- Voice `.wav` files go in `voice_library/`
- Preprocessed voices are cached as `.pkl` files
- Cache invalidates automatically if `.wav` file changes
