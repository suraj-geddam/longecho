# LongEcho: Long-Form Audio Generation with Echo-TTS

Generate coherent audio for arbitrary-length text using Echo-TTS, which has a ~30-second generation limit per inference call.

## Features

- Generate audio for arbitrary-length text
- Maintains voice coherence across chunks using blockwise inference
- Real-time streaming — audio plays as it generates
- Simple web interface for generation and playback
- Voice library with automatic preprocessing and caching
- Text normalization for better TTS output (currencies, abbreviations, etc.)

## Quick Start

### 1. Install Dependencies

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable package management.

**Install uv (if not already installed):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Install project dependencies:**
```bash
uv sync
```

This automatically installs PyTorch with CUDA 12.8 support from the PyTorch index.

#### Windows: FFmpeg

On Windows, `torchcodec` requires FFmpeg shared libraries in PATH. Install via `winget install ffmpeg` or download the **full-shared** build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).

### 2. Add Voice Samples

Place `.wav` files in the `voice_library/` directory:

```bash
cp path/to/your/voice.wav voice_library/
```

The first time you run the app, it will preprocess these files and cache them as `.pkl` files for fast loading.

### 3. Run the Server

```bash
uv run python -m longecho.main
```

Or with uvicorn directly (use `--host 0.0.0.0` to expose on your local network):
```bash
uv run uvicorn longecho.main:app --port 8100 --reload
```

### 4. Open Web Interface

Visit http://localhost:8100 in your browser.

1. Enter your text (any length)
2. Select a voice from the dropdown
3. Click "Generate Audio"
4. Audio chunks stream as they're generated

## How It Works

### Text Normalization

Before generation, text is normalized for better TTS output:

- **Currencies**: `$5M` → "5 million dollars", `$99.99` → "99 dollars and 99 cents"
- **Abbreviations**: `Dr.` → "Doctor", `etc.` → "etcetera", `vs.` → "versus"
- **Parentheses**: Content is preserved but parens removed — Echo-TTS uses WhisperD format where text in parentheses denotes sound effects rather than speech

Two normalization levels available via API:
- `moderate` (default): Normalize currencies and abbreviations, preserve plain numbers
- `full`: Also convert numbers to words

### Text Segmentation

Text is split into ~160-220 character chunks at natural boundaries:

1. Sentence boundaries (`.`, `!`, `?`)
2. Clause separators (`,`, `;`)
3. Word boundaries (spaces)
4. Hard cut if no boundary found

### Contextual Generation

1. First chunk: Generate fresh with selected voice
2. Subsequent chunks: Use last ~10 seconds (210 latents) as context
3. Each chunk includes previous text for natural continuation
4. Audio streams to browser as each chunk completes

### Voice Management

- Voices are preprocessed using Fish autoencoder + PCA
- Results cached as `.pkl` files for fast loading
- Cache automatically invalidates if `.wav` file changes
- New `.wav` files are detected automatically via file watcher

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.8+
- 8GB+ VRAM recommended
- [uv](https://docs.astral.sh/uv/) package manager
- **Windows only:** FFmpeg shared libraries in PATH (see installation instructions)

## API Endpoints

- `GET /` - Web UI
- `GET /voices` - List available voices
- `POST /generate` - Generate audio (SSE stream). JSON body: `{"text": "...", "voice": "...", "normalization_level": "moderate"}`
- `GET /health` - Health check

## Development

### Project Structure

```
longecho/
├── src/
│   └── longecho/
│       ├── __init__.py
│       ├── main.py              # FastAPI server
│       ├── text_normalizer.py   # TTS text preprocessing
│       ├── text_segmenter.py    # Text chunking
│       ├── voice_manager.py     # Voice preprocessing & caching
│       ├── audio_generator.py   # Audio generation with continuation
│       ├── file_watcher.py      # Voice file auto-detection
│       ├── voice_event_broadcaster.py  # SSE voice events
│       └── _vendor/
│           └── echo_tts/        # Vendored Echo-TTS inference code
├── voice_library/               # Voice .wav files (user-provided)
├── static/
│   └── index.html               # Web UI
├── tests/                       # Test suite
└── pyproject.toml
```

### Running Tests

```bash
uv run pytest
```

Or with verbose output:
```bash
uv run pytest -v
```

## License

MIT — see [LICENSE](LICENSE).

## Credits

Built with [Echo-TTS](https://github.com/jordandare/echo-tts) by Jordan Darefsky.

### Third-Party Licenses

This project includes vendored code from Echo-TTS:

- **Echo-TTS** by Jordan Darefsky - MIT License
- **autoencoder.py** - Apache-2.0 License (derived from Fish Speech)
- **Model weights** - CC-BY-NC-SA-4.0 (non-commercial use only)

See `src/longecho/_vendor/echo_tts/LICENSE` for full license text.
