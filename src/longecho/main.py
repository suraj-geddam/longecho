import asyncio
import base64
import json
import logging
import signal
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
import torch
import torchaudio

from longecho._vendor.echo_tts import (
    load_model_from_hf,
    load_fish_ae_from_hf,
    load_pca_state_from_hf,
)
from .voice_manager import VoiceManager
from .audio_generator import AudioGenerator
from .text_segmenter import segment_text
from .text_normalizer import TextNormalizer, NormalizationLevel
from .voice_event_broadcaster import VoiceEventBroadcaster
from .file_watcher import FileWatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and voices on startup, cleanup on shutdown"""
    # Startup
    logger.info("Loading Echo models...")
    model = load_model_from_hf()
    fish_ae = load_fish_ae_from_hf()
    pca_state = load_pca_state_from_hf()
    logger.info("Models loaded successfully")

    # Initialize voice manager and load voices
    logger.info("Loading voices...")
    app.state.voice_manager = VoiceManager(fish_ae, pca_state)
    app.state.voice_manager.load_voices()
    logger.info(f"Loaded {len(app.state.voice_manager.get_voice_names())} voices")

    # Initialize audio generator
    app.state.audio_generator = AudioGenerator(model, fish_ae, pca_state)
    logger.info("Audio generator initialized")

    # Initialize voice event broadcaster
    app.state.voice_broadcaster = VoiceEventBroadcaster()
    logger.info("Voice event broadcaster initialized")

    # Create callback for new voice files
    async def on_new_voice_file(wav_path: Path):
        if not wav_path.exists():
            logger.warning(f"Voice file '{wav_path}' was deleted before processing")
            return

        voice_name = wav_path.stem
        logger.info(f"New voice file detected: {voice_name}")

        # Broadcast processing event
        app.state.voice_broadcaster.broadcast({
            "type": "processing",
            "voice": voice_name,
        })

        try:
            # Process the voice (runs in thread pool to avoid blocking)
            result = await asyncio.to_thread(
                app.state.voice_manager.add_voice, wav_path
            )

            if result is not None:
                # Broadcast ready event
                app.state.voice_broadcaster.broadcast({
                    "type": "ready",
                    "voice": voice_name,
                })
                logger.info(f"Voice '{voice_name}' ready")
            else:
                logger.info(f"Voice '{voice_name}' was already loaded")
        except Exception as e:
            logger.error(f"Failed to process voice '{voice_name}': {e}")
            app.state.voice_broadcaster.broadcast({
                "type": "error",
                "voice": voice_name,
                "reason": str(e),
            })

    # Create callback for deleted voice files
    async def on_deleted_voice_file(wav_path: Path):
        voice_name = wav_path.stem
        logger.info(f"Voice file deleted: {voice_name}")

        # Remove from loaded voices
        removed = app.state.voice_manager.remove_voice(voice_name)

        if removed:
            # Broadcast removed event
            app.state.voice_broadcaster.broadcast({
                "type": "removed",
                "voice": voice_name,
            })

    # Start file watcher
    app.state.file_watcher = FileWatcher(
        watch_dir=Path("voice_library"),
        on_new_wav=on_new_voice_file,
        on_deleted_wav=on_deleted_voice_file,
    )
    app.state.file_watcher.start(asyncio.get_running_loop())
    logger.info("File watcher started")

    # Set up Ctrl+C handler to stop generation gracefully
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum, frame):
        if app.state.audio_generator.is_generating:
            logger.info("Ctrl+C received - requesting generation stop")
            app.state.audio_generator.request_stop()
        else:
            logger.info("Ctrl+C received - shutting down")
            # Call the original handler to exit normally
            if callable(original_sigint_handler):
                original_sigint_handler(signum, frame)
            else:
                raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, handle_sigint)
        logger.info("Ctrl+C handler installed")
    except ValueError:
        # Signal handlers can only be set in the main thread
        logger.debug("Skipping signal handler (not in main thread)")

    yield

    # Shutdown (cleanup if needed)
    logger.info("Shutting down...")

    # Stop file watcher
    if hasattr(app.state, 'file_watcher'):
        app.state.file_watcher.stop()


# Initialize FastAPI app with lifespan
app = FastAPI(title="LongEcho", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Dependency injection functions
def get_voice_manager(request: Request) -> VoiceManager:
    """Get the VoiceManager instance from app state"""
    return request.app.state.voice_manager


def get_audio_generator(request: Request) -> AudioGenerator:
    """Get the AudioGenerator instance from app state"""
    return request.app.state.audio_generator


def get_voice_broadcaster(request: Request) -> VoiceEventBroadcaster:
    """Get the VoiceEventBroadcaster instance from app state"""
    return request.app.state.voice_broadcaster


@app.get("/")
async def root():
    """Serve the main page"""
    return FileResponse("static/index.html")


@app.get("/favicon.ico")
async def favicon():
    """Return 204 No Content for favicon requests"""
    return Response(status_code=204)


@app.get("/health")
async def health(voice_manager: VoiceManager = Depends(get_voice_manager)):
    """Health check endpoint"""
    return {
        "status": "ok",
        "voices_loaded": len(voice_manager.get_voice_names())
    }


@app.get("/voices")
async def get_voices(voice_manager: VoiceManager = Depends(get_voice_manager)):
    """Get list of available voices"""
    return {
        "voices": voice_manager.get_voice_names()
    }


@app.get("/voice-events")
async def voice_events(
    voice_broadcaster: VoiceEventBroadcaster = Depends(get_voice_broadcaster),
):
    """
    SSE endpoint for voice library events.

    Streams events:
    - processing: New voice file detected, processing started
    - ready: Voice processed and available
    - error: Voice processing failed
    - removed: Voice file deleted and unloaded
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        queue = voice_broadcaster.subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Send heartbeat to detect dead connections
                    yield ": heartbeat\n\n"
        finally:
            voice_broadcaster.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/stop")
async def stop_generation(
    generation_id: int | None = None,
    audio_generator: AudioGenerator = Depends(get_audio_generator)
):
    """Stop a specific audio generation or all generations if no ID provided."""
    audio_generator.request_stop(generation_id)
    return {"status": "stop requested", "generation_id": generation_id}


@app.get("/generate")
async def generate(
    text: str = Query(..., min_length=1, max_length=100000, description="Text to generate audio for"),
    voice: str = Query(..., min_length=1, description="Voice name to use"),
    normalization_level: NormalizationLevel = Query("moderate", description="Text normalization level: 'moderate' (preserve plain numbers) or 'full' (convert all numbers to words)"),
    voice_manager: VoiceManager = Depends(get_voice_manager),
    audio_generator: AudioGenerator = Depends(get_audio_generator),
):
    """
    Generate audio for long-form text with SSE streaming.

    Streams events:
    - progress: Generation progress updates
    - chunk: Base64-encoded audio chunks
    - complete: Generation finished
    - error: Error occurred
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Validate voice exists
            if voice not in voice_manager.get_voice_names():
                yield f"data: {json.dumps({'type': 'error', 'message': f'Voice {voice} not found'})}\n\n"
                return

            # Get voice data
            speaker_latent, speaker_mask = voice_manager.get_voice(voice)

            # Normalize text for TTS (expand currencies, abbreviations, etc.)
            normalizer = TextNormalizer(level=normalization_level)
            normalized_text = normalizer.normalize(text)
            logger.info(f"Normalized text ({len(text)} -> {len(normalized_text)} chars)")

            # Segment text
            text_chunks = segment_text(normalized_text)
            logger.info(f"Created {len(text_chunks)} chunks")

            # Generate chunks - use thread pool to allow event loop to process other requests
            generation_id, generator = audio_generator.generate_long_audio(
                text_chunks, speaker_latent, speaker_mask
            )

            # Send generation_id first so frontend can use it for stop requests
            yield f"data: {json.dumps({'type': 'start', 'generation_id': generation_id, 'chunks': len(text_chunks)})}\n\n"

            i = 0
            try:
                while True:
                    # Run the blocking generator iteration in a thread pool
                    audio_chunk = await asyncio.to_thread(next, generator, None)
                    if audio_chunk is None:
                        break

                    # Send progress
                    progress_msg = f"Generated chunk {i+1}/{len(text_chunks)}"
                    yield f"data: {json.dumps({'type': 'progress', 'message': progress_msg})}\n\n"

                    # Convert audio to base64 WAV
                    audio_base64 = _audio_to_base64_wav(audio_chunk)

                    # Send chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'data': audio_base64, 'index': i})}\n\n"

                    i += 1

                # Send completion
                yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            finally:
                # Close the generator to release the lock if it's suspended at yield.
                # If the generator is still executing in the thread pool, close() raises
                # ValueError - in that case the generator will finish its current iteration
                # and release the lock when it checks the stop flag.
                try:
                    generator.close()
                except ValueError:
                    pass  # Generator still executing, will stop on next iteration

        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


def _audio_to_base64_wav(audio_tensor: torch.Tensor) -> str:
    """
    Convert audio tensor to base64-encoded WAV.

    Args:
        audio_tensor: Audio tensor (shape: [batch, channels, samples])

    Returns:
        Base64-encoded WAV string
    """
    # Convert to CPU and squeeze batch dimension
    audio_cpu = audio_tensor[0].cpu()

    # Save to temporary WAV file (TorchCodec backend doesn't support BytesIO)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        # Save audio to temp file
        torchaudio.save(tmp_path, audio_cpu, 44100)

        # Read back as bytes
        with open(tmp_path, 'rb') as f:
            audio_bytes = f.read()

        # Encode to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        return audio_base64
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("longecho.main:app", host="127.0.0.1", port=8100, reload=True)
