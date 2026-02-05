import logging
import threading
import time
from typing import List, Generator, Tuple, Any

import torch

from longecho._vendor.echo_tts import (
    get_text_input_ids_and_mask,
    ae_decode,
    crop_audio_to_flattening_point,
    get_speaker_latent_and_mask,
    sample_blockwise_euler_cfg_independent_guidances,
)

logger = logging.getLogger(__name__)

# Generation parameters from design
DEFAULT_PARAMS = {
    "num_steps": 40,
    "cfg_scale_text": 3.0,
    "cfg_scale_speaker": 8.0,
    "cfg_min_t": 0.5,
    "cfg_max_t": 1.0,
    "truncation_factor": 1.0,
    "rescale_k": 1.0,
    "rescale_sigma": 3.0,
    "speaker_kv_scale": None,
    "speaker_kv_max_layers": None,
    "speaker_kv_min_t": None,
}

# Max latents for continuation - if previous chunk exceeds this, we truncate
# With smaller chunks (~12-14s), full chunk should be ~280-320 latents
MAX_CONTINUATION_LATENTS = 400


class AudioGenerator:
    """
    Generates long-form audio by chunking text and maintaining context.

    Uses Echo's blockwise inference to generate audio chunks with
    continuation from previous chunks for coherence.
    """

    def __init__(self, model: Any, fish_ae: Any, pca_state: Any):
        """
        Initialize audio generator.

        Args:
            model: Echo DiT model
            fish_ae: Fish autoencoder
            pca_state: PCA state
        """
        self.model = model
        self.fish_ae = fish_ae
        self.pca_state = pca_state
        self.device = model.device
        self._generation_id = 0  # Unique ID for each generation
        self._stop_generation_id = -1  # Which generation ID to stop (-1 = none)
        self._is_generating = False
        self._generation_lock = threading.Lock()  # Prevent concurrent generations
        self._id_lock = threading.Lock()  # Protect generation_id and stop_generation_id

    @property
    def is_generating(self) -> bool:
        """Check if generation is currently in progress."""
        return self._is_generating

    def request_stop(self, generation_id: int | None = None) -> None:
        """Request a specific generation to stop after the current chunk.

        Args:
            generation_id: The ID of the generation to stop. If None, stops all generations.
        """
        with self._id_lock:
            if generation_id is None:
                # Stop all generations (backward compat / emergency stop)
                self._stop_generation_id = self._generation_id
                logger.info(f"Stop requested for all generations (<= {self._generation_id})")
            else:
                # Stop only the specific generation
                # Only update if this would stop more generations than currently marked
                if generation_id > self._stop_generation_id:
                    self._stop_generation_id = generation_id
                logger.info(f"Stop requested for generation {generation_id}")

    def generate_long_audio(
        self,
        text_chunks: List[str],
        speaker_latent: torch.Tensor,
        speaker_mask: torch.Tensor,
        rng_seed: int = 0,
    ) -> tuple[int, Generator[torch.Tensor, None, None]]:
        """
        Generate audio for multiple text chunks with continuation.

        Returns a tuple of (generation_id, generator). The generation_id can be used
        with request_stop(generation_id) to stop this specific generation.

        Args:
            text_chunks: List of text strings to generate
            speaker_latent: Speaker latent from voice
            speaker_mask: Speaker mask from voice
            rng_seed: Random seed for generation

        Returns:
            Tuple of (generation_id, generator) where generator yields audio tensors
            for each chunk (shape: [1, 1, audio_length])
        """
        # Task 6: Validate that we have text to generate
        if not text_chunks:
            raise ValueError("No text to generate after normalization")

        # Assign unique ID to this generation for stop isolation (thread-safe)
        with self._id_lock:
            self._generation_id += 1
            my_generation_id = self._generation_id

            # Stop any previous generation that might be running
            if self._stop_generation_id < my_generation_id - 1:
                self._stop_generation_id = my_generation_id - 1
                logger.info(f"Auto-stopping generations <= {self._stop_generation_id} (new generation {my_generation_id} starting)")

        logger.info(f"Generation {my_generation_id} waiting for lock...")

        return my_generation_id, self._generate_chunks(
            text_chunks, speaker_latent, speaker_mask, rng_seed, my_generation_id
        )

    def _generate_chunks(
        self,
        text_chunks: List[str],
        speaker_latent: torch.Tensor,
        speaker_mask: torch.Tensor,
        rng_seed: int,
        my_generation_id: int,
    ) -> Generator[torch.Tensor, None, None]:
        """Internal generator that yields audio chunks."""

        # Acquire lock to prevent concurrent generations
        # This blocks until any previous generation finishes its current chunk
        self._generation_lock.acquire()
        try:
            logger.info(f"Generation {my_generation_id} acquired lock, starting")
            self._is_generating = True

            # Check if we were stopped while waiting for the lock
            if self._stop_generation_id >= my_generation_id:
                logger.info(f"Generation {my_generation_id} was stopped while waiting for lock")
                return

            continuation_latent = None
            previous_chunk_text = ""  # Continuation text from previous chunk

            for i, chunk_text in enumerate(text_chunks):
                # Check for stop request targeting THIS generation
                if self._stop_generation_id >= my_generation_id:
                    logger.info(f"Generation {my_generation_id} stopped by user")
                    return

                # Log chunk number and full text (encode to ASCII to avoid unicode errors in Windows console)
                chunk_text_safe = chunk_text.encode('ascii', 'replace').decode('ascii')
                logger.info(f"Generating chunk {i+1}/{len(text_chunks)}: {chunk_text_safe}")

                # When using continuation, include portion of previous chunk's text
                # (~last 60% of previous chunk, aligned to sentence boundary)
                if previous_chunk_text:
                    # Encode to ASCII to avoid unicode errors
                    prev_preview = previous_chunk_text[:80].encode('ascii', 'replace').decode('ascii')
                    logger.debug(f"  Previous chunk text ({len(previous_chunk_text)} chars): {prev_preview}..." if len(previous_chunk_text) > 80 else f"  Previous chunk text: {prev_preview}")
                    full_text = previous_chunk_text + " " + chunk_text
                    full_preview = full_text[:150].encode('ascii', 'replace').decode('ascii')
                    logger.debug(f"  Full text for generation ({len(full_text)} chars): {full_preview}..." if len(full_text) > 150 else f"  Full text: {full_preview}")
                else:
                    full_text = chunk_text

                audio_chunk, latent_out, audio_full = self._generate_chunk(
                    full_text,
                    speaker_latent,
                    speaker_mask,
                    continuation_latent,
                    rng_seed + i,  # Different seed per chunk
                )

                # audio_chunk = NEW audio only (continuation removed, for yielding)
                # audio_full = FULL cropped audio (includes continuation)

                # Extract continuation from NEW audio only (not including previous continuation)
                # This prevents continuation from growing each iteration
                if i < len(text_chunks) - 1:  # Not the last chunk
                    continuation_latent = self._extract_continuation(latent_out, audio_chunk)

                    # Use full previous chunk text - continuation latents now cover most of the chunk
                    previous_chunk_text = chunk_text

                yield audio_chunk
        finally:
            self._is_generating = False
            self._generation_lock.release()
            logger.info(f"Generation {my_generation_id} released lock")

    def _generate_chunk(
        self,
        text: str,
        speaker_latent: torch.Tensor,
        speaker_mask: torch.Tensor,
        continuation_latent: torch.Tensor | None,
        rng_seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a single audio chunk.

        Args:
            text: Text to generate (includes previous text if continuation)
            speaker_latent: Speaker latent tensor
            speaker_mask: Speaker mask tensor
            continuation_latent: Optional continuation from previous chunk
            rng_seed: Random seed

        Returns:
            Tuple of (new_audio, latent_output, full_audio):
                - new_audio: Audio with continuation removed (for yielding)
                - latent_output: Generated latent tensor
                - full_audio: Full cropped audio (for extracting next continuation)
        """
        start_time = time.time()

        # Encode text
        logger.debug(f"  [1/4] Encoding text ({len(text)} chars)...")
        # Encode to ASCII to avoid unicode errors in Windows console
        text_preview = text[:200].encode('ascii', 'replace').decode('ascii')
        logger.debug(f"  Text content: {text_preview}..." if len(text) > 200 else f"  Text content: {text_preview}")
        t0 = time.time()
        text_input_ids, text_mask = get_text_input_ids_and_mask(
            [text],
            max_length=None,
            device=self.device,
        )
        logger.debug(f"  [1/4] Text encoded in {time.time() - t0:.2f}s")

        # Determine block sizes
        if continuation_latent is None:
            block_sizes = [640]  # Full generation
            logger.debug(f"  [2/4] Starting fresh generation (block size: {block_sizes[0]})")
        else:
            # Account for continuation latent length
            cont_len = continuation_latent.shape[1]
            remaining = 640 - cont_len
            block_sizes = [remaining] if remaining > 0 else [640]
            logger.debug(f"  [2/4] Continuing from {cont_len} latents (block size: {block_sizes[0]})")

        # Generate latents
        logger.debug(f"  [2/4] Generating latents ({DEFAULT_PARAMS['num_steps']} diffusion steps)...")
        t0 = time.time()
        latent_out = sample_blockwise_euler_cfg_independent_guidances(
            model=self.model,
            speaker_latent=speaker_latent,
            speaker_mask=speaker_mask,
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            rng_seed=rng_seed,
            block_sizes=block_sizes,
            continuation_latent=continuation_latent,
            **DEFAULT_PARAMS,
        )
        logger.debug(f"  [2/4] Latents generated in {time.time() - t0:.2f}s")
        logger.debug(f"  Latent shape: {latent_out.shape}")

        # Decode to audio
        logger.debug(f"  [3/4] Decoding latents to audio...")
        t0 = time.time()
        audio_out = ae_decode(self.fish_ae, self.pca_state, latent_out)
        logger.debug(f"  [3/4] Audio decoded in {time.time() - t0:.2f}s")
        logger.debug(f"  Audio shape before crop: {audio_out.shape}")

        logger.debug(f"  [4/4] Cropping audio...")
        logger.debug(f"  Audio shape before crop: {audio_out.shape} = {audio_out.shape[-1] / 44100:.2f}s")
        logger.debug(f"  Latent shape: {latent_out.shape}")

        # Crop FULL audio using FULL latents to find proper endpoint
        audio_out_full = crop_audio_to_flattening_point(audio_out, latent_out[0])
        logger.debug(f"  Audio shape after crop: {audio_out_full.shape} = {audio_out_full.shape[-1] / 44100:.2f}s")

        # Remove continuation portion if present (for yielding)
        # Use the fundamental relationship: 1 latent = 2048 audio samples
        if continuation_latent is not None:
            continuation_len = continuation_latent.shape[1]

            # Direct calculation: each latent produces 2048 audio samples
            continuation_samples = continuation_len * 2048

            logger.debug(f"  Removing continuation: {continuation_len} latents = {continuation_samples} samples ({continuation_samples / 44100:.2f}s)")

            # Return audio WITHOUT continuation (for yielding to user)
            audio_out_new = audio_out_full[..., continuation_samples:]
            logger.debug(f"  New audio (to yield): {audio_out_new.shape[-1]} samples ({audio_out_new.shape[-1] / 44100:.2f}s)")

            total_time = time.time() - start_time
            logger.info(f"  Chunk complete in {total_time:.2f}s")

            # Return: new audio, latents, full audio for continuation extraction
            return audio_out_new, latent_out, audio_out_full
        else:
            logger.debug(f"  Final audio shape: {audio_out_full.shape}")

            total_time = time.time() - start_time
            logger.info(f"  Chunk complete in {total_time:.2f}s")

            return audio_out_full, latent_out, audio_out_full

    def _extract_continuation(self, latent_out: torch.Tensor, audio_out: torch.Tensor) -> torch.Tensor:
        """
        Extract continuation context from generated audio.

        Uses the FULL chunk audio as continuation (not just the last N seconds).
        This ensures perfect text-audio alignment when we pass full chunk text.

        Args:
            latent_out: Generated latent tensor (used to calculate expected latent count)
            audio_out: Generated audio tensor (cropped, the full chunk to use as continuation)

        Returns:
            Continuation latent tensor (re-encoded from full chunk audio)
        """
        # Use the FULL chunk audio as continuation
        continuation_audio = audio_out

        # Re-encode through Fish AE (same as Echo-TTS example)
        continuation_latent, continuation_mask = get_speaker_latent_and_mask(
            self.fish_ae,
            self.pca_state,
            continuation_audio[0],
        )

        # Trim to actual content (use mask)
        actual_latents = int(continuation_mask.sum().item())
        continuation_latent = continuation_latent[:, :actual_latents]

        # Cap at max to leave room for new generation (need at least 240 new latents)
        if continuation_latent.shape[1] > MAX_CONTINUATION_LATENTS:
            logger.warning(f"Continuation too long ({continuation_latent.shape[1]} latents), truncating to {MAX_CONTINUATION_LATENTS}")
            continuation_latent = continuation_latent[:, -MAX_CONTINUATION_LATENTS:]

        return continuation_latent
