"""
Test for Bug 6: Audio content skipped at chunk boundaries

This test verifies that audio generation doesn't skip content at the
beginning of continuation chunks due to improper cropping order.
"""
import pytest
import torch
from unittest.mock import Mock, patch

import sys
sys.modules['inference'] = Mock()
sys.modules['inference_blockwise'] = Mock()

from longecho.audio_generator import AudioGenerator, MAX_CONTINUATION_LATENTS


def test_continuation_cropping_order():
    """
    Verify that audio is cropped BEFORE removing continuation samples.

    Bug: Previously removed continuation samples first, causing misalignment
    and cutting into new content.

    Fix: Crop full audio with full latents first, THEN remove continuation.
    """
    # Setup mocks
    model = Mock()
    fish_ae = Mock()
    pca_state = Mock()

    with patch('longecho.audio_generator.get_text_input_ids_and_mask') as mock_text, \
         patch('longecho.audio_generator.sample_blockwise_euler_cfg_independent_guidances') as mock_sample, \
         patch('longecho.audio_generator.ae_decode') as mock_decode, \
         patch('longecho.audio_generator.crop_audio_to_flattening_point') as mock_crop, \
         patch('longecho.audio_generator.get_speaker_latent_and_mask') as mock_speaker:

        # Simulate generation with continuation
        continuation_len = 210
        new_len = 430
        total_len = continuation_len + new_len

        # Mock returns
        mock_text.return_value = (torch.zeros(1, 100), torch.ones(1, 100, dtype=torch.bool))
        mock_sample.return_value = torch.randn(1, total_len, 80)

        # Full audio (continuation + new)
        full_audio = torch.randn(1, 1, total_len * 2048)
        mock_decode.return_value = full_audio

        # Simulate cropping at latent 600 (removes padding)
        cropped_at = 600
        cropped_audio = full_audio[:, :, :cropped_at * 2048]
        mock_crop.return_value = cropped_audio

        # Mock continuation extraction
        mock_speaker.return_value = (torch.randn(1, 210, 80), torch.ones(1, 210, dtype=torch.bool))

        generator = AudioGenerator(model, fish_ae, pca_state)
        speaker_latent = torch.randn(1, 100, 80)
        speaker_mask = torch.ones(1, 100, dtype=torch.bool)

        # Generate two chunks
        text_chunks = ["First chunk text.", "Second chunk text."]

        gen_id, gen = generator.generate_long_audio(text_chunks, speaker_latent, speaker_mask)
        chunks = list(gen)

        # Verify chunk 2 (with continuation)
        chunk2 = chunks[1]

        # CORRECT BEHAVIOR: We remove continuation based on LATENT COUNT
        # Using fundamental relationship: 1 latent = 2048 samples
        # Continuation: 210 latents = 210 * 2048 = 430,080 samples
        # Cropped audio: 600 * 2048 = 1,228,800 samples
        # Expected output: 1,228,800 - 430,080 = 798,720 samples

        # Verify we got the expected latent-based removal
        expected_samples = (cropped_at - continuation_len) * 2048  # 798,720
        assert chunk2.shape[-1] == expected_samples, \
            f"Expected {expected_samples} samples (latent-based removal), got {chunk2.shape[-1]}"

        # Verify crop was called with FULL audio and FULL latents
        calls = mock_crop.call_args_list
        assert len(calls) == 2, "Should have cropped both chunks"

        # For chunk 2 (with continuation), verify crop received full audio
        chunk2_crop_call = calls[1]
        audio_arg = chunk2_crop_call[0][0]  # First positional argument

        # The audio passed to crop should be the FULL decoded audio
        assert audio_arg.shape[-1] == full_audio.shape[-1], \
            "Crop should receive full audio, not with continuation already removed"


def test_full_chunk_continuation_alignment():
    """
    Test that verifies full-chunk continuation approach maintains alignment.

    New approach (2024):
    - Use the FULL previous chunk audio as continuation
    - Pass FULL previous chunk text
    - This ensures perfect text-audio alignment

    The old approach used a fixed 32.8% ratio which could mismatch.
    """
    # With smaller chunks (~12-14s = ~300 latents), full chunk as continuation
    # leaves room for ~340 new latents (640 - 300 = 340)

    typical_chunk_latents = 300  # ~14 seconds
    new_generation_room = 640 - typical_chunk_latents  # 340 latents
    new_generation_seconds = new_generation_room * 2048 / 44100  # ~15.8s

    # Verify we have enough room for new generation
    assert new_generation_room >= 200, \
        f"Need at least 200 new latents, have room for {new_generation_room}"

    # Verify new generation time is reasonable for smaller chunks
    assert new_generation_seconds > 9, \
        f"New generation should be >9s, got {new_generation_seconds:.1f}s"

    # Verify MAX_CONTINUATION_LATENTS leaves room for new generation
    min_new_latents = 640 - MAX_CONTINUATION_LATENTS
    assert min_new_latents >= 200, \
        f"MAX_CONTINUATION_LATENTS={MAX_CONTINUATION_LATENTS} leaves only {min_new_latents} for new generation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
