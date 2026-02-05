"""
Vendored Echo-TTS inference code.

Echo-TTS is a text-to-speech model by Jordan Darefsky.
Original repository: https://github.com/jordandare/echo-tts

License: MIT (see LICENSE file in this directory)
         autoencoder.py is Apache-2.0 (see file header)

Note: Model weights are licensed CC-BY-NC-SA-4.0 (non-commercial).
"""

from .inference import (
    load_model_from_hf,
    load_fish_ae_from_hf,
    load_pca_state_from_hf,
    load_audio,
    get_text_input_ids_and_mask,
    get_speaker_latent_and_mask,
    ae_encode,
    ae_decode,
    crop_audio_to_flattening_point,
    PCAState,
)
from .inference_blockwise import sample_blockwise_euler_cfg_independent_guidances
from .model import EchoDiT
from .autoencoder import DAC, build_ae

__all__ = [
    # Loaders
    "load_model_from_hf",
    "load_fish_ae_from_hf",
    "load_pca_state_from_hf",
    "load_audio",
    # Inference
    "get_text_input_ids_and_mask",
    "get_speaker_latent_and_mask",
    "ae_encode",
    "ae_decode",
    "crop_audio_to_flattening_point",
    "sample_blockwise_euler_cfg_independent_guidances",
    # Types
    "PCAState",
    "EchoDiT",
    "DAC",
    "build_ae",
]
