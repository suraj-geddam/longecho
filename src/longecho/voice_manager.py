import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Any

import torch

from longecho._vendor.echo_tts import load_audio, get_speaker_latent_and_mask

logger = logging.getLogger(__name__)


class VoiceManager:
    """
    Manages voice sample preprocessing and caching.

    Scans voice_library/ for .wav files, preprocesses them using Echo's
    Fish autoencoder + PCA, and caches the speaker latents to .pkl files
    for fast loading.
    """

    def __init__(self, fish_ae: Any, pca_state: Any, voice_dir: Path = None):
        """
        Initialize voice manager.

        Args:
            fish_ae: Fish autoencoder model
            pca_state: PCA state for latent compression
            voice_dir: Directory containing voice .wav files
        """
        self.fish_ae = fish_ae
        self.pca_state = pca_state
        self.voice_dir = voice_dir or Path("voice_library")
        self.voices: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def load_voices(self):
        """
        Load all voices from voice_library directory.

        For each .wav file:
        - Check if .pkl cache exists and is newer
        - Load from cache if valid, otherwise preprocess and cache
        """
        if not self.voice_dir.exists():
            logger.warning(f"Voice directory {self.voice_dir} does not exist")
            return

        wav_files = list(self.voice_dir.glob("*.wav"))
        logger.info(f"Found {len(wav_files)} voice files in {self.voice_dir}")

        for wav_path in wav_files:
            try:
                voice_name = wav_path.stem
                pkl_path = wav_path.with_suffix('.pkl')

                # Check if cache is valid
                if pkl_path.exists() and pkl_path.stat().st_mtime > wav_path.stat().st_mtime:
                    logger.info(f"Loading cached voice: {voice_name}")
                    with open(pkl_path, 'rb') as f:
                        speaker_latent, speaker_mask = pickle.load(f)
                else:
                    logger.info(f"Preprocessing voice: {voice_name}")
                    speaker_latent, speaker_mask = self._preprocess_voice(wav_path)

                    # Cache the preprocessed data
                    with open(pkl_path, 'wb') as f:
                        pickle.dump((speaker_latent, speaker_mask), f)

                self.voices[voice_name] = (speaker_latent, speaker_mask)
                logger.info(f"Loaded voice '{voice_name}': latent shape {speaker_latent.shape}")

            except Exception as e:
                logger.error(f"Failed to load voice {wav_path}: {e}")
                continue

    def _preprocess_voice(self, wav_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess a voice file into speaker latents.

        Args:
            wav_path: Path to .wav file

        Returns:
            Tuple of (speaker_latent, speaker_mask)
        """
        # Load audio
        audio = load_audio(str(wav_path)).to(self.fish_ae.device)

        # Get speaker latent and mask
        speaker_latent, speaker_mask = get_speaker_latent_and_mask(
            self.fish_ae,
            self.pca_state,
            audio
        )

        return speaker_latent, speaker_mask

    def get_voice(self, voice_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get speaker latent and mask for a voice.

        Args:
            voice_name: Name of the voice (filename without extension)

        Returns:
            Tuple of (speaker_latent, speaker_mask)

        Raises:
            KeyError: If voice not found
        """
        if voice_name not in self.voices:
            raise KeyError(f"Voice '{voice_name}' not found. Available: {list(self.voices.keys())}")

        return self.voices[voice_name]

    def get_voice_names(self) -> List[str]:
        """Get list of available voice names."""
        return list(self.voices.keys())

    def add_voice(self, wav_path: Path) -> str | None:
        """
        Process a single voice file and add it to loaded voices.

        Args:
            wav_path: Path to .wav file

        Returns:
            Voice name if processed, None if already loaded

        Raises:
            Exception if processing fails
        """
        voice_name = wav_path.stem

        # Skip if already loaded
        if voice_name in self.voices:
            logger.info(f"Voice '{voice_name}' already loaded, skipping")
            return None

        logger.info(f"Processing new voice: {voice_name}")
        speaker_latent, speaker_mask = self._preprocess_voice(wav_path)

        # Cache the preprocessed data
        pkl_path = wav_path.with_suffix('.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump((speaker_latent, speaker_mask), f)

        self.voices[voice_name] = (speaker_latent, speaker_mask)
        logger.info(f"Added voice '{voice_name}': latent shape {speaker_latent.shape}")

        return voice_name

    def remove_voice(self, voice_name: str) -> bool:
        """
        Remove a voice from loaded voices.

        Args:
            voice_name: Name of the voice to remove

        Returns:
            True if removed, False if not found
        """
        if voice_name not in self.voices:
            logger.info(f"Voice '{voice_name}' not loaded, nothing to remove")
            return False

        del self.voices[voice_name]
        logger.info(f"Removed voice '{voice_name}'")
        return True
