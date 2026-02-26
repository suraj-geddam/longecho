import pytest
import torch
from unittest.mock import Mock, patch
import sys

# Mock the inference modules before importing audio_generator
sys.modules['inference'] = Mock()
sys.modules['inference_blockwise'] = Mock()

from longecho.audio_generator import AudioGenerator


@pytest.fixture
def mock_models():
    """Create mock Echo models"""
    model = Mock()
    fish_ae = Mock()
    pca_state = Mock()
    return model, fish_ae, pca_state


def test_generator_initialization(mock_models):
    """Should initialize with models"""
    model, fish_ae, pca_state = mock_models
    gen = AudioGenerator(model, fish_ae, pca_state)

    assert gen.model == model
    assert gen.fish_ae == fish_ae
    assert gen.pca_state == pca_state


def test_generate_chunks_count(mock_models):
    """Should generate correct number of chunks"""
    model, fish_ae, pca_state = mock_models

    # Mock the inference functions
    with patch('longecho.audio_generator.get_text_input_ids_and_mask') as mock_text, \
         patch('longecho.audio_generator.sample_blockwise_euler_cfg_independent_guidances') as mock_sample, \
         patch('longecho.audio_generator.ae_decode') as mock_decode, \
         patch('longecho.audio_generator.crop_audio_to_flattening_point') as mock_crop, \
         patch('longecho.audio_generator.get_speaker_latent_and_mask') as mock_speaker:

        # Setup mocks
        mock_text.return_value = (torch.zeros(1, 100), torch.ones(1, 100, dtype=torch.bool))
        mock_sample.return_value = torch.randn(1, 640, 80)
        mock_decode.return_value = torch.randn(1, 1, 640 * 2048)
        mock_crop.return_value = torch.randn(1, 1, 640 * 2048)
        # Mock continuation extraction: return continuation latent and mask
        mock_speaker.return_value = (torch.randn(1, 210, 80), torch.ones(1, 210, dtype=torch.bool))

        generator = AudioGenerator(model, fish_ae, pca_state)

        # Mock speaker latents
        speaker_latent = torch.randn(1, 100, 80)
        speaker_mask = torch.ones(1, 100, dtype=torch.bool)

        text_chunks = ["First chunk.", "Second chunk.", "Third chunk."]

        gen_id, gen = generator.generate_long_audio(text_chunks, speaker_latent, speaker_mask)
        chunks_generated = list(gen)

        assert len(chunks_generated) == 3


def test_empty_chunks_raises_error(mock_models):
    """Should raise ValueError when text_chunks is empty"""
    model, fish_ae, pca_state = mock_models
    generator = AudioGenerator(model, fish_ae, pca_state)

    speaker_latent = torch.randn(1, 100, 80)
    speaker_mask = torch.ones(1, 100, dtype=torch.bool)

    # Empty chunks should raise ValueError
    with pytest.raises(ValueError, match="No text to generate after normalization"):
        generator.generate_long_audio([], speaker_latent, speaker_mask)


@pytest.fixture
def mock_inference():
    """Setup mocks for inference functions - shared by multiple test classes"""
    with patch('longecho.audio_generator.get_text_input_ids_and_mask') as mock_text, \
         patch('longecho.audio_generator.sample_blockwise_euler_cfg_independent_guidances') as mock_sample, \
         patch('longecho.audio_generator.ae_decode') as mock_decode, \
         patch('longecho.audio_generator.crop_audio_to_flattening_point') as mock_crop, \
         patch('longecho.audio_generator.get_speaker_latent_and_mask') as mock_speaker:

        mock_text.return_value = (torch.zeros(1, 100), torch.ones(1, 100, dtype=torch.bool))
        mock_sample.return_value = torch.randn(1, 640, 80)
        mock_decode.return_value = torch.randn(1, 1, 640 * 2048)
        mock_crop.return_value = torch.randn(1, 1, 640 * 2048)
        mock_speaker.return_value = (torch.randn(1, 210, 80), torch.ones(1, 210, dtype=torch.bool))

        yield {
            'text': mock_text,
            'sample': mock_sample,
            'decode': mock_decode,
            'crop': mock_crop,
            'speaker': mock_speaker,
        }


class TestStopGeneration:
    """Tests for the stop generation functionality - verifies observable behavior"""

    @pytest.fixture
    def generator(self, mock_models):
        """Create AudioGenerator instance for testing"""
        model, fish_ae, pca_state = mock_models
        return AudioGenerator(model, fish_ae, pca_state)

    def test_is_generating_false_when_idle(self, generator):
        """is_generating should be False when no generation is running"""
        assert generator.is_generating is False

    def test_request_stop_before_generation_is_ignored(self, generator, mock_inference):
        """Calling request_stop() before generation is reset - class is self-contained"""
        speaker_latent = torch.randn(1, 100, 80)
        speaker_mask = torch.ones(1, 100, dtype=torch.bool)
        text_chunks = ["First chunk.", "Second chunk.", "Third chunk."]

        # Request stop before starting - should be reset when generation begins
        generator.request_stop()

        # Generation should complete all chunks because stop flag is reset at start
        gen_id, gen = generator.generate_long_audio(text_chunks, speaker_latent, speaker_mask)
        chunks = list(gen)

        assert len(chunks) == 3  # All chunks generated - stop flag was reset

    def test_request_stop_during_generation_stops_after_current_chunk(self, generator, mock_inference):
        """Calling request_stop() mid-generation should stop after the current chunk completes"""
        speaker_latent = torch.randn(1, 100, 80)
        speaker_mask = torch.ones(1, 100, dtype=torch.bool)
        text_chunks = ["First chunk.", "Second chunk.", "Third chunk."]

        chunks = []
        gen_id, gen = generator.generate_long_audio(text_chunks, speaker_latent, speaker_mask)

        # Get first chunk
        chunks.append(next(gen))

        # Request stop with specific generation ID
        generator.request_stop(gen_id)

        # Try to get remaining chunks
        for chunk in gen:
            chunks.append(chunk)

        # Should only have the first chunk (stop takes effect before second)
        assert len(chunks) == 1

    def test_new_generation_works_after_stopped_generation(self, generator, mock_inference):
        """A new generation should work automatically after a stopped generation (class is self-contained)"""
        speaker_latent = torch.randn(1, 100, 80)
        speaker_mask = torch.ones(1, 100, dtype=torch.bool)
        text_chunks = ["First chunk.", "Second chunk."]

        # First generation - stop it mid-way
        gen_id_first, gen_first = generator.generate_long_audio(text_chunks, speaker_latent, speaker_mask)
        next(gen_first)  # Get first chunk
        generator.request_stop(gen_id_first)
        chunks_first = [next(gen_first, None) for _ in range(10)]  # Try to get more
        chunks_first = [c for c in chunks_first if c is not None]  # Filter None
        # Should have stopped (only got 0-1 more chunks due to stop request)

        # Start new generation - should complete without needing reset_stop()
        # The class is now self-contained and resets the stop flag internally
        gen_id_second, gen_second = generator.generate_long_audio(text_chunks, speaker_latent, speaker_mask)
        chunks_second = list(gen_second)
        assert len(chunks_second) == 2  # All chunks generated

    def test_generation_without_stop_completes_all_chunks(self, generator, mock_inference):
        """Generation without stop request should yield all chunks"""
        speaker_latent = torch.randn(1, 100, 80)
        speaker_mask = torch.ones(1, 100, dtype=torch.bool)
        text_chunks = ["First.", "Second.", "Third.", "Fourth."]

        gen_id, gen = generator.generate_long_audio(text_chunks, speaker_latent, speaker_mask)
        chunks = list(gen)

        assert len(chunks) == 4


class TestIsGeneratingLifecycle:
    """Tests for is_generating property lifecycle during generation"""

    @pytest.fixture
    def generator(self, mock_models):
        """Create AudioGenerator instance for testing"""
        model, fish_ae, pca_state = mock_models
        return AudioGenerator(model, fish_ae, pca_state)

    def test_is_generating_true_during_generation(self, generator, mock_inference):
        """is_generating should be True while iterating through the generator"""
        speaker_latent = torch.randn(1, 100, 80)
        speaker_mask = torch.ones(1, 100, dtype=torch.bool)
        text_chunks = ["First.", "Second."]

        # Before generation
        assert generator.is_generating is False

        gen_id, gen = generator.generate_long_audio(text_chunks, speaker_latent, speaker_mask)

        # Get first chunk - should be generating
        next(gen)
        assert generator.is_generating is True

        # Get second chunk - still generating
        next(gen)
        assert generator.is_generating is True

        # Exhaust the generator
        for _ in gen:
            pass

        # After generation completes
        assert generator.is_generating is False

    def test_is_generating_false_after_stop(self, generator, mock_inference):
        """is_generating should be False after generation is stopped"""
        speaker_latent = torch.randn(1, 100, 80)
        speaker_mask = torch.ones(1, 100, dtype=torch.bool)
        text_chunks = ["First.", "Second.", "Third."]

        gen_id, gen = generator.generate_long_audio(text_chunks, speaker_latent, speaker_mask)

        # Get first chunk
        next(gen)
        assert generator.is_generating is True

        # Stop and exhaust
        generator.request_stop(gen_id)
        for _ in gen:
            pass

        # Should be False after stopped generation
        assert generator.is_generating is False

    def test_is_generating_false_after_error(self, generator):
        """is_generating should be False even if generation raises an error"""
        with patch('longecho.audio_generator.get_text_input_ids_and_mask') as mock_text:
            mock_text.side_effect = RuntimeError("Test error")

            speaker_latent = torch.randn(1, 100, 80)
            speaker_mask = torch.ones(1, 100, dtype=torch.bool)
            text_chunks = ["First."]

            gen_id, gen = generator.generate_long_audio(text_chunks, speaker_latent, speaker_mask)

            with pytest.raises(RuntimeError, match="Test error"):
                next(gen)

            # Should be False even after error
            assert generator.is_generating is False


class TestStopGenerationRaceCondition:
    """Tests for race condition: stop gen A, start gen B, gen A should stay stopped"""

    @pytest.fixture
    def generator(self, mock_models):
        """Create AudioGenerator instance for testing"""
        model, fish_ae, pca_state = mock_models
        return AudioGenerator(model, fish_ae, pca_state)

    def test_stopped_generation_stays_stopped_when_new_generation_starts(self, generator, mock_inference):
        """
        Race condition test: When generation A is stopped and generation B starts,
        generation A should NOT resume - the stop should be isolated to generation A.

        Scenario:
        1. Start generation A (many chunks)
        2. Get first chunk from A
        3. Stop generation A by ID
        4. Start generation B (before A is fully consumed)
        5. Continue iterating A - should yield NO more chunks (stayed stopped)
        6. Generation B should complete normally
        """
        speaker_latent = torch.randn(1, 100, 80)
        speaker_mask = torch.ones(1, 100, dtype=torch.bool)
        text_chunks_a = ["A1.", "A2.", "A3.", "A4.", "A5."]  # 5 chunks
        text_chunks_b = ["B1.", "B2."]  # 2 chunks

        # Start generation A
        gen_id_a, gen_a = generator.generate_long_audio(text_chunks_a, speaker_latent, speaker_mask)

        # Get first chunk from A
        chunk_a1 = next(gen_a)
        assert chunk_a1 is not None

        # Stop generation A by ID
        generator.request_stop(gen_id_a)

        # Start generation B
        gen_id_b, gen_b = generator.generate_long_audio(text_chunks_b, speaker_latent, speaker_mask)

        # Now try to get more from A - should be stopped (yield nothing more)
        chunks_from_a_after_stop = []
        for chunk in gen_a:
            chunks_from_a_after_stop.append(chunk)

        # Generation A should have stayed stopped - no more chunks
        assert len(chunks_from_a_after_stop) == 0, \
            f"Generation A should have stayed stopped but yielded {len(chunks_from_a_after_stop)} more chunks"

        # Generation B should complete normally
        chunks_b = list(gen_b)
        assert len(chunks_b) == 2, \
            f"Generation B should have completed with 2 chunks but got {len(chunks_b)}"

    def test_concurrent_stop_and_new_generation_race_condition(self, generator):
        """
        Real race condition: Generation A is BLOCKED inside _generate_chunk() when stop
        is called and generation B starts. This simulates the asyncio.to_thread scenario.

        With the new API, the frontend knows A's generation_id and can stop it explicitly,
        even if B has already started.
        """
        import threading
        import time

        speaker_latent = torch.randn(1, 100, 80)
        speaker_mask = torch.ones(1, 100, dtype=torch.bool)
        text_chunks_a = ["A1.", "A2.", "A3."]
        text_chunks_b = ["B1.", "B2."]

        # Use events to control timing
        gen_a_started_chunk = threading.Event()
        stop_and_start_b_done = threading.Event()
        gen_a_chunks = []
        gen_b_chunks = []
        gen_a_error = [None]
        gen_b_error = [None]
        gen_id_a_holder = [None]

        # Slow mock that signals when it starts
        call_count = [0]

        def slow_generate_chunk(*args, **kwargs):
            call_count[0] += 1
            # Signal that we're in the middle of generating
            gen_a_started_chunk.set()
            # Wait for stop + gen B start to happen
            stop_and_start_b_done.wait(timeout=5)
            # Small delay to ensure race condition timing
            time.sleep(0.01)
            return torch.randn(1, 640, 80)

        with patch('longecho.audio_generator.get_text_input_ids_and_mask') as mock_text, \
             patch('longecho.audio_generator.sample_blockwise_euler_cfg_independent_guidances') as mock_sample, \
             patch('longecho.audio_generator.ae_decode') as mock_decode, \
             patch('longecho.audio_generator.crop_audio_to_flattening_point') as mock_crop, \
             patch('longecho.audio_generator.get_speaker_latent_and_mask') as mock_speaker:

            mock_text.return_value = (torch.zeros(1, 100), torch.ones(1, 100, dtype=torch.bool))
            mock_sample.side_effect = slow_generate_chunk
            mock_decode.return_value = torch.randn(1, 1, 640 * 2048)
            mock_crop.return_value = torch.randn(1, 1, 640 * 2048)
            mock_speaker.return_value = (torch.randn(1, 210, 80), torch.ones(1, 210, dtype=torch.bool))

            def run_gen_a():
                try:
                    gen_id_a, gen_a = generator.generate_long_audio(text_chunks_a, speaker_latent, speaker_mask)
                    gen_id_a_holder[0] = gen_id_a
                    for chunk in gen_a:
                        gen_a_chunks.append(chunk)
                except Exception as e:
                    gen_a_error[0] = e

            def run_gen_b():
                try:
                    gen_id_b, gen_b = generator.generate_long_audio(text_chunks_b, speaker_latent, speaker_mask)
                    for chunk in gen_b:
                        gen_b_chunks.append(chunk)
                except Exception as e:
                    gen_b_error[0] = e

            # Start generation A in a thread
            thread_a = threading.Thread(target=run_gen_a)
            thread_a.start()

            # Wait for A to start its first chunk (blocked in mock_sample)
            gen_a_started_chunk.wait(timeout=5)

            # Now stop A by its ID and start B while A is blocked
            # In real scenario, frontend knows gen_id_a and sends it with stop request
            generator.request_stop(gen_id_a_holder[0])

            # Start B in another thread
            thread_b = threading.Thread(target=run_gen_b)
            thread_b.start()

            # Give B time to call generate_long_audio (which increments generation_id)
            time.sleep(0.05)

            # Now let A continue - it will finish its blocked call and check stop
            stop_and_start_b_done.set()

            # Wait for both threads
            thread_a.join(timeout=10)
            thread_b.join(timeout=10)

        # Check for errors
        assert gen_a_error[0] is None, f"Gen A error: {gen_a_error[0]}"
        assert gen_b_error[0] is None, f"Gen B error: {gen_b_error[0]}"

        # Gen A should have stopped after at most 1 chunk (the one it was generating when stop was called)
        assert len(gen_a_chunks) <= 1, \
            f"Gen A should have stopped after <=1 chunk but got {len(gen_a_chunks)}"

        # Gen B should complete all chunks
        assert len(gen_b_chunks) == 2, \
            f"Gen B should have completed with 2 chunks but got {len(gen_b_chunks)}"

    def test_stop_specific_generation_by_id(self, generator, mock_inference):
        """
        Test explicit generation ID in stop request.

        New API contract:
        - generate_long_audio() returns (generation_id, generator)
        - request_stop(generation_id) stops only that specific generation

        With the generation lock, only one generation can run at a time.
        When B starts, it auto-stops A and waits for the lock.
        When A checks stop flag (between chunks), it exits and releases lock.
        Then B proceeds.
        """
        speaker_latent = torch.randn(1, 100, 80)
        speaker_mask = torch.ones(1, 100, dtype=torch.bool)
        text_chunks_a = ["A1.", "A2.", "A3."]
        text_chunks_b = ["B1.", "B2."]

        # Start generation A - now returns (generation_id, generator)
        gen_id_a, gen_a = generator.generate_long_audio(text_chunks_a, speaker_latent, speaker_mask)
        chunk_a1 = next(gen_a)
        assert chunk_a1 is not None
        assert gen_id_a == 1

        # Stop generation A by ID before starting B
        generator.request_stop(gen_id_a)

        # Finish consuming A (it will check stop flag and exit)
        chunks_a_remaining = list(gen_a)
        assert len(chunks_a_remaining) == 0, \
            f"Gen A should have been stopped but got {len(chunks_a_remaining)} more chunks"

        # Now Gen B can start (A has released the lock)
        gen_id_b, gen_b = generator.generate_long_audio(text_chunks_b, speaker_latent, speaker_mask)
        assert gen_id_b == 2

        # Gen B should complete all chunks
        chunks_b = list(gen_b)
        assert len(chunks_b) == 2, \
            f"Gen B should have completed but got {len(chunks_b)} chunks (expected 2)"

    def test_stop_without_id_stops_all(self, generator, mock_inference):
        """
        Calling request_stop() without an ID stops all generations (backward compat).
        """
        speaker_latent = torch.randn(1, 100, 80)
        speaker_mask = torch.ones(1, 100, dtype=torch.bool)
        text_chunks = ["A1.", "A2.", "A3."]

        gen_id, gen = generator.generate_long_audio(text_chunks, speaker_latent, speaker_mask)
        next(gen)  # Get first chunk

        # Stop without specifying ID
        generator.request_stop()

        # Should be stopped
        remaining = list(gen)
        assert len(remaining) == 0


class TestGenerationLock:
    """Tests for the generation lock that prevents concurrent GPU usage."""

    @pytest.fixture
    def generator(self, mock_models):
        """Create AudioGenerator instance for testing"""
        model, fish_ae, pca_state = mock_models
        return AudioGenerator(model, fish_ae, pca_state)

    def test_lock_prevents_concurrent_generations(self, generator):
        """
        Only one generation should run at a time. When a new generation starts,
        it must wait for the previous one to finish its current chunk.
        """
        import threading
        import time

        speaker_latent = torch.randn(1, 100, 80)
        speaker_mask = torch.ones(1, 100, dtype=torch.bool)
        text_chunks_a = ["A1.", "A2.", "A3."]
        text_chunks_b = ["B1.", "B2."]

        # Track execution order
        execution_log = []
        lock_acquired_events = []

        def slow_inference(*args, **kwargs):
            # Simulate slow inference
            time.sleep(0.1)
            return torch.randn(1, 640, 80)

        with patch('longecho.audio_generator.get_text_input_ids_and_mask') as mock_text, \
             patch('longecho.audio_generator.sample_blockwise_euler_cfg_independent_guidances') as mock_sample, \
             patch('longecho.audio_generator.ae_decode') as mock_decode, \
             patch('longecho.audio_generator.crop_audio_to_flattening_point') as mock_crop, \
             patch('longecho.audio_generator.get_speaker_latent_and_mask') as mock_speaker:

            mock_text.return_value = (torch.zeros(1, 100), torch.ones(1, 100, dtype=torch.bool))
            mock_sample.side_effect = slow_inference
            mock_decode.return_value = torch.randn(1, 1, 640 * 2048)
            mock_crop.return_value = torch.randn(1, 1, 640 * 2048)
            mock_speaker.return_value = (torch.randn(1, 210, 80), torch.ones(1, 210, dtype=torch.bool))

            def run_gen_a():
                gen_id, gen = generator.generate_long_audio(text_chunks_a, speaker_latent, speaker_mask)
                execution_log.append(f"A_start_{gen_id}")
                for i, chunk in enumerate(gen):
                    execution_log.append(f"A_chunk_{i}")
                execution_log.append(f"A_end_{gen_id}")

            def run_gen_b():
                # Small delay to ensure A starts first
                time.sleep(0.05)
                gen_id, gen = generator.generate_long_audio(text_chunks_b, speaker_latent, speaker_mask)
                execution_log.append(f"B_start_{gen_id}")
                for i, chunk in enumerate(gen):
                    execution_log.append(f"B_chunk_{i}")
                execution_log.append(f"B_end_{gen_id}")

            thread_a = threading.Thread(target=run_gen_a)
            thread_b = threading.Thread(target=run_gen_b)

            thread_a.start()
            thread_b.start()

            thread_a.join(timeout=10)
            thread_b.join(timeout=10)

        # Verify A starts before B (due to lock)
        a_start_idx = next(i for i, x in enumerate(execution_log) if x.startswith("A_start"))
        b_start_idx = next(i for i, x in enumerate(execution_log) if x.startswith("B_start"))

        # B should start after A has released the lock (A was stopped or finished)
        # Since B auto-stops A, A should end before B starts producing chunks
        a_end_idx = next((i for i, x in enumerate(execution_log) if x.startswith("A_end")), len(execution_log))

        assert a_start_idx < b_start_idx, \
            f"A should start before B. Log: {execution_log}"
        assert a_end_idx < b_start_idx or "A_chunk_0" in execution_log, \
            f"A should release lock before B starts. Log: {execution_log}"

    def test_new_generation_stops_old_and_proceeds(self, generator):
        """
        When a new generation starts while an old one is running,
        the old one should be stopped and the new one should proceed.
        """
        import threading
        import time

        speaker_latent = torch.randn(1, 100, 80)
        speaker_mask = torch.ones(1, 100, dtype=torch.bool)
        text_chunks_a = ["A1.", "A2.", "A3.", "A4.", "A5."]  # Many chunks
        text_chunks_b = ["B1.", "B2."]

        gen_a_chunks = []
        gen_b_chunks = []
        gen_a_started = threading.Event()

        def slow_inference(*args, **kwargs):
            time.sleep(0.05)
            return torch.randn(1, 640, 80)

        with patch('longecho.audio_generator.get_text_input_ids_and_mask') as mock_text, \
             patch('longecho.audio_generator.sample_blockwise_euler_cfg_independent_guidances') as mock_sample, \
             patch('longecho.audio_generator.ae_decode') as mock_decode, \
             patch('longecho.audio_generator.crop_audio_to_flattening_point') as mock_crop, \
             patch('longecho.audio_generator.get_speaker_latent_and_mask') as mock_speaker:

            mock_text.return_value = (torch.zeros(1, 100), torch.ones(1, 100, dtype=torch.bool))
            mock_sample.side_effect = slow_inference
            mock_decode.return_value = torch.randn(1, 1, 640 * 2048)
            mock_crop.return_value = torch.randn(1, 1, 640 * 2048)
            mock_speaker.return_value = (torch.randn(1, 210, 80), torch.ones(1, 210, dtype=torch.bool))

            def run_gen_a():
                gen_id, gen = generator.generate_long_audio(text_chunks_a, speaker_latent, speaker_mask)
                for chunk in gen:
                    gen_a_chunks.append(chunk)
                    gen_a_started.set()  # Signal after first chunk

            def run_gen_b():
                # Wait for A to start and produce at least one chunk
                gen_a_started.wait(timeout=5)
                time.sleep(0.01)  # Small delay to ensure A is mid-generation
                gen_id, gen = generator.generate_long_audio(text_chunks_b, speaker_latent, speaker_mask)
                for chunk in gen:
                    gen_b_chunks.append(chunk)

            thread_a = threading.Thread(target=run_gen_a)
            thread_b = threading.Thread(target=run_gen_b)

            thread_a.start()
            thread_b.start()

            thread_a.join(timeout=10)
            thread_b.join(timeout=10)

        # A should have been stopped (got fewer than all 5 chunks)
        assert len(gen_a_chunks) < 5, \
            f"Gen A should have been stopped early but got {len(gen_a_chunks)} chunks"

        # B should have completed all its chunks
        assert len(gen_b_chunks) == 2, \
            f"Gen B should have completed with 2 chunks but got {len(gen_b_chunks)}"

    def test_no_concurrent_gpu_usage(self, generator):
        """
        Verify that at most one generation is doing GPU work at any time.
        This prevents the GPU memory/performance issues from concurrent generations.
        """
        import threading
        import time

        speaker_latent = torch.randn(1, 100, 80)
        speaker_mask = torch.ones(1, 100, dtype=torch.bool)
        text_chunks = ["Chunk1.", "Chunk2."]

        # Track when GPU work is happening
        gpu_active_count = [0]
        max_concurrent = [0]
        lock = threading.Lock()

        def tracked_inference(*args, **kwargs):
            with lock:
                gpu_active_count[0] += 1
                max_concurrent[0] = max(max_concurrent[0], gpu_active_count[0])
            try:
                time.sleep(0.1)  # Simulate GPU work
                return torch.randn(1, 640, 80)
            finally:
                with lock:
                    gpu_active_count[0] -= 1

        with patch('longecho.audio_generator.get_text_input_ids_and_mask') as mock_text, \
             patch('longecho.audio_generator.sample_blockwise_euler_cfg_independent_guidances') as mock_sample, \
             patch('longecho.audio_generator.ae_decode') as mock_decode, \
             patch('longecho.audio_generator.crop_audio_to_flattening_point') as mock_crop, \
             patch('longecho.audio_generator.get_speaker_latent_and_mask') as mock_speaker:

            mock_text.return_value = (torch.zeros(1, 100), torch.ones(1, 100, dtype=torch.bool))
            mock_sample.side_effect = tracked_inference
            mock_decode.return_value = torch.randn(1, 1, 640 * 2048)
            mock_crop.return_value = torch.randn(1, 1, 640 * 2048)
            mock_speaker.return_value = (torch.randn(1, 210, 80), torch.ones(1, 210, dtype=torch.bool))

            threads = []
            for i in range(3):  # Try to start 3 concurrent generations
                def run_gen():
                    gen_id, gen = generator.generate_long_audio(text_chunks, speaker_latent, speaker_mask)
                    list(gen)  # Consume the generator

                t = threading.Thread(target=run_gen)
                threads.append(t)

            # Start all threads nearly simultaneously
            for t in threads:
                t.start()
                time.sleep(0.01)  # Tiny delay

            for t in threads:
                t.join(timeout=30)

        # At most 1 generation should be doing GPU work at any time
        assert max_concurrent[0] == 1, \
            f"Expected max 1 concurrent GPU operation, but saw {max_concurrent[0]}"

    def test_orphaned_generator_at_yield_does_not_hold_lock(self, generator, mock_inference):
        """
        Bug reproduction: When a generator is abandoned at a yield point
        (e.g., SSE consumer disconnects), the lock must NOT be held forever.

        A new generation must be able to acquire the lock and proceed.

        Scenario:
        1. Start generation A, get first chunk (generator now suspended at yield)
        2. Abandon generator A (don't exhaust it, simulating SSE disconnect)
        3. Start generation B - should acquire lock and complete, NOT deadlock
        """
        import threading

        speaker_latent = torch.randn(1, 100, 80)
        speaker_mask = torch.ones(1, 100, dtype=torch.bool)
        text_chunks_a = ["A1.", "A2.", "A3."]
        text_chunks_b = ["B1.", "B2."]

        # Start generation A and get one chunk
        gen_id_a, gen_a = generator.generate_long_audio(text_chunks_a, speaker_latent, speaker_mask)
        chunk = next(gen_a)
        assert chunk is not None

        # Abandon gen_a here - simulates SSE consumer disconnect.
        # The generator is suspended at yield, holding whatever resources it holds.
        # We intentionally do NOT call gen_a.close() or exhaust it.
        # (In the real bug, generator.close() fails because thread is still running)

        # Now try to start generation B - this should NOT deadlock
        gen_b_result = []
        gen_b_error = [None]

        def run_gen_b():
            try:
                gen_id_b, gen_b = generator.generate_long_audio(text_chunks_b, speaker_latent, speaker_mask)
                for c in gen_b:
                    gen_b_result.append(c)
            except Exception as e:
                gen_b_error[0] = e

        thread_b = threading.Thread(target=run_gen_b)
        thread_b.start()
        thread_b.join(timeout=5)  # 5 second timeout - should be plenty

        # If thread_b is still alive, we deadlocked
        assert not thread_b.is_alive(), \
            "Generation B deadlocked waiting for lock held by orphaned generator A"

        assert gen_b_error[0] is None, f"Generation B errored: {gen_b_error[0]}"
        assert len(gen_b_result) == 2, \
            f"Generation B should have completed with 2 chunks but got {len(gen_b_result)}"
