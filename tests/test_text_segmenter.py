import pytest
from longecho.text_segmenter import segment_text


def test_segment_short_text():
    """Short text should return single chunk"""
    text = "This is a short sentence."
    chunks = segment_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_segment_at_sentence_boundary():
    """Should split at sentence boundaries"""
    # Create text with two sentences, each within MAX_CHUNK_SIZE (220 chars)
    # First sentence ~200 chars, second ~100 chars = 301 total > MAX_CHUNK_SIZE
    sentence1 = "A" * 200 + "."
    sentence2 = "B" * 100 + "."
    text = sentence1 + " " + sentence2

    chunks = segment_text(text)
    assert len(chunks) == 2
    # First chunk should contain sentence1
    assert "A" in chunks[0] and chunks[0].strip().endswith(".")
    # Second chunk should contain sentence2
    assert "B" in chunks[1] and chunks[1].strip().endswith(".")


def test_segment_no_sentence_boundary():
    """Should fall back to comma if no sentence boundary"""
    # Text with comma but no period within range (under MAX_CHUNK_SIZE of 220)
    # ~200 chars with comma, then more text pushing total over MAX_CHUNK_SIZE
    text = "A" * 200 + ", " + "B" * 80
    chunks = segment_text(text)

    assert len(chunks) == 2
    assert "," in chunks[0]
    assert "B" in chunks[1]


def test_segment_hard_cut():
    """Should hard cut if no punctuation or spaces"""
    text = "A" * 1000  # No punctuation or spaces at all
    chunks = segment_text(text)

    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        # MAX_CHUNK_SIZE is 220
        assert len(chunk) <= 250


def test_segment_at_word_boundary():
    """Should split at word boundary when no punctuation available"""
    # Create text with words but no punctuation - forces word boundary fallback
    # Words of ~10 chars each, enough to exceed MAX_CHUNK_SIZE (220)
    words = ["wordnumber"] * 30  # 300 chars of words + 29 spaces = 329 chars
    text = " ".join(words)

    chunks = segment_text(text)

    assert len(chunks) >= 2
    # Each chunk should end with a complete word, not mid-word
    for chunk in chunks:
        # Chunk should not end with partial word (no trailing letters without space before)
        assert not chunk.endswith("wordnumbe")  # Partial word
        assert not chunk.endswith("wordnum")
        assert not chunk.endswith("wordn")
        # Should end with complete word
        assert chunk.rstrip().endswith("wordnumber") or len(chunk) < 10


def test_colon_followed_by_newline_no_extra_punctuation():
    """Newline after colon should not add extra period or comma"""
    # Single newline after colon
    text = "Here is a list:\nitem one"
    chunks = segment_text(text)
    result = chunks[0]
    # Should not have ":," or ":."
    assert ":," not in result
    assert ":." not in result
    # Colon should be preserved
    assert ":" in result

    # Double newline after colon (paragraph break)
    text2 = "Introduction:\n\nFirst paragraph"
    chunks2 = segment_text(text2)
    result2 = chunks2[0]
    assert ":," not in result2
    assert ":." not in result2


def test_segment_preserves_words_in_real_text():
    """Should never split words in realistic text without punctuation"""
    # Realistic text without punctuation marks (must exceed MAX_CHUNK_SIZE of 220)
    text = "The quick brown fox jumps over the lazy dog sleeping peacefully in the warm afternoon sun while birds sing melodiously in the tall oak trees nearby and squirrels gather acorns for the coming winter season as the gentle breeze carries the scent of pine through the forest clearing where deer graze quietly"

    chunks = segment_text(text)

    # Should split into multiple chunks
    assert len(chunks) >= 2, f"Text should split into multiple chunks, got {len(chunks)}"

    # Verify no chunk ends mid-word by checking all chunks end with a complete word
    # A complete word ends with a letter followed by end-of-string or was followed by space
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        # Get the last word of this chunk
        last_word = chunk.split()[-1] if chunk.split() else ""
        # This word should exist in the original text as a complete word
        assert f" {last_word} " in f" {text} " or text.startswith(last_word + " ") or text.endswith(" " + last_word), \
            f"Chunk {i} ends with partial word: '{last_word}'"
