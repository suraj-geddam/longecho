from typing import List

# Constants based on design
# Realistic speech rate: ~130 WPM = ~13 chars/second
CHARS_PER_SECOND = 13
TARGET_DURATION_SECONDS = 10
# Larger chunks for more natural speech flow
# Max ~220 chars = ~17s = ~375 latents, under MAX_CONTINUATION_LATENTS (400)
# Leaves ~265 latents for new generation (minimum needed is ~240)
MIN_CHUNK_SIZE = 160  # ~12 seconds minimum
MAX_CHUNK_SIZE = 220  # ~17 seconds maximum
FALLBACK_SEARCH_WINDOW = 100

SENTENCE_TERMINATORS = {'.', '!', '?'}
CLAUSE_SEPARATORS = {',', ';'}


def _normalize_text(text: str) -> str:
    """
    Normalize text to match Echo-TTS's preprocessing with smart pause handling.

    Converts line breaks to pauses:
    - Paragraph breaks (2+ newlines) → period (long pause) if no punctuation
    - Single line breaks → comma (short pause) if no punctuation
    - Preserves existing punctuation

    Args:
        text: Raw input text

    Returns:
        Normalized text with proper pauses
    """
    import re

    # Process text to handle newlines smartly
    result = []
    i = 0

    while i < len(text):
        if text[i] == '\n':
            # Count consecutive newlines
            newline_count = 0
            while i < len(text) and text[i] == '\n':
                newline_count += 1
                i += 1

            # Check if previous character was punctuation
            prev_char = result[-1] if result else ''
            has_punctuation = prev_char in '.!?,;:'

            # Paragraph break (2+ newlines) → add period for long pause
            if newline_count >= 2:
                if not has_punctuation:
                    result.append('.')
                result.append(' ')
            # Single newline → add comma for short pause
            else:
                if not has_punctuation:
                    result.append(',')
                result.append(' ')
        else:
            result.append(text[i])
            i += 1

    text = ''.join(result)

    # Collapse multiple spaces into single space
    text = re.sub(r' +', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def segment_text(text: str) -> List[str]:
    """
    Segment text into chunks suitable for Echo-TTS generation.

    Each chunk targets 20-25 seconds of audio (~240-390 chars at 13 chars/sec).
    Splits at sentence boundaries when possible, falls back to clauses,
    then hard cuts if necessary.

    Args:
        text: Input text to segment

    Returns:
        List of text chunks
    """
    # Normalize text to match Echo-TTS's preprocessing
    # This ensures accurate character counting and proper chunk boundaries
    text = _normalize_text(text)

    if len(text) <= MAX_CHUNK_SIZE:
        return [text]

    chunks = []
    remaining = text

    while len(remaining) > MAX_CHUNK_SIZE:
        # Start at max chunk size, search backward for boundary
        split_point = _find_split_point(remaining[:MAX_CHUNK_SIZE + FALLBACK_SEARCH_WINDOW])

        # Extract chunk and update remaining
        chunk = remaining[:split_point].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_point:].strip()

    # Add final chunk if any text remains
    if remaining:
        chunks.append(remaining)

    return chunks


def _find_split_point(text_window: str) -> int:
    """
    Find the best split point in the text window.

    Priority:
    1. Sentence boundary (. ! ?) near target size
    2. Clause separator (, ;) near target size
    3. Word boundary (space) near target size
    4. Hard cut at max size (only if no spaces at all)

    Args:
        text_window: Text to search for split point

    Returns:
        Index to split at
    """
    # Search backward from around MAX_CHUNK_SIZE for sentence terminator
    # Look in window from MAX_CHUNK_SIZE down to MIN_CHUNK_SIZE
    search_start = min(len(text_window), MAX_CHUNK_SIZE)
    search_end = max(0, MIN_CHUNK_SIZE)

    # Look for sentence boundaries (prefer closer to MAX_CHUNK_SIZE)
    for i in range(search_start - 1, search_end - 1, -1):
        if text_window[i] in SENTENCE_TERMINATORS:
            return i + 1

    # Fall back to clause separators in same range
    for i in range(search_start - 1, search_end - 1, -1):
        if text_window[i] in CLAUSE_SEPARATORS:
            return i + 1

    # Fall back to word boundaries - search FORWARD to complete current word
    # This gives larger chunks that end at natural word boundaries
    for i in range(search_start, len(text_window)):
        if text_window[i] == ' ':
            return i + 1  # Split after space, next chunk starts with word

    # If no space found forward, search backward (text might end soon)
    for i in range(search_start - 1, search_end - 1, -1):
        if text_window[i] == ' ':
            return i + 1

    # Hard cut at max size (only reached if no spaces at all, e.g., "AAAA...")
    return min(MAX_CHUNK_SIZE, len(text_window))
