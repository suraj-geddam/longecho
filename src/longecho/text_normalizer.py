"""
Text normalization for TTS preprocessing.

Converts text patterns that TTS models struggle with into speakable forms:
- "$5M" → "5 million dollars"
- "Dr. Smith" → "Doctor Smith"
- "(important note)" → "important note" (removes parens, keeps content)

Usage:
    normalizer = TextNormalizer(level="moderate")
    result = normalizer.normalize("The cost is $5M.")
"""

import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

NormalizationLevel = Literal["moderate", "full"]


class TextNormalizer:
    """
    Text normalizer for TTS preprocessing.

    Supports two normalization levels:
    - "moderate": Normalize currencies, abbreviations, dates, times, measures.
                  Leaves plain numbers (e.g., "123") as-is.
    - "full": Normalize everything including plain numbers to words.
    """

    def __init__(self, level: NormalizationLevel = "moderate") -> None:
        """
        Initialize the text normalizer.

        Args:
            level: Normalization level - "moderate" or "full"
        """
        if level not in ("moderate", "full"):
            raise ValueError(f"Invalid normalization level: {level}. Must be 'moderate' or 'full'.")
        self._level = level

    @property
    def level(self) -> NormalizationLevel:
        """Current normalization level."""
        return self._level

    def normalize(self, text: str) -> str:
        """
        Normalize text for TTS processing.

        Pipeline:
        1. Expand currencies, abbreviations, etc.
        2. If level is "full", convert plain numbers to words
        3. Strip parentheses but keep content inside

        Args:
            text: Input text to normalize

        Returns:
            Normalized text ready for TTS
        """
        if not text or not text.strip():
            return text

        try:
            # Step 1-2: Apply normalization
            normalized = self._apply_normalization(text)

            # Step 3: Strip parentheses but keep content
            normalized = self._strip_parentheses(normalized)

            return normalized

        except Exception as e:
            logger.warning(f"Text normalization failed, returning original: {e}")
            return text

    def _apply_normalization(self, text: str) -> str:
        """
        Apply regex-based normalization.

        Handles common patterns:
        - Currency: $5, $5M, $5B, $5K
        - Common abbreviations: Dr., Mr., Mrs., Ms., etc.
        - Numbers to words (full mode only)
        """
        result = text

        # Currency with multipliers: $5M, $5B, $5K, $5T
        result = re.sub(
            r'\$(\d+(?:\.\d+)?)\s*([KkMmBbTt])\b',
            self._expand_currency_multiplier,
            result
        )

        # Simple currency: $5, $123.45
        result = re.sub(
            r'\$(\d+(?:\.\d+)?)\b',
            self._expand_simple_currency,
            result
        )

        # Common abbreviations
        abbreviations = {
            r'\bDr\.': 'Doctor',
            r'\bMr\.': 'Mister',
            r'\bMrs\.': 'Missus',
            r'\bMs\.': 'Miss',
            r'\bProf\.': 'Professor',
            r'\bSt\.': 'Saint',
            r'\bJr\.': 'Junior',
            r'\bSr\.': 'Senior',
            r'\bvs\.': 'versus',
            r'\betc\.': 'etcetera',
            r'\be\.g\.': 'for example',
            r'\bi\.e\.': 'that is',
        }
        for pattern, replacement in abbreviations.items():
            result = re.sub(pattern, replacement, result)

        # Symbol expansions for TTS
        # Order matters - do these before number conversion
        result = re.sub(r'\s*=\s*', ' equals ', result)  # = to equals
        result = re.sub(r'\s*\+\s*', ' plus ', result)   # + to plus
        result = re.sub(r'&', ' and ', result)           # & to and
        result = re.sub(r'(\d+)\s*%', r'\1 percent', result)  # 50% to 50 percent
        result = re.sub(r'\s*@\s*', ' at ', result)      # @ to at
        # Only expand slash between pure alphabetic words, but NOT in URLs/paths
        # Require space or start-of-string before the first word to avoid URL segments
        result = re.sub(r'(?<![:/.\w])([a-zA-Z]+)/([a-zA-Z]+)(?![:/.\w])', r'\1 or \2', result)

        # Clean up extra spaces from symbol expansion
        result = re.sub(r' +', ' ', result)

        # Convert numbers to words only in full mode
        if self._level == "full":
            result = re.sub(r'\b(\d+)\b', self._number_to_words, result)

        return result

    def _expand_currency_multiplier(self, match: re.Match) -> str:
        """Expand currency with multiplier like $5M to 5 million dollars."""
        amount = match.group(1)
        multiplier = match.group(2).upper()

        multiplier_words = {
            'K': 'thousand',
            'M': 'million',
            'B': 'billion',
            'T': 'trillion',
        }

        word = multiplier_words.get(multiplier, '')
        return f"{amount} {word} dollars"

    def _expand_simple_currency(self, match: re.Match) -> str:
        """Expand simple currency like $5 to 5 dollars."""
        amount = match.group(1)
        if '.' in amount:
            dollars, cents = amount.split('.')
            if cents == '00':
                return f"{dollars} dollars"
            return f"{dollars} dollars and {cents} cents"
        return f"{amount} dollars"

    def _number_to_words(self, match: re.Match) -> str:
        """Convert a number to words (for numbers up to 999)."""
        num = int(match.group(1))
        return self._int_to_words(num) if num <= 999 else match.group(0)

    def _int_to_words(self, num: int) -> str:
        """Convert an integer (0-999) to words."""
        ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
                'seventeen', 'eighteen', 'nineteen']
        tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']

        if num == 0:
            return 'zero'
        if num < 20:
            return ones[num]
        if num < 100:
            return tens[num // 10] + ('' if num % 10 == 0 else ' ' + ones[num % 10])

        # 100-999
        hundreds = num // 100
        remainder = num % 100
        result = ones[hundreds] + ' hundred'
        if remainder > 0:
            if remainder < 20:
                result += ' ' + ones[remainder]
            else:
                result += ' ' + tens[remainder // 10]
                if remainder % 10 > 0:
                    result += ' ' + ones[remainder % 10]
        return result

    def _strip_parentheses(self, text: str) -> str:
        """
        Remove parentheses but keep the content inside, adding commas for speech pauses.

        "(important note)" → "important note"
        "Hello (world) there" → "Hello, world, there"

        WhisperD uses parens for non-speech events like "(coughs)",
        so we need to strip them to avoid content being skipped.
        Adding commas creates natural pauses where parentheses were.
        """
        # Replace parenthetical content with comma-delimited version
        # Pattern: text before ( content ) text after
        # We want: text before, content, text after

        def add_commas(match: re.Match) -> str:
            before = match.group(1)  # Text before opening paren
            content = match.group(2)  # Content inside parens
            after = match.group(3)    # Text after closing paren

            # Add comma before content if there's text before and no punctuation already
            if before and before.rstrip() and not before.rstrip()[-1] in ',:;-':
                before = before.rstrip() + ','

            # Add comma after content if there's text after (not just punctuation)
            after_stripped = after.lstrip()
            if after_stripped and after_stripped[0].isalnum():
                content = content + ','

            return f"{before} {content} {after}"

        # Process parentheses one at a time (handles nested by iterating)
        result = text
        while '(' in result and ')' in result:
            # Match: (optional text before)(content)(optional text after)
            new_result = re.sub(
                r'(^|.*?)(?<![(\s])\s*\(([^()]+)\)\s*(?![)\s])(.*)$',
                add_commas,
                result,
                count=1
            )
            if new_result == result:
                # Fallback: just strip remaining parens
                result = result.replace('(', '').replace(')', '')
                break
            result = new_result

        # Clean up any double spaces, double commas, or comma-space-comma
        result = re.sub(r',\s*,', ',', result)
        result = re.sub(r' +', ' ', result)
        # Remove space before punctuation
        result = re.sub(r' ([.,!?;:])', r'\1', result)

        return result.strip()
