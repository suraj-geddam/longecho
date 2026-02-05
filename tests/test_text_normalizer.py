"""Tests for text_normalizer module."""

import pytest


class TestTextNormalizerInit:
    """Tests for TextNormalizer initialization."""

    def test_init_default_level(self):
        """Default level should be moderate."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        assert normalizer.level == "moderate"

    def test_init_moderate_level(self):
        """Should accept moderate level."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer(level="moderate")
        assert normalizer.level == "moderate"

    def test_init_full_level(self):
        """Should accept full level."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer(level="full")
        assert normalizer.level == "full"

    def test_init_invalid_level(self):
        """Should raise ValueError for invalid level."""
        from longecho.text_normalizer import TextNormalizer
        with pytest.raises(ValueError, match="Invalid normalization level"):
            TextNormalizer(level="invalid")


class TestStripParentheses:
    """Tests for parentheses stripping functionality."""

    def test_strip_simple_parentheses(self):
        """Should remove parentheses and keep content with comma."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._strip_parentheses("(important note)")
        # Standalone parens just become the content
        assert result == "important note"

    def test_strip_parentheses_in_sentence(self):
        """Should add comma before parenthetical content for natural pause."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._strip_parentheses("Hello (world) there")
        # Should add comma before and after for speech pause
        assert result == "Hello, world, there"

    def test_strip_multiple_parentheses(self):
        """Should handle multiple sets of parentheses with commas."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._strip_parentheses("(first) and (second)")
        # Each parenthetical gets comma treatment
        assert result == "first, and, second"

    def test_strip_nested_parentheses(self):
        """Should handle nested parentheses."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._strip_parentheses("((nested))")
        assert result == "nested"

    def test_strip_parentheses_no_parens(self):
        """Should return text unchanged if no parentheses."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._strip_parentheses("no parens here")
        assert result == "no parens here"

    def test_strip_parentheses_cleans_double_spaces(self):
        """Should clean up double spaces after removing parentheses."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._strip_parentheses("word (removed) word")
        # Should have commas instead of spaces around parenthetical
        assert "  " not in result
        assert result == "word, removed, word"

    def test_strip_parentheses_at_end_of_sentence(self):
        """Should handle parentheses at end of sentence."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._strip_parentheses("See details (below).")
        # Comma before, period stays at end
        assert result == "See details, below."

    def test_strip_parentheses_preserves_existing_punctuation(self):
        """Should not double up punctuation."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._strip_parentheses("Hello, (world) there")
        # Already has comma before, don't add another
        assert result == "Hello, world, there"
        assert ",," not in result


class TestNormalization:
    """Tests for text normalization."""

    def test_currency_with_multiplier_millions(self):
        """Should expand $5M to 5 million dollars."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("The cost is $5M.")
        assert "5 million dollars" in result
        assert "$" not in result

    def test_currency_with_multiplier_billions(self):
        """Should expand $2.5B to 2.5 billion dollars."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("Revenue: $2.5B")
        assert "2.5 billion dollars" in result

    def test_currency_with_multiplier_thousands(self):
        """Should expand $100K to 100 thousand dollars."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("Salary: $100K")
        assert "100 thousand dollars" in result

    def test_simple_currency(self):
        """Should expand $5 to 5 dollars."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("It costs $5.")
        assert "5 dollars" in result
        assert "$" not in result

    def test_currency_with_cents(self):
        """Should expand $5.99 to 5 dollars and 99 cents."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("Price: $5.99")
        assert "5 dollars and 99 cents" in result

    def test_abbreviation_dr(self):
        """Should expand Dr. to Doctor."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("Dr. Smith arrived.")
        assert "Doctor Smith" in result
        assert "Dr." not in result

    def test_abbreviation_mr_mrs(self):
        """Should expand Mr. and Mrs."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("Mr. and Mrs. Jones")
        assert "Mister" in result
        assert "Missus" in result

    def test_abbreviation_etc(self):
        """Should expand etc."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("apples, oranges, etc.")
        assert "etcetera" in result

    def test_symbol_equals(self):
        """Should expand = to equals."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("experience = credibility")
        assert "experience equals credibility" in result
        assert " = " not in result

    def test_symbol_plus(self):
        """Should expand + to plus."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("2 + 2")
        assert "2 plus 2" in result

    def test_symbol_ampersand(self):
        """Should expand & to and."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("R&D department")
        assert "R and D" in result

    def test_symbol_percent(self):
        """Should expand % to percent."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("50% off")
        assert "50 percent" in result

    def test_symbol_at(self):
        """Should expand @ to at."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("email @ domain")
        assert "email at domain" in result

    def test_symbol_slash_as_or(self):
        """Should expand / to or in word contexts."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("yes/no question")
        assert "yes or no" in result

    def test_symbol_slash_preserves_urls(self):
        """Should NOT expand slashes in URLs."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("Visit http://example.com/path")
        assert "http://example.com/path" in result
        assert " or " not in result

    def test_symbol_slash_preserves_dates(self):
        """Should NOT expand slashes in dates."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("Date: 12/25/2024")
        assert "12/25/2024" in result
        assert " or " not in result

    def test_moderate_preserves_plain_numbers(self):
        """Moderate level should preserve plain numbers."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer(level="moderate")
        result = normalizer._apply_normalization("I have 42 items.")
        assert "42" in result

    def test_full_converts_numbers(self):
        """Full level should convert numbers to words."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer(level="full")
        result = normalizer._apply_normalization("I have 42 items.")
        assert "forty two" in result
        assert "42" not in result


class TestIntToWords:
    """Tests for the _int_to_words helper."""

    def test_zero(self):
        """Should convert 0 to zero."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        assert normalizer._int_to_words(0) == "zero"

    def test_single_digits(self):
        """Should convert single digits."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        assert normalizer._int_to_words(1) == "one"
        assert normalizer._int_to_words(5) == "five"
        assert normalizer._int_to_words(9) == "nine"

    def test_teens(self):
        """Should convert teen numbers."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        assert normalizer._int_to_words(11) == "eleven"
        assert normalizer._int_to_words(15) == "fifteen"
        assert normalizer._int_to_words(19) == "nineteen"

    def test_tens(self):
        """Should convert tens."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        assert normalizer._int_to_words(20) == "twenty"
        assert normalizer._int_to_words(42) == "forty two"
        assert normalizer._int_to_words(99) == "ninety nine"

    def test_hundreds(self):
        """Should convert hundreds."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        assert normalizer._int_to_words(100) == "one hundred"
        assert normalizer._int_to_words(123) == "one hundred twenty three"
        assert normalizer._int_to_words(500) == "five hundred"
        assert normalizer._int_to_words(999) == "nine hundred ninety nine"


class TestNormalizeEndToEnd:
    """End-to-end tests for the normalize() method."""

    def test_normalize_empty_string(self):
        """Empty string should return empty string."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        assert normalizer.normalize("") == ""
        assert normalizer.normalize("   ") == "   "

    def test_normalize_currency_and_parens(self):
        """Should handle currency and strip parentheses."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer.normalize("($5M)")
        assert "5 million dollars" in result
        assert "(" not in result
        assert ")" not in result

    def test_normalize_preserves_plain_text(self):
        """Plain text without special patterns should be preserved."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer.normalize("Hello world")
        assert result == "Hello world"

    def test_normalize_complex_sentence(self):
        """Should handle complex sentences."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer.normalize("Dr. Smith invested $5M (see details below).")
        assert "Doctor" in result
        assert "million dollars" in result
        assert "(" not in result
        assert ")" not in result

    def test_normalize_error_returns_original(self):
        """Should return original text if normalization fails."""
        from longecho.text_normalizer import TextNormalizer
        normalizer = TextNormalizer()
        # Normal text should work fine
        result = normalizer.normalize("Simple text")
        assert result == "Simple text"
