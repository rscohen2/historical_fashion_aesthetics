import pytest
from fashion.wearing import WearingLLM


@pytest.fixture
def llm():
    return WearingLLM()


def test_format_datum_basic(llm):
    """Test basic formatting with a single fashion term and entity."""
    datum = {
        "excerpt_text": "She was wearing a red hat.",
        "fashion_term": "hat",
        "fashion_start_idx": 22,
        "fashion_end_idx": 25,
        "excerpt_start": 0,
        "excerpt_end": 25,
        "characters": [
            {
                "character_start_idx": 0,
                "character_end_idx": 3,
                "coref": "123",
                "text": "She",
                "distance": 0,
            }
        ],
    }
    expected = "[She]_123 was wearing a red [hat]_fashion."
    assert llm.format_datum(datum) == expected


def test_format_datum_nested_entities(llm):
    """Test formatting with nested entity references."""
    datum = {
        "excerpt_text": "The young man was wearing a red hat. He was the man's son.",
        "fashion_term": "hat",
        "fashion_start_idx": 32,
        "fashion_end_idx": 35,
        "excerpt_start": 0,
        "excerpt_end": 50,
        "characters": [
            {
                "character_start_idx": 4,
                "character_end_idx": 13,
                "coref": "123",
                "text": "young man",
                "distance": 0,
            },
            {
                "character_start_idx": 37,
                "character_end_idx": 39,
                "coref": "123",
                "text": "He",
                "distance": 0,
            },
            {
                "character_start_idx": 48,
                "character_end_idx": 51,
                "coref": "456",
                "text": "man",
                "distance": 0,
            },
            {
                "character_start_idx": 44,
                "character_end_idx": 57,
                "text": "the man's son",
                "coref": "123",
                "distance": 0,
            },
        ],
    }
    expected = "The [young man]_123 was wearing a red [hat]_fashion. [He]_123 was [the [man]_456's son]_123."
    assert llm.format_datum(datum) == expected


def test_format_datum_complex_nesting(llm):
    """Test complex nested entity structure."""
    datum = {
        "excerpt_text": "The dress belonged to the mother of the hairdresser's wife.",
        "fashion_term": "dress",
        "fashion_start_idx": 4,
        "fashion_end_idx": 9,
        "excerpt_start": 0,
        "excerpt_end": 40,
        "characters": [
            {
                "character_start_idx": 26,
                "character_end_idx": 32,
                "coref": "3",
                "text": "mother",
                "distance": 0,
            },
            {
                "character_start_idx": 40,
                "character_end_idx": 51,
                "coref": "1",
                "text": "hairdresser",
                "distance": 0,
            },
            {
                "character_start_idx": 40,
                "character_end_idx": 58,
                "coref": "2",
                "text": "hairdresser's wife",
                "distance": 0,
            },
            {
                "character_start_idx": 22,
                "character_end_idx": 58,
                "coref": "3",
                "text": "the mother of the hairdresser's wife",
                "distance": 0,
            },
        ],
    }
    expected = "The [dress]_fashion belonged to [the [mother]_3 of the [[hairdresser]_1's wife]_2]_3."
    assert llm.format_datum(datum) == expected
