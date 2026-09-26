import re
import unicodedata


def normalize_text(text):
    """
    Clean a business name or address before comparing it.

    The dataset contains differences such as:
    - upper/lower case
    - punctuation
    - extra spaces
    - accented characters

    Normalization makes these records easier to compare.
    """

    # Missing business names or addresses are treated as empty text.
    if text is None:
        return ""

    text = str(text).lower().strip()

    # Convert accented characters into a simpler form.
    # Example: "é" becomes "e" in most cases.
    text = unicodedata.normalize("NFKD", text)

    # Remove Unicode accent marks.
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    # Keep English letters, numbers and Devanagari characters.
    # Devanagari is important because the dataset contains Indian
    # business names written in Hindi.
    text = re.sub(
        r"[^a-z0-9\u0900-\u097f]+",
        " ",
        text
    )

    # Convert multiple spaces into one space.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compact_text(text):
    """
    Return normalized text without spaces.

    This helps when two records have the same words but different
    spacing or punctuation.
    """

    return normalize_text(text).replace(" ", "")


def name_prefix(name, length=5):
    """
    Create a short blocking key from a business name.

    This is NOT used to decide whether two businesses match.
    It is only used to find a smaller group of possible candidates.
    """

    return compact_text(name)[:length]


def address_prefix(address, length=8):
    """
    Create a short blocking key from an address.

    Address information is useful for finding businesses that have
    similar locations, while the final matching decision will use
    more detailed similarity features.
    """

    return compact_text(address)[:length]