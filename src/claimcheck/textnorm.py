"""Text normalisation shared by ingestion, retrieval, the pipeline and the claims tools.

- Unicode: NFC, ligatures, odd spaces, soft hyphens, zero-width and control characters.
- Urdu digits: Extended Arabic-Indic and Arabic-Indic digits and separators to ASCII.
- Input guard: trim, NFC, Urdu digits to ASCII, 10-500 characters.
- Numbers: thousands separators (Western and South Asian grouping), accounting negatives,
  Unicode minus, precision-aware matching (verifier rule 3).
- Units: percent / percentage points / basis points, currencies, scale words
  (verifier rule 4: 1 trillion = 1,000 billion; 1 billion = 100 crore = 1 arab;
  1 kharab = 100 billion; 1 crore = 10 million; 1 lakh = 100,000).
- Fiscal periods: FY2026 = 2025-26 = FY26 (July 2025 - June 2026); month ranges
  such as July-March mark a partial fiscal year.
- BM25 tokenizer: lowercase, keep decimals, strip thousands separators, emit a
  canonical fiscal-year token (`fy2026`) alongside the raw token.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

# --------------------------------------------------------------------------- Unicode

LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}
_ZERO_WIDTH = "​‌‍⁠﻿"
_SOFT_HYPHEN = "­"
_QUOTES = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "′": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "″": '"',
}
_DASHES = "‐‑‒–—―−﹘﹣－"
_UNICODE_MINUS = "−"

_CLEAN_TABLE = str.maketrans(
    {
        **LIGATURES,
        **{c: "" for c in _ZERO_WIDTH + _SOFT_HYPHEN},
        "\t": " ",
        "\r": "\n",
    }
)
_ASCII_PUNCT_TABLE = str.maketrans({**_QUOTES, **{c: "-" for c in _DASHES}})


def clean_unicode(text: str) -> str:
    """NFC; expand ligatures; drop soft hyphens, zero-width and control characters;
    turn every Unicode space separator into a plain space. Newlines are kept."""
    text = text.replace("\r\n", "\n")
    text = unicodedata.normalize("NFC", text).translate(_CLEAN_TABLE)
    out = []
    for ch in text:
        if ch == "\n":
            out.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat == "Zs":
            out.append(" ")
        elif cat in ("Cc", "Cf"):
            continue
        else:
            out.append(ch)
    return "".join(out)


def ascii_punct(text: str) -> str:
    """Curly quotes and primes to ASCII quotes; every dash and the minus sign to '-'."""
    return text.translate(_ASCII_PUNCT_TABLE)


def normalize_ws(text: str) -> str:
    """Collapse runs of spaces, strip each line, keep at most one blank line in a row."""
    lines = [re.sub(r" {2,}", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# --------------------------------------------------------------------------- Urdu digits

URDU_DIGITS = "۰۱۲۳۴۵۶۷۸۹"  # U+06F0..U+06F9, used in Urdu
ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"  # U+0660..U+0669
_DIGIT_TABLE = str.maketrans(
    {
        **{d: str(i) for i, d in enumerate(URDU_DIGITS)},
        **{d: str(i) for i, d in enumerate(ARABIC_INDIC_DIGITS)},
        "٫": ".",  # Arabic decimal separator
        "٬": ",",  # Arabic thousands separator
        "٪": "%",  # Arabic percent sign
    }
)


def ascii_digits(text: str) -> str:
    """Map Urdu / Arabic-Indic digits, decimal and thousands separators and % to ASCII."""
    return text.translate(_DIGIT_TABLE)


# --------------------------------------------------------------------------- input guard


class InputGuardError(ValueError):
    """Raised by `guard_input`; `code` is `empty`, `too_short` or `too_long`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def guard_input(text: str, min_chars: int = 10, max_chars: int = 500) -> str:
    """Trim, Unicode NFC, Urdu digits to ASCII, length 10-500."""
    text = ascii_digits(unicodedata.normalize("NFC", text)).strip()
    if not text:
        raise InputGuardError("empty", "Enter a claim to check.")
    if len(text) < min_chars:
        raise InputGuardError("too_short", f"A claim needs at least {min_chars} characters.")
    if len(text) > max_chars:
        raise InputGuardError("too_long", f"A claim can have at most {max_chars} characters.")
    return text


# --------------------------------------------------------------------------- numbers

# Western grouping (1,234,567.8) or South Asian grouping (12,34,567.8).
_GROUPED = r"(?:\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})+,\d{3})"
_GROUPED_RE = re.compile(rf"(?<![\d,.]){_GROUPED}(?:\.\d+)?(?![\d,]|\.\d)")
_NUMBER_BODY = rf"(?:{_GROUPED}|\d+)(?:\.\d+)?|\.\d+"
NUMBER_RE = re.compile(
    rf"(?P<paren>\()?(?P<sign>[-+{_UNICODE_MINUS}–])?(?P<num>{_NUMBER_BODY})(?(paren)\))"
)
_NUMBER_TOKEN_RE = re.compile(rf"^\(?[-+{_UNICODE_MINUS}–]?(?:{_NUMBER_BODY})\)?%?$")


def strip_thousands(text: str) -> str:
    """Remove thousands separators from grouped numbers: '1,234.5' -> '1234.5'.

    Only well-formed groups are touched, so '2024,25' or '1,2' stay as they are."""
    return _GROUPED_RE.sub(lambda m: m.group(0).replace(",", ""), text)


def is_number_token(token: str) -> bool:
    """True for a single numeric cell or token: '1,234.5', '(12.3)', '-0.5', '3.7%'."""
    return bool(_NUMBER_TOKEN_RE.match(token.replace(_UNICODE_MINUS, "-")))


def parse_number(token: str) -> float | None:
    """Parse one numeric token. '(1,234.5)' -> -1234.5 (accounting negative);
    '−3.7' (Unicode minus) -> -3.7; '3.7%' -> 3.7. Returns None if it is not a number."""
    s = ascii_digits(token.strip()).replace(_UNICODE_MINUS, "-").replace("–", "-")
    s = s.rstrip("%").strip()
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative, s = True, s[1:-1].strip()
    if s[:1] in "+-":
        negative ^= s[0] == "-"
        s = s[1:]
    if not re.fullmatch(_NUMBER_BODY, s):
        return None
    value = float(s.replace(",", ""))
    return -value if negative else value


def decimals(raw: str) -> int:
    """Number of digits after the decimal point in a numeric string ('3.70' -> 2)."""
    m = re.search(r"\.(\d+)", raw)
    return len(m.group(1)) if m else 0


def round_half_up(value: float, ndigits: int) -> Decimal:
    try:
        return Decimal(repr(value)).quantize(Decimal(1).scaleb(-ndigits), rounding=ROUND_HALF_UP)
    except InvalidOperation:  # pragma: no cover - inf / nan
        return Decimal(value)


def matches_at_precision(claim_raw: str, source_value: float) -> bool:
    """A claim's number matches if the source value rounds to it at the
    claim's stated precision. 'about 3' matches 2.68; '2.5' does not match 2.68."""
    claim_value = parse_number(claim_raw)
    if claim_value is None:
        return False
    nd = decimals(claim_raw)
    return round_half_up(source_value, nd) == round_half_up(claim_value, nd)


# --------------------------------------------------------------------------- units

SCALE_WORDS: dict[str, float] = {
    "thousand": 1e3,
    "lakh": 1e5,
    "lac": 1e5,
    "million": 1e6,
    "mn": 1e6,
    "mln": 1e6,
    "crore": 1e7,
    "billion": 1e9,
    "bn": 1e9,
    "arab": 1e9,
    "kharab": 1e11,
    "trillion": 1e12,
    "tn": 1e12,
    "trn": 1e12,
}
_SCALE_PLURALS = {
    "lakhs",
    "lacs",
    "millions",
    "crores",
    "billions",
    "arabs",
    "kharabs",
    "trillions",
    "thousands",
}

PERCENT_WORDS = {
    "%": "percent",
    "percent": "percent",
    "per cent": "percent",
    "pc": "percent",
    "pct": "percent",
    "percentage points": "pp",
    "percentage point": "pp",
    "pp": "pp",
    "ppt": "pp",
    "ppts": "pp",
    "basis points": "bps",
    "basis point": "bps",
    "bps": "bps",
    "bp": "bps",
}
CURRENCY_WORDS = {
    "rs": "PKR",
    "rs.": "PKR",
    "pkr": "PKR",
    "rupees": "PKR",
    "rupee": "PKR",
    "us$": "USD",
    "$": "USD",
    "usd": "USD",
    "dollars": "USD",
    "dollar": "USD",
    "us dollars": "USD",
}


def scale_factor(word: str) -> float | None:
    """Multiplier for a scale word ('crore' -> 1e7, 'billions' -> 1e9), else None."""
    w = word.lower().strip(". ")
    if w in _SCALE_PLURALS:
        w = w[:-1]
    return SCALE_WORDS.get(w)


def convert_scale(value: float, from_scale: str, to_scale: str) -> float:
    """Convert between scale words: convert_scale(1.2, 'trillion', 'billion') == 1200."""
    f, t = scale_factor(from_scale), scale_factor(to_scale)
    if f is None or t is None:
        raise ValueError(f"unknown scale word: {from_scale if f is None else to_scale!r}")
    return value * f / t


@dataclass(frozen=True)
class Quantity:
    raw: str
    number: float  # as written, sign applied, scale not applied
    decimals: int
    scale: str | None  # canonical scale word, e.g. 'billion'
    unit: str | None  # 'percent', 'pp', 'bps'
    currency: str | None  # 'PKR', 'USD'
    start: int
    end: int

    @property
    def value(self) -> float:
        """The number with its scale applied (3 crore -> 30,000,000)."""
        f = scale_factor(self.scale) if self.scale else None
        return self.number * f if f else self.number


_SCALE_ALT = "|".join(sorted({*SCALE_WORDS, *_SCALE_PLURALS}, key=len, reverse=True))
_UNIT_ALT = "|".join(re.escape(u) for u in sorted(PERCENT_WORDS, key=len, reverse=True) if u != "%")
_CUR_ALT = r"US\s?\$|Rs\.?|PKR|USD|\$"
QUANTITY_RE = re.compile(
    rf"(?P<cur>{_CUR_ALT})?\s?"
    rf"(?P<num>\(?[-+{_UNICODE_MINUS}]?(?:{_NUMBER_BODY})\)?)"
    rf"(?:\s?(?P<pct>%)|\s+(?P<scale>{_SCALE_ALT})\b)?"
    rf"(?:\s+(?P<unit>{_UNIT_ALT})\b)?"
    rf"(?:\s+(?P<curw>rupees?|dollars?)\b)?",
    re.IGNORECASE,
)


def find_quantities(text: str) -> list[Quantity]:
    """Numbers with their scale, unit and currency, in text order.

    Fiscal years ('2025-26', 'FY2026') are not quantities; use `find_fiscal_years`."""
    text = ascii_digits(text)
    fy_spans = [(m.start, m.end) for m in find_fiscal_years(text)]
    out: list[Quantity] = []
    for m in QUANTITY_RE.finditer(text):
        start, end = m.span("num")
        if any(s <= start < e for s, e in fy_spans):
            continue
        glued = start > 0 and (text[start - 1].isalnum() or text[start - 1] in "-/")
        if glued and not m.group("cur"):
            continue  # part of a word or code such as 'COVID-19' or 'A1'
        raw = m.group("num")
        if raw.startswith("(") != raw.endswith(")"):
            raw = raw.strip("()")
        number = parse_number(raw)
        if number is None:
            continue
        unit = None
        if m.group("pct"):
            unit = "percent"
        elif m.group("unit"):
            unit = PERCENT_WORDS[re.sub(r"\s+", " ", m.group("unit").lower())]
        scale = m.group("scale")
        if scale:
            scale = scale.lower()
            scale = scale[:-1] if scale in _SCALE_PLURALS else scale
        cur = m.group("cur") or m.group("curw")
        currency = CURRENCY_WORDS.get(re.sub(r"\s+", "", cur.lower())) if cur else None
        q_start = m.start("cur") if m.group("cur") else start
        out.append(
            Quantity(
                raw=raw,
                number=number,
                decimals=decimals(raw),
                scale=scale,
                unit=unit,
                currency=currency,
                start=q_start,
                end=m.end(),
            )
        )
    return out


# --------------------------------------------------------------------------- fiscal periods


@dataclass(frozen=True)
class FiscalYear:
    raw: str
    end_year: int  # FY2026 = 2025-26 -> 2026
    start: int
    end: int

    @property
    def canonical(self) -> str:
        return canonical_fy(self.end_year)

    @property
    def token(self) -> str:
        return fy_token(self.end_year)


def canonical_fy(end_year: int) -> str:
    """2026 -> '2025-26'."""
    return f"{end_year - 1}-{end_year % 100:02d}"


def fy_token(end_year: int) -> str:
    """2026 -> 'fy2026' (the canonical BM25 token)."""
    return f"fy{end_year}"


def _century(two_digits: int) -> int:
    return 2000 + two_digits if two_digits < 70 else 1900 + two_digits


# 'FY2026', 'FY26', 'FY 2026', 'FY-26', "FY'26", 'FY2025-26', 'FY 2025-2026'
_FY_PREFIXED = re.compile(
    r"(?<![A-Za-z0-9])F\.?Y\.?\s?[-'’]?\s?(?P<a>\d{4}|\d{2})"
    r"(?:\s?[-–/]\s?(?P<b>\d{4}|\d{2}))?(?!\d)",
    re.IGNORECASE,
)
# '2025-26', '2025-2026', '2025/26' (consecutive years only)
_FY_RANGE = re.compile(r"(?<![\d.,])(?P<a>(?:19|20)\d{2})\s?[-–/]\s?(?P<b>\d{4}|\d{2})(?![\d])")
# 'fiscal year 2026', 'financial year 2026'
_FY_WORDS = re.compile(
    r"\b(?:fiscal|financial)\s+year\s+(?P<a>(?:19|20)\d{2})(?![\d\-–/])", re.IGNORECASE
)


def _range_end(a: int, b_raw: str) -> int | None:
    b = int(b_raw)
    if len(b_raw) == 2:
        return a + 1 if (a + 1) % 100 == b else None
    return b if b == a + 1 else None


def find_fiscal_years(text: str) -> list[FiscalYear]:
    """Fiscal-year mentions in text order. A plain range must be consecutive years
    ('2025-26'); a lone 4-digit year is a fiscal year only after 'FY' or 'fiscal year'."""
    text = ascii_digits(text)
    found: list[FiscalYear] = []
    taken: list[tuple[int, int]] = []

    def add(m: re.Match[str], end_year: int | None) -> None:
        if end_year is None or any(s < m.end() and m.start() < e for s, e in taken):
            return
        taken.append(m.span())
        found.append(FiscalYear(m.group(0), end_year, m.start(), m.end()))

    for m in _FY_PREFIXED.finditer(text):
        a_raw, b_raw = m.group("a"), m.group("b")
        a = int(a_raw) if len(a_raw) == 4 else _century(int(a_raw))
        if b_raw is None:
            add(m, a)
        else:
            add(m, _range_end(a, b_raw if len(b_raw) == 2 else b_raw))
    for m in _FY_RANGE.finditer(text):
        add(m, _range_end(int(m.group("a")), m.group("b")))
    for m in _FY_WORDS.finditer(text):
        add(m, int(m.group("a")))
    return sorted(found, key=lambda f: f.start)


def parse_fiscal_year(text: str) -> int | None:
    """End year of the first fiscal year in `text` ('FY26' -> 2026), else None."""
    found = find_fiscal_years(text)
    return found[0].end_year if found else None


def canonicalize_fiscal_years(text: str) -> str:
    """Rewrite every fiscal-year mention as 'FY2025-26'."""
    out, pos = [], 0
    for fy in find_fiscal_years(text):
        out.append(text[pos : fy.start])
        out.append(f"FY{fy.canonical}")
        pos = fy.end
    out.append(text[pos:])
    return "".join(out)


def fiscal_year_of(year: int, month: int) -> int:
    """Fiscal year (end year) of a calendar month: July 2025 .. June 2026 -> 2026."""
    return year + 1 if month >= 7 else year


MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_MONTH_RANGE = re.compile(
    rf"\b(?P<a>{_MONTH})\s?(?:-|–|to|through)\s?(?P<b>{_MONTH})(?![a-z])", re.IGNORECASE
)


@dataclass(frozen=True)
class MonthRange:
    raw: str
    start_month: int
    end_month: int
    start: int
    end: int

    @property
    def n_months(self) -> int:
        return (self.end_month - self.start_month) % 12 + 1

    @property
    def partial_year(self) -> bool:
        """Anything other than July-June is a partial fiscal year."""
        return not (self.start_month == 7 and self.end_month == 6)


def find_month_ranges(text: str) -> list[MonthRange]:
    """Month ranges such as 'July-March', 'Jul-Mar', 'July to April'."""
    out = []
    for m in _MONTH_RANGE.finditer(text):
        a = MONTHS[m.group("a")[:3].lower()]
        b = MONTHS[m.group("b")[:3].lower()]
        out.append(MonthRange(m.group(0), a, b, m.start(), m.end()))
    return out


# --------------------------------------------------------------------------- matching


def normalize_for_match(text: str) -> str:
    """Key for comparing a quote with a passage (code guards): Unicode clean-up,
    ASCII digits, quotes and dashes, casefold, thousands separators stripped, '%' and
    'per cent' as 'percent', whitespace collapsed, line-end hyphenation undone."""
    text = ascii_punct(ascii_digits(clean_unicode(text))).casefold()
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = strip_thousands(text)
    text = re.sub(r"\s*%", " percent", text)
    text = re.sub(r"\bper\s+cent\b", "percent", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- tokenizer

_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[a-z]+(?:'[a-z]+)?|%|\$")


def tokenize(text: str) -> list[str]:
    """BM25 tokenizer.

    Lowercase; keep decimals ('3.7'); strip thousands separators ('1,234.5' -> '1234.5');
    for every fiscal-year mention emit the raw token and then the canonical token
    ('2025-26' -> '2025-26', 'fy2026'; 'FY26' -> 'fy26', 'fy2026'; 'FY2026' -> 'fy2026').
    '%' becomes 'percent' and '$' becomes 'usd'. Signs are dropped, so '-1.3' and
    'declined 1.3' share '1.3'."""
    text = ascii_punct(ascii_digits(clean_unicode(text)))
    text = strip_thousands(text)
    text = re.sub(r"\bper\s+cent\b", "percent", text, flags=re.IGNORECASE)
    fys = find_fiscal_years(text)
    tokens: list[str] = []
    pos = 0
    for fy in fys:
        tokens.extend(_plain_tokens(text[pos : fy.start]))
        raw = fy.raw.lower().replace("/", "-")
        if raw.startswith(("fiscal", "financial")):
            tokens.extend(_plain_tokens(raw))  # 'fiscal', 'year', '2026'
        else:
            raw = re.sub(r"^fy-", "fy", re.sub(r"[\s.']", "", raw))  # "FY '26" -> 'fy26'
            tokens.append(raw)
        if fy.token != raw:
            tokens.append(fy.token)
        pos = fy.end
    tokens.extend(_plain_tokens(text[pos:]))
    return tokens


def _plain_tokens(text: str) -> list[str]:
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok == "%":
            out.append("percent")
        elif tok == "$":
            out.append("usd")
        else:
            out.append(tok)
    return out
