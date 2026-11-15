"""textnorm: Unicode, Urdu digits, input guard, numbers, units, fiscal periods, tokenizer."""

import pytest

from claimcheck.textnorm import (
    InputGuardError,
    ascii_digits,
    ascii_punct,
    canonical_fy,
    canonicalize_fiscal_years,
    clean_unicode,
    convert_scale,
    decimals,
    find_fiscal_years,
    find_month_ranges,
    find_quantities,
    fiscal_year_of,
    fy_token,
    guard_input,
    is_number_token,
    matches_at_precision,
    normalize_for_match,
    normalize_ws,
    parse_fiscal_year,
    parse_number,
    scale_factor,
    strip_thousands,
    tokenize,
)

# --------------------------------------------------------------------------- Unicode


def test_clean_unicode_ligatures_spaces_and_invisibles() -> None:
    raw = "eﬃcient ﬁscal policy and soft­hyphen​ x\r\ny"
    assert clean_unicode(raw) == "efficient fiscal policy and softhyphen x\ny"


def test_clean_unicode_is_nfc() -> None:
    assert clean_unicode("café") == "café"


def test_ascii_punct_quotes_and_dashes() -> None:
    assert ascii_punct("“GDP” ‘up’ 2024–25 −3.7") == "\"GDP\" 'up' 2024-25 -3.7"


def test_normalize_ws() -> None:
    assert normalize_ws("  a   b \n\n\n\n c  ") == "a b\n\nc"


# --------------------------------------------------------------------------- Urdu digits


def test_urdu_and_arabic_indic_digits() -> None:
    assert ascii_digits("۳.۷ فیصد") == "3.7 فیصد"
    assert ascii_digits("٢٠٢٥") == "2025"
    assert ascii_digits("۱٬۲۳۴٫۵٪") == "1,234.5%"


# --------------------------------------------------------------------------- input guard


def test_guard_input_normalises() -> None:
    assert guard_input("  جی ڈی پی ۳.۷ فیصد بڑھی  ") == "جی ڈی پی 3.7 فیصد بڑھی"
    assert guard_input("café prices rose") == "café prices rose"


@pytest.mark.parametrize(
    ("text", "code"),
    [("   ", "empty"), ("GDP 3%", "too_short"), ("x" * 501, "too_long")],
)
def test_guard_input_rejects(text: str, code: str) -> None:
    with pytest.raises(InputGuardError) as err:
        guard_input(text)
    assert err.value.code == code


def test_guard_input_limits_are_inclusive() -> None:
    assert guard_input("a" * 10) == "a" * 10
    assert guard_input("a" * 500) == "a" * 500


# --------------------------------------------------------------------------- numbers


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("1,234.5", 1234.5),
        ("(1,234.5)", -1234.5),
        ("−3.7", -3.7),
        ("-0.5", -0.5),
        ("3.7%", 3.7),
        ("12,34,567", 1234567.0),
        ("۳.۷", 3.7),
        ("abc", None),
        ("1,23", None),
    ],
)
def test_parse_number(raw: str, value: float | None) -> None:
    assert parse_number(raw) == value


def test_strip_thousands_only_well_formed_groups() -> None:
    assert strip_thousands("Rs 1,234.5 bn and 12,34,567") == "Rs 1234.5 bn and 1234567"
    assert strip_thousands("in 2024,25 and 1,2") == "in 2024,25 and 1,2"


def test_is_number_token() -> None:
    assert all(is_number_token(t) for t in ["1,234", "(12.3)", "-0.5", "3.7%", "100.0"])
    assert not any(is_number_token(t) for t in ["FY26", "2025-26", "abc", "3.7.1"])


def test_decimals() -> None:
    assert (decimals("3"), decimals("3.70"), decimals("(1,234.5)")) == (0, 2, 1)


def test_matches_at_precision_rule_3() -> None:
    assert matches_at_precision("3", 2.68)  # "about 3 %" matches 2.68 %
    assert not matches_at_precision("2.5", 2.68)  # "2.5 %" does not
    assert matches_at_precision("2.7", 2.65)  # half-up rounding
    assert matches_at_precision("6.2", 6.2)
    assert not matches_at_precision("x", 6.2)


# --------------------------------------------------------------------------- units


def test_scale_rules_of_plan_9_1() -> None:
    assert convert_scale(1, "trillion", "billion") == 1000
    assert convert_scale(1, "billion", "crore") == 100
    assert convert_scale(1, "arab", "billion") == 1
    assert convert_scale(1, "kharab", "billion") == 100
    assert convert_scale(1, "crore", "million") == 10
    assert scale_factor("lakh") == 100_000
    assert scale_factor("Crores") == 1e7
    assert scale_factor("dozen") is None
    with pytest.raises(ValueError):
        convert_scale(1, "dozen", "billion")


def test_find_quantities() -> None:
    qs = find_quantities(
        "Rs 1.2 trillion, 3.7 percent, 25 basis points, US$ 82 per barrel, 3 crore rupees, "
        "a fall of (1,234.5), COVID-19 in 2024-25"
    )
    got = [(q.number, q.scale, q.unit, q.currency) for q in qs]
    assert got == [
        (1.2, "trillion", None, "PKR"),
        (3.7, None, "percent", None),
        (25.0, None, "bps", None),
        (82.0, None, None, "USD"),
        (3.0, "crore", None, "PKR"),
        (-1234.5, None, None, None),
    ]
    assert qs[0].value == pytest.approx(1.2e12)
    assert qs[4].value == pytest.approx(3e7)


def test_find_quantities_percent_sign_and_urdu_digits() -> None:
    (q,) = find_quantities("۳.۷%")
    assert (q.number, q.unit, q.decimals) == (3.7, "percent", 1)


# --------------------------------------------------------------------------- fiscal periods


@pytest.mark.parametrize(
    "text",
    ["FY2026", "FY26", "FY 2026", "FY-26", "FY'26", "FY2025-26", "FY 2025-2026", "2025-26",
     "2025-2026", "2025/26", "2025–26", "fiscal year 2026", "F.Y. 2026"],
)  # fmt: skip
def test_fiscal_year_forms_are_equal(text: str) -> None:
    assert parse_fiscal_year(text) == 2026


def test_fiscal_year_rejects_non_consecutive_and_plain_years() -> None:
    assert parse_fiscal_year("2020-2025") is None
    assert parse_fiscal_year("in 2025 prices rose") is None
    assert parse_fiscal_year("pages 25-26") is None


def test_find_fiscal_years_positions() -> None:
    text = "Growth in FY26 beat 2024-25 and FY2023."
    found = find_fiscal_years(text)
    assert [(f.raw, f.end_year) for f in found] == [
        ("FY26", 2026),
        ("2024-25", 2025),
        ("FY2023", 2023),
    ]
    assert all(text[f.start : f.end] == f.raw for f in found)


def test_canonical_forms() -> None:
    assert canonical_fy(2026) == "2025-26"
    assert canonical_fy(2000) == "1999-00"
    assert fy_token(2026) == "fy2026"
    assert canonicalize_fiscal_years("In FY26 and 2024-25") == "In FY2025-26 and FY2024-25"


def test_fiscal_year_of_month() -> None:
    assert fiscal_year_of(2025, 7) == 2026
    assert fiscal_year_of(2026, 6) == 2026
    assert fiscal_year_of(2026, 7) == 2027


def test_month_ranges_and_partial_year() -> None:
    ranges = find_month_ranges("July-March FY2026, Jul–Apr, July to June")
    assert [(r.start_month, r.end_month, r.n_months, r.partial_year) for r in ranges] == [
        (7, 3, 9, True),
        (7, 4, 10, True),
        (7, 6, 12, False),
    ]


# --------------------------------------------------------------------------- matching


def test_normalize_for_match() -> None:
    a = normalize_for_match("GDP grew by 3.7 per cent to Rs 1,234.5 bn — “record”")
    assert a == 'gdp grew by 3.7 percent to rs 1234.5 bn - "record"'
    assert normalize_for_match("pass-\nthrough  of 3.7%") == "passthrough of 3.7 percent"


# --------------------------------------------------------------------------- tokenizer


def test_tokenizer_plan_rules() -> None:
    assert tokenize("GDP grew 3.7% to Rs 1,234.5 billion") == [
        "gdp", "grew", "3.7", "percent", "to", "rs", "1234.5", "billion",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("FY2026", ["fy2026"]),
        ("FY26", ["fy26", "fy2026"]),
        ("FY 2025-26", ["fy2025-26", "fy2026"]),
        ("2025-26", ["2025-26", "fy2026"]),
        ("2025–26", ["2025-26", "fy2026"]),
        ("fiscal year 2026", ["fiscal", "year", "2026", "fy2026"]),
    ],
)
def test_tokenizer_emits_canonical_fiscal_year(text: str, tokens: list[str]) -> None:
    assert tokenize(text) == tokens


def test_tokenizer_drops_signs_and_keeps_decimals() -> None:
    assert tokenize("(−1.3) and -0.50") == ["1.3", "and", "0.50"]


def test_tokenizer_handles_urdu_digits_and_per_cent() -> None:
    assert tokenize("۳.۷ per cent") == ["3.7", "percent"]
