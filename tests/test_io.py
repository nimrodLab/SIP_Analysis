from __future__ import annotations

from sip_analysis.io import guess_delimiter, parse_timestamp_string, find_data_layout


def test_guess_delimiter_prefers_comma():
    lines = ["a,b,c", "1,2,3", "4,5,6"]
    assert guess_delimiter(lines) == ","


def test_parse_timestamp_string_variants():
    assert parse_timestamp_string("2025-09-06 13:57:27") is not None
    assert parse_timestamp_string("09/06/2025 13:57:27") is not None
    assert parse_timestamp_string(" ") is None


def test_find_data_layout_detects_columns():
    rows = [
        ["meta", "x"],
        ["Frequency[Hz]", "Impedance[Ohm]", "Phase_Shift[rad]"],
        ["1", "100", "0.1"],
    ]
    start, fi, ii, pi, pu = find_data_layout(rows)
    assert start == 2
    assert (fi, ii, pi) == (0, 1, 2)
    assert pu == "rad"
