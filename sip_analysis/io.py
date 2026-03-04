from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def to_float(value: str) -> float:
    if value is None:
        raise ValueError("None is not a number")
    text = str(value).strip().replace("\ufeff", "")
    if not text:
        raise ValueError("empty string")

    if text.count(",") == 1 and text.count(".") == 0:
        text = text.replace(",", ".")
    elif text.count(",") > 1 and text.count(".") == 0:
        text = text.replace(",", "")
    else:
        text = text.replace(",", "")

    return float(text)


def guess_delimiter(lines: list[str]) -> str:
    candidates = [",", ";", "\t"]
    scores = {d: 0 for d in candidates}
    for line in lines[:50]:
        for d in candidates:
            scores[d] += line.count(d)
    return max(scores, key=scores.get)


def normalize_header(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def parse_timestamp_string(text: str) -> str | None:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return None

    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%y %H:%M:%S",
        "%m/%d/%y %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(text, fmt).isoformat(sep=" ")
        except ValueError:
            pass

    dt = pd.to_datetime(text, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        dt = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.notna(dt):
        return pd.Timestamp(dt).isoformat(sep=" ")
    return None


def detect_timestamp(lines: list[str]) -> str | None:
    joined = "\n".join(lines[:120])

    patterns = [
        r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)",
        r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}[ T]\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)",
    ]
    for pat in patterns:
        match = re.search(pat, joined)
        if match:
            ts = parse_timestamp_string(match.group(1))
            if ts:
                return ts

    date_match = re.search(r"(?:date|measurement\s*date)\s*[:=]\s*([^\n\r]+)", joined, re.I)
    time_match = re.search(r"(?:time|measurement\s*time)\s*[:=]\s*([^\n\r]+)", joined, re.I)
    if date_match and time_match:
        ts = parse_timestamp_string(f"{date_match.group(1).strip()} {time_match.group(1).strip()}")
        if ts:
            return ts

    return None


def find_data_layout(rows: list[list[str]]) -> tuple[int, int, int, int, str]:
    header_idx = -1
    header_cells: list[str] = []

    for i, row in enumerate(rows[:120]):
        if len(row) < 3:
            continue
        norm = [normalize_header(cell) for cell in row]

        has_freq = any("freq" in c for c in norm)
        has_phase = any("phase" in c or c == "phi" for c in norm)
        has_imp = any(("imp" in c) or ("z" in c and "hz" not in c) for c in norm)

        if has_freq and has_phase and has_imp:
            header_idx = i
            header_cells = row
            break

    if header_idx == -1:
        for i, row in enumerate(rows):
            if len(row) < 3:
                continue
            try:
                to_float(row[0]); to_float(row[1]); to_float(row[2])
                return i, 0, 1, 2, "unknown"
            except ValueError:
                continue
        raise ValueError("Could not locate SIP data columns (frequency/impedance/phase).")

    norm = [normalize_header(c) for c in header_cells]

    def pick_idx(predicate):
        for idx, cell in enumerate(norm):
            if predicate(cell):
                return idx
        return -1

    freq_idx = pick_idx(lambda c: "freq" in c)
    imp_idx = pick_idx(
        lambda c: (("imp" in c) or c in {"z", "absz", "modz", "zmag", "impedanceohm"} or "zohm" in c or "zmod" in c)
        and "real" not in c
        and "imag" not in c
    )
    phase_idx = pick_idx(lambda c: "phase" in c or c == "phi")

    if min(freq_idx, imp_idx, phase_idx) < 0:
        raise ValueError("Found data header, but missing frequency/impedance/phase columns.")

    phase_header = norm[phase_idx]
    if "mrad" in phase_header:
        phase_unit = "mrad"
    elif "deg" in phase_header:
        phase_unit = "deg"
    elif "rad" in phase_header:
        phase_unit = "rad"
    else:
        phase_unit = "unknown"

    return header_idx + 1, freq_idx, imp_idx, phase_idx, phase_unit


def parse_sip_csv(path: str) -> pd.DataFrame:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        raise ValueError("File is empty.")

    timestamp = detect_timestamp(lines)
    delimiter = guess_delimiter(lines)
    rows = list(csv.reader(lines, delimiter=delimiter))

    start_row, freq_idx, imp_idx, phase_idx, phase_unit = find_data_layout(rows)

    data = []
    for row in rows[start_row:]:
        max_idx = max(freq_idx, imp_idx, phase_idx)
        if len(row) <= max_idx:
            continue
        try:
            freq = to_float(row[freq_idx])
            imp = to_float(row[imp_idx])
            phase_raw = to_float(row[phase_idx])
        except ValueError:
            continue

        if phase_unit == "deg":
            phase_rad = np.deg2rad(phase_raw)
        elif phase_unit == "mrad":
            phase_rad = phase_raw / 1000.0
        elif phase_unit == "rad":
            phase_rad = phase_raw
        else:
            phase_rad = np.deg2rad(phase_raw) if abs(phase_raw) > 3.2 else phase_raw

        data.append(
            {
                "file": os.path.basename(path),
                "filepath": str(path),
                "timestamp": timestamp,
                "frequency_hz": freq,
                "impedance_ohm": imp,
                "phase_rad": phase_rad,
            }
        )

    if not data:
        raise ValueError("No numeric SIP rows found in file.")

    return pd.DataFrame(data).sort_values("frequency_hz").reset_index(drop=True)
