"""
Emission Factors Data Processing Pipeline
==========================================
A reusable, scalable pipeline for extracting structured information
from large emission factors datasets (.json.gz format).

Usage:
    python emission_pipeline.py
    # or programmatically:
    from emission_pipeline import run_pipeline
    result = run_pipeline("file.json.gz", "vehicle")
    result = run_pipeline("file.json.gz", ["vehicle", "fuel"])  # multiple keywords
"""

import gzip
import json
import re
import logging
import pandas as pd
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Data Loading
# ---------------------------------------------------------------------------

def load_data(filepath: str, chunksize: int = 50_000) -> pd.DataFrame:
    """
    Load a JSON or JSON.GZ dataset using ijson for true C-speed streaming.

    Supports two layouts automatically:
      - Django fixture  [ {"model":..., "fields": {...}}, ... ]
      - Flat array      [ {"activity_id": ..., ...}, ... ]

    Extracts ALL fields from each record (not just activity_id), so that
    downstream steps can work with the full record data.

    NOTE: activity_id values are NOT unique — the same activity_id can appear
    multiple times across records with different field values (e.g. different
    year, region, factor). Each record is retained as a distinct row.

    Args:
        filepath:  Path to the dataset file (.json or .json.gz).
        chunksize: Records per batch before flushing to DataFrame chunks.

    Returns:
        DataFrame with an `activity_id` column plus all other extracted fields.

    Raises:
        FileNotFoundError: File does not exist.
        ValueError:        File cannot be parsed or lacks activity_id.
    """
    try:
        import ijson
    except ImportError:
        raise ImportError(
            "ijson is required for fast streaming. "
            "Install it with:  pip install ijson"
        )

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    logger.info(f"Loading dataset from: {filepath} (chunksize={chunksize:,})")

    is_gz   = filepath.endswith(".gz")
    open_fn = gzip.open if is_gz else open

    chunks: list[pd.DataFrame] = []
    buffer: list[dict]         = []
    total  = 0

    def _flush():
        nonlocal total
        chunks.append(pd.DataFrame(buffer))
        total += len(buffer)
        logger.info(f"  Parsed {total:,} records ...")
        buffer.clear()

    # ── Format detection ──────────────────────────────────────────────────────
    # Django fixture: each top-level item has a "fields" object containing all
    # the record data (including activity_id).
    # Flat array: each item is directly the record dict.
    #
    # We use ijson.items() to pull each top-level array element as a complete
    # Python dict, then normalise to a flat record. This is slightly less
    # memory-efficient than event-streaming, but still O(chunksize) peak RAM
    # and avoids complex state machines for nested objects.

    IS_FIXTURE_KEY = "fields"   # Django fixture marker

    # ── Peek at the raw bytes to give early, specific error messages ──────────
    try:
        with open_fn(filepath, "rb") as f:
            peek = f.read(512).strip()
    except Exception as exc:
        raise ValueError(
            f"Cannot open '{Path(filepath).name}': {exc}\n"
            "Check the file is not corrupted and is a valid .json or .json.gz file."
        ) from exc

    if not peek:
        raise ValueError(
            f"'{Path(filepath).name}' appears to be empty (0 bytes after decompression)."
        )
    if peek[0:1] not in (b"[", b"{"):
        raise ValueError(
            f"'{Path(filepath).name}' does not look like JSON — "
            f"first character is '{peek[0:1].decode('utf-8', errors='replace')}', expected '[' or '{{'.\n"
            "Make sure you uploaded the correct file."
        )
    if peek[0:1] == b"{":
        raise ValueError(
            f"'{Path(filepath).name}' is a JSON object (starts with '{{'), "
            "but the pipeline expects a JSON array (starts with '[').\n"
            "The file should be a list of records: [ {{...}}, {{...}}, ... ]"
        )

    # ── Stream and parse ──────────────────────────────────────────────────────
    items_seen   = 0
    items_no_aid = 0

    try:
        with open_fn(filepath, "rb") as f:
            detected_format = None
            for raw_item in ijson.items(f, "item"):
                if not isinstance(raw_item, dict):
                    continue

                items_seen += 1

                # Detect format on first item
                if detected_format is None:
                    if IS_FIXTURE_KEY in raw_item and isinstance(raw_item[IS_FIXTURE_KEY], dict):
                        detected_format = "fixture"
                    else:
                        detected_format = "flat"
                    logger.info(f"  Detected format: '{detected_format}'")

                # Flatten to a single record dict
                if detected_format == "fixture":
                    record: dict = dict(raw_item.get(IS_FIXTURE_KEY, {}))
                    if "pk" in raw_item:
                        record.setdefault("_pk", raw_item["pk"])
                else:
                    record = dict(raw_item)

                # Must have an activity_id to be useful
                if "activity_id" not in record:
                    items_no_aid += 1
                    continue

                buffer.append(record)
                if len(buffer) >= chunksize:
                    _flush()

    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Failed to read '{Path(filepath).name}': {exc}\n"
            "The file may be corrupted or incompletely uploaded."
        ) from exc
    except Exception as exc:
        # Catch ijson parse errors (JSONError, IncompleteJSONError, etc.)
        raise ValueError(
            f"JSON parse error in '{Path(filepath).name}': {exc}\n"
            "The file may be malformed or truncated."
        ) from exc

    if buffer:
        chunks.append(pd.DataFrame(buffer))

    if not chunks:
        if items_seen == 0:
            raise ValueError(
                f"No records found in '{Path(filepath).name}'.\n"
                "The file parsed as valid JSON but contained no array items. "
                "Expected: [ {{...}}, {{...}}, ... ]"
            )
        raise ValueError(
            f"Parsed {items_seen:,} records from '{Path(filepath).name}' "
            f"but none contained an 'activity_id' field "
            f"({items_no_aid:,} records had no 'activity_id').\n"
            "Check that this is the correct emission factors dataset file. "
            "Expected either a Django fixture with 'fields.activity_id' "
            "or a flat array with 'activity_id' at the top level."
        )

    df = pd.concat(chunks, ignore_index=True)
    logger.info(
        f"Loaded {len(df):,} raw records "
        f"({df['activity_id'].nunique():,} distinct activity_ids before deduplication)."
    )
    return df


# ---------------------------------------------------------------------------
# Step 2: Keyword-Based Filtering
# ---------------------------------------------------------------------------

def filter_by_keyword(df: pd.DataFrame, keywords: str | list[str]) -> pd.DataFrame:
    """
    Filter rows where `activity_id` contains any of the given keywords.

    Filtering is:
      - Case-insensitive
      - Null-safe (NaN/None values in activity_id are excluded)
      - Supports a single keyword (str) or multiple keywords (list[str])

    Args:
        df:       Input DataFrame with an `activity_id` column.
        keywords: One or more keywords to match against activity_id.

    Returns:
        Filtered DataFrame. May be empty if no matches found.
    """
    if isinstance(keywords, str):
        keywords = [keywords]

    # Build a case-insensitive OR pattern across all keywords
    pattern = "|".join(re.escape(kw.strip()) for kw in keywords if kw.strip())
    if not pattern:
        raise ValueError("At least one non-empty keyword must be provided.")

    logger.info(f"Filtering by keyword(s): {keywords}")

    # na=False ensures NaN activity_id values evaluate to False (null-safe)
    mask = df["activity_id"].str.contains(pattern, case=False, na=False, regex=True)
    filtered = df[mask].copy()

    logger.info(f"Found {len(filtered):,} matching records out of {len(df):,}.")
    return filtered


# ---------------------------------------------------------------------------
# Step 2b: Deduplication
# ---------------------------------------------------------------------------

def deduplicate_activity_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce the DataFrame to one row per unique activity_id.

    When the same activity_id appears across multiple records (e.g. different
    year, region, or factor in a Django fixture), only the first occurrence is
    kept. All subsequent pipeline steps (parse, clean, aggregate) operate on
    the deduplicated set, so each structural pattern in the ID is processed
    exactly once.

    Args:
        df: DataFrame with an `activity_id` column (may contain duplicates).

    Returns:
        DataFrame with duplicate activity_ids dropped, index reset.
    """
    before = len(df)
    df_dedup = df.drop_duplicates(subset=["activity_id"]).reset_index(drop=True)
    after = len(df_dedup)
    dropped = before - after
    logger.info(
        f"Deduplicated activity_ids: {before:,} rows → {after:,} unique IDs "
        f"({dropped:,} duplicates removed)."
    )
    return df_dedup


# ---------------------------------------------------------------------------
# Step 3: Parsing Logic
# ---------------------------------------------------------------------------

def parse_activity_id(activity_id: str) -> dict[str, str]:
    """
    Parse a structured activity_id string into key-value pairs.

    Format:
        <category>-<key_subkey_value>-<key_subkey_value>-...

    Examples:
        "freight_vehicle-vehicle_type_hgv-fuel_source_na"
        → {
            "category": "freight_vehicle",
            "vehicle_type": "hgv",
            "fuel_source": "na"
          }

    Rules:
      - First segment (split on `-`) → `category`
      - Remaining segments split on `_`:
          * Key  = first two tokens joined by `_`
          * Value = remaining tokens joined by `_`
      - Malformed segments (fewer than 3 tokens) are skipped with a warning.

    Args:
        activity_id: Raw activity_id string.

    Returns:
        Dict of parsed attributes, always containing at least `category`.
        Returns {"category": activity_id} for completely unparseable inputs.
    """
    if not isinstance(activity_id, str) or not activity_id.strip():
        return {}

    segments = activity_id.strip().split("-")
    result: dict[str, str] = {}

    # First segment is the category (may itself contain underscores)
    result["category"] = segments[0]

    for seg in segments[1:]:
        parts = seg.split("_")

        # Need at least: key_value → 2 tokens
        if len(parts) < 2:
            logger.debug(f"Skipping malformed segment '{seg}' in '{activity_id}'")
            continue

        # Prefer 2-token compound keys (e.g. "vehicle_type_hgv" → key="vehicle_type", value="hgv")
        # Fall back to 1-token key (e.g. "type_diesel" → key="type", value="diesel")
        if len(parts) >= 3:
            key = "_".join(parts[:2])   # e.g. "vehicle_type"
            value = "_".join(parts[2:]) # e.g. "hgv" or "gt_20t"
        else:
            key = parts[0]              # e.g. "type"
            value = parts[1]            # e.g. "diesel"

        # If the same key appears twice, append with a numeric suffix
        if key in result:
            suffix = 2
            while f"{key}_{suffix}" in result:
                suffix += 1
            key = f"{key}_{suffix}"

        result[key] = value

    return result


# ---------------------------------------------------------------------------
# Step 4: Data Cleaning
# ---------------------------------------------------------------------------

# Country/year suffixes: _IN_25, _US_2020, _gb_2019, etc.
_COUNTRY_YEAR_RE = re.compile(
    r"_[A-Za-z]{2,3}_\d{2,4}$"
)

# Standalone operator tokens to strip entirely
_INVALID_FRAGMENTS = {"lt", "gt", "gte", "lte", "lt_", "gt_", "gte_", "lte_"}

# Operator normalisation map
_OPERATOR_MAP = {
    "gt":  ">",
    "lt":  "<",
    "gte": ">=",
    "lte": "<=",
}


def clean_value(value: str) -> str | None:
    """
    Apply cleaning and normalisation rules to a parsed attribute value.

    Rules applied in order:
      1. Strip whitespace.
      2. "na" or values starting with "na_" → None.
      3. Remove country/year suffix patterns (_IN_25, _US_2020).
      4. Normalise numeric separators: 3_5 → 3.5.
      5. Normalise comparison operators: gt → >, lt → <, etc.
      6. Remove invalid lone operator fragments ("lt", "gt", "lt_", …).
      7. Return None for empty strings after cleaning.

    Args:
        value: Raw string value from parse_activity_id.

    Returns:
        Cleaned string, or None if the value should be treated as missing.
    """
    if not isinstance(value, str):
        return None

    v = value.strip().lower()

    # Rule 2 – null sentinel
    if v == "na" or v.startswith("na_"):
        return None

    # Rule 3 – remove country/year suffix
    v = _COUNTRY_YEAR_RE.sub("", v)

    # Rule 6 – remove invalid lone fragments
    if v in _INVALID_FRAGMENTS:
        return None

    # Rule 5 – normalise operators (replace whole-word tokens, underscore-aware)
    def _replace_operator(match: re.Match) -> str:
        return _OPERATOR_MAP.get(match.group(1), match.group(1))

    v = re.sub(r"(?<![a-z])(gte|lte|gt|lt)(?=_|$)", _replace_operator, v)

    # Rule 4 – normalise numeric underscore separators
    # Only when flanked by digits: "3_5" → "3.5" but NOT "vehicle_type"
    v = re.sub(r"(?<=\d)_(?=\d)", ".", v)

    return v if v else None


# ---------------------------------------------------------------------------
# Step 5: Range Normalisation
# ---------------------------------------------------------------------------

# Matches patterns like: ">20t", "3.5kg", "<=100", "50", "20_30t"
_RANGE_RE = re.compile(
    r"^(?P<op>[><=]{0,2})\s*(?P<num1>\d+\.?\d*)\s*(?:[-–]\s*(?P<num2>\d+\.?\d*))?\s*(?P<unit>[a-zA-Z%°]*)$"
)


def normalize_range(value: str | None) -> dict[str, Any] | str | None:
    """
    Attempt to parse numeric or range-like values into structured dicts.

    Examples:
        ">20t"    → {"min": 20, "unit": "t"}
        "3.5kg"   → {"value": 3.5, "unit": "kg"}
        "20-30t"  → {"min": 20, "max": 30, "unit": "t"}
        "diesel"  → "diesel"   (non-numeric, returned as-is)
        None      → None

    Args:
        value: Cleaned string from clean_value(), or None.

    Returns:
        A dict for numeric/range values, the original string for non-numeric,
        or None for missing values.
    """
    if value is None:
        return None

    match = _RANGE_RE.match(value.strip())
    if not match:
        return value  # Non-numeric — return as plain string

    op    = match.group("op") or ""
    num1  = float(match.group("num1"))
    num2  = match.group("num2")
    unit  = match.group("unit") or None

    result: dict[str, Any] = {}

    if num2 is not None:
        # Explicit range: "20-30t"
        result["min"] = num1
        result["max"] = float(num2)
    elif op in (">", ">="):
        result["min"] = num1
    elif op in ("<", "<="):
        result["max"] = num1
    else:
        result["value"] = num1

    if unit:
        result["unit"] = unit

    return result


# ---------------------------------------------------------------------------
# Step 6 & 7: Aggregation + Output
# ---------------------------------------------------------------------------

def _is_numeric_key(key: str) -> bool:
    """
    Heuristic: keys containing 'weight', 'size', 'distance', 'capacity',
    'power', 'volume', or 'speed' are treated as numeric attributes.
    """
    numeric_keywords = {
        "weight", "size", "distance", "capacity",
        "power", "volume", "speed", "range", "age", "length"
    }
    return any(kw in key for kw in numeric_keywords)


def aggregate_attributes(parsed_records: list[dict]) -> dict[str, list]:
    """
    Collect unique cleaned values for each attribute across all parsed records.

    For numeric-like keys, values are passed through normalize_range().
    Duplicate structured dicts are deduplicated via JSON serialisation.

    Args:
        parsed_records: List of dicts from parse_activity_id.

    Returns:
        Dict mapping attribute name → sorted list of unique values.
    """
    # Use a unified structure: list of unique values, tracked via a seen set
    # seen stores JSON-serialised representations for deduplication
    aggregated: dict[str, list] = {}
    seen:       dict[str, set]  = {}

    for record in parsed_records:
        for key, raw_value in record.items():
            # Determine the cleaned/normalised final value
            if key == "category":
                # Also apply country/year suffix stripping to categories
                raw_stripped = raw_value.strip().lower() if isinstance(raw_value, str) else None
                final_value = _COUNTRY_YEAR_RE.sub("", raw_stripped) if raw_stripped else None
                final_value = final_value if final_value else None
            else:
                cleaned = clean_value(raw_value)
                if cleaned is None:
                    continue
                final_value = normalize_range(cleaned) if _is_numeric_key(key) else cleaned

            if final_value is None:
                continue

            # Serialise for deduplication (works for both str and dict)
            fingerprint = json.dumps(final_value, sort_keys=True)

            if fingerprint not in seen.setdefault(key, set()):
                seen[key].add(fingerprint)
                aggregated.setdefault(key, []).append(final_value)

    # Sort string-only lists for deterministic output; leave mixed/dict lists as-is
    result = {}
    for key, values in aggregated.items():
        if not values:
            continue
        if all(isinstance(v, str) for v in values):
            result[key] = sorted(values)
        else:
            result[key] = values

    return result


# ---------------------------------------------------------------------------
# Main Pipeline Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    filepath: str,
    keywords: str | list[str],
    output_dir: str = ".",
    chunksize: int = 50_000,
) -> dict[str, list]:
    """
    End-to-end emission factors data processing pipeline.

    Steps:
      1. Load the compressed JSON dataset.
      2. Deduplicate rows so activity_id is unique (first occurrence kept).
      3. Filter rows by keyword(s) in `activity_id`.
      4. Parse each `activity_id` into structured key-value pairs.
      5. Clean and normalise values.
      6. Normalise numeric/range attributes.
      7. Aggregate unique values per attribute.
      8. Save output JSON and return the result dict.

    Args:
        filepath:   Path to the .json.gz (or .json) dataset file.
        keywords:   Keyword string or list of keywords to filter by.
        output_dir: Directory to write the output JSON file (default: cwd).
        chunksize:  Records per read-batch during loading (default 50,000).
                    Increase for faster I/O on machines with ample RAM;
                    decrease if memory is tight.

    Returns:
        Aggregated attribute dict, e.g.:
        {
            "vehicle_type": ["hgv", "lgv"],
            "fuel_source":  ["diesel", "petrol"],
            "vehicle_weight": [{"min": 20, "unit": "t"}]
        }

    Example:
        >>> result = run_pipeline("emissions.json.gz", "vehicle")
        >>> result = run_pipeline("emissions.json.gz", ["vehicle", "freight"])
    """
    logger.info("=" * 60)
    logger.info("Emission Factors Pipeline — START")
    logger.info("=" * 60)

    # ── Step 1: Load ──────────────────────────────────────────────
    df = load_data(filepath, chunksize=chunksize)

    # ── Step 2: Deduplicate ───────────────────────────────────────
    # activity_id is not unique in the raw dataset — the same ID can appear
    # across multiple records (different year/region/factor). Dedup first so
    # all downstream operations work on one row per structural pattern.
    df = deduplicate_activity_ids(df)

    # ── Step 3: Filter ────────────────────────────────────────────
    filtered_df = filter_by_keyword(df, keywords)
    if filtered_df.empty:
        logger.warning("No records matched the given keyword(s). Exiting.")
        return {}

    # ── Step 4 & 5 & 6: Parse → Clean → Normalise ────────────────
    # Each activity_id is now unique after Step 2 deduplication.
    logger.info("Parsing and cleaning activity_id fields …")
    parsed_records = []
    skipped = 0

    for activity_id in filtered_df["activity_id"]:
        parsed = parse_activity_id(activity_id)
        if parsed:
            parsed_records.append(parsed)
        else:
            skipped += 1

    logger.info(
        f"Parsed {len(parsed_records):,} unique activity_ids "
        f"({skipped} skipped due to missing/malformed activity_id)."
    )

    # ── Step 7: Aggregate ─────────────────────────────────────────
    logger.info("Aggregating unique attribute values …")
    result = aggregate_attributes(parsed_records)
    logger.info(f"Aggregated {len(result)} distinct attributes.")

    # ── Step 8: Output ────────────────────────────────────────────
    keyword_tag = (
        keywords if isinstance(keywords, str)
        else "_".join(keywords)
    )
    output_filename = Path(output_dir) / f"{keyword_tag}_output.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Output saved → {output_filename}")
    logger.info("=" * 60)
    logger.info("Emission Factors Pipeline — DONE")
    logger.info("=" * 60)

    return result


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python emission_pipeline.py <file.json.gz> <keyword> [keyword2 ...]")
        print("Example: python emission_pipeline.py emissions.json.gz vehicle freight")
        sys.exit(1)

    file_path = sys.argv[1]
    kw_args   = sys.argv[2:]  # supports multiple keywords

    output = run_pipeline(file_path, kw_args)

    print("\n── Pipeline Result ──────────────────────────────")
    print(json.dumps(output, indent=2, default=str))