"""
Unit Tests — Emission Factors Pipeline
=======================================
Run with:
    python -m pytest test_emission_pipeline.py -v
"""

import json
import gzip
import tempfile
import os
import pytest
import pandas as pd

from emission_pipeline import (
    filter_by_keyword,
    parse_activity_id,
    clean_value,
    normalize_range,
    aggregate_attributes,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "activity_id": [
            "freight_vehicle-vehicle_type_hgv-fuel_source_diesel-vehicle_weight_gt_20t-distance_basis_sfd",
            "freight_vehicle-vehicle_type_lgv-fuel_source_petrol-vehicle_weight_3_5t-distance_basis_tfd",
            "electricity_supply-fuel_source_na-grid_type_national",
            None,
            "malformed",
        ]
    })


@pytest.fixture
def sample_gz(tmp_path):
    records = [
        {"activity_id": "freight_vehicle-vehicle_type_hgv-fuel_source_diesel-vehicle_weight_gt_20t"},
        {"activity_id": "freight_vehicle-vehicle_type_lgv-fuel_source_petrol"},
        {"activity_id": "electricity_supply-fuel_source_na"},
    ]
    gz_file = tmp_path / "test.json.gz"
    with gzip.open(gz_file, "wt") as f:
        json.dump(records, f)
    return str(gz_file)


# ---------------------------------------------------------------------------
# filter_by_keyword
# ---------------------------------------------------------------------------

class TestFilterByKeyword:

    def test_basic_filter(self, sample_df):
        result = filter_by_keyword(sample_df, "vehicle")
        assert len(result) == 2
        assert all("vehicle" in aid for aid in result["activity_id"])

    def test_case_insensitive(self, sample_df):
        result = filter_by_keyword(sample_df, "VEHICLE")
        assert len(result) == 2

    def test_null_safe(self, sample_df):
        # Should not raise, None rows just excluded
        result = filter_by_keyword(sample_df, "vehicle")
        assert result["activity_id"].isna().sum() == 0

    def test_multiple_keywords(self, sample_df):
        result = filter_by_keyword(sample_df, ["vehicle", "electricity"])
        assert len(result) == 3

    def test_no_match(self, sample_df):
        result = filter_by_keyword(sample_df, "nuclear")
        assert result.empty

    def test_empty_keyword_raises(self, sample_df):
        with pytest.raises(ValueError):
            filter_by_keyword(sample_df, "")

    def test_string_keyword_works(self, sample_df):
        result = filter_by_keyword(sample_df, "electricity")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# parse_activity_id
# ---------------------------------------------------------------------------

class TestParseActivityId:

    def test_full_parse(self):
        aid = "freight_vehicle-vehicle_type_hgv-fuel_source_diesel-vehicle_weight_gt_20t-distance_basis_sfd"
        result = parse_activity_id(aid)
        assert result["category"] == "freight_vehicle"
        assert result["vehicle_type"] == "hgv"
        assert result["fuel_source"] == "diesel"
        assert result["vehicle_weight"] == "gt_20t"
        assert result["distance_basis"] == "sfd"

    def test_empty_string(self):
        assert parse_activity_id("") == {}

    def test_none_input(self):
        assert parse_activity_id(None) == {}

    def test_category_only(self):
        result = parse_activity_id("electricity_supply")
        assert result["category"] == "electricity_supply"
        assert len(result) == 1

    def test_malformed_segment_skipped(self):
        # Single-token segment (no underscore) should be skipped; 2-token "bad_val" is now valid
        result = parse_activity_id("category-nounderscore-vehicle_type_hgv")
        assert "vehicle_type" in result
        assert "category" in result

    def test_duplicate_key_suffixed(self):
        aid = "cat-fuel_source_diesel-fuel_source_petrol"
        result = parse_activity_id(aid)
        assert result["fuel_source"] == "diesel"
        assert "fuel_source_2" in result
        assert result["fuel_source_2"] == "petrol"


# ---------------------------------------------------------------------------
# clean_value
# ---------------------------------------------------------------------------

class TestCleanValue:

    def test_na_returns_none(self):
        assert clean_value("na") is None
        assert clean_value("NA") is None

    def test_na_prefix_returns_none(self):
        assert clean_value("na_extra") is None

    def test_country_year_suffix_removed(self):
        assert clean_value("diesel_IN_25") == "diesel"
        assert clean_value("petrol_US_2020") == "petrol"

    def test_numeric_underscore_normalised(self):
        assert clean_value("3_5") == "3.5"
        assert clean_value("20_30") == "20.30"

    def test_operator_normalised(self):
        assert clean_value("gt_20t") == ">_20t"
        assert clean_value("lt_5t") == "<_5t"
        assert clean_value("gte_100") == ">=_100"
        assert clean_value("lte_50") == "<=_50"

    def test_invalid_fragment_returns_none(self):
        assert clean_value("gt") is None
        assert clean_value("lt") is None

    def test_normal_string_unchanged(self):
        assert clean_value("diesel") == "diesel"
        assert clean_value("hgv") == "hgv"

    def test_empty_returns_none(self):
        assert clean_value("") is None
        assert clean_value("   ") is None

    def test_non_string_returns_none(self):
        assert clean_value(None) is None
        assert clean_value(42) is None


# ---------------------------------------------------------------------------
# normalize_range
# ---------------------------------------------------------------------------

class TestNormalizeRange:

    def test_plain_number(self):
        result = normalize_range("20")
        assert result == {"value": 20.0}

    def test_number_with_unit(self):
        result = normalize_range("3.5t")
        assert result == {"value": 3.5, "unit": "t"}

    def test_min_range(self):
        result = normalize_range(">20t")
        assert result == {"min": 20.0, "unit": "t"}

    def test_max_range(self):
        result = normalize_range("<100kg")
        assert result == {"max": 100.0, "unit": "kg"}

    def test_explicit_range(self):
        result = normalize_range("20-30t")
        assert result == {"min": 20.0, "max": 30.0, "unit": "t"}

    def test_non_numeric_passthrough(self):
        assert normalize_range("diesel") == "diesel"
        assert normalize_range("hgv") == "hgv"

    def test_none_passthrough(self):
        assert normalize_range(None) is None


# ---------------------------------------------------------------------------
# aggregate_attributes
# ---------------------------------------------------------------------------

class TestAggregateAttributes:

    def test_basic_aggregation(self):
        records = [
            {"category": "freight_vehicle", "vehicle_type": "hgv", "fuel_source": "diesel"},
            {"category": "freight_vehicle", "vehicle_type": "lgv", "fuel_source": "diesel"},
        ]
        result = aggregate_attributes(records)
        assert "hgv" in result["vehicle_type"]
        assert "lgv" in result["vehicle_type"]
        assert result["fuel_source"] == ["diesel"]  # deduped

    def test_na_values_excluded(self):
        records = [{"vehicle_type": "na", "fuel_source": "diesel"}]
        result = aggregate_attributes(records)
        assert "vehicle_type" not in result

    def test_empty_attributes_excluded(self):
        records = [{"vehicle_type": "", "fuel_source": "diesel"}]
        result = aggregate_attributes(records)
        assert "vehicle_type" not in result

    def test_numeric_key_produces_dict(self):
        records = [{"vehicle_weight": "20t"}]
        result = aggregate_attributes(records)
        assert isinstance(result["vehicle_weight"][0], dict)


# ---------------------------------------------------------------------------
# run_pipeline (integration)
# ---------------------------------------------------------------------------

class TestRunPipeline:

    def test_full_pipeline(self, sample_gz, tmp_path):
        result = run_pipeline(str(sample_gz), "vehicle", output_dir=str(tmp_path))
        assert isinstance(result, dict)
        assert "vehicle_type" in result or "category" in result

    def test_output_file_created(self, sample_gz, tmp_path):
        run_pipeline(str(sample_gz), "vehicle", output_dir=str(tmp_path))
        output_file = tmp_path / "vehicle_output.json"
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_no_match_returns_empty(self, sample_gz, tmp_path):
        result = run_pipeline(str(sample_gz), "nuclear", output_dir=str(tmp_path))
        assert result == {}

    def test_multiple_keywords(self, sample_gz, tmp_path):
        result = run_pipeline(str(sample_gz), ["vehicle", "electricity"], output_dir=str(tmp_path))
        assert isinstance(result, dict)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_pipeline("nonexistent.json.gz", "vehicle", output_dir=str(tmp_path))