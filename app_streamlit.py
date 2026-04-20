"""
Streamlit App — Emission Factors Pipeline Explorer
====================================================
Run with:
    streamlit run app_streamlit.py

Key fix in this version:
    Streamlit re-runs the entire script on every interaction (button click,
    slider move, etc.). When the button is clicked, the file uploader widget
    resets and uploaded_file becomes None — this is why you saw
    "Please upload a dataset file" even though the file was there.

    Fix: save the file to st.session_state as soon as it is uploaded,
    BEFORE the button is clicked. The session_state persists across re-runs
    so the file path is always available when Run Pipeline is pressed.
"""

import json
import tempfile
import os
import streamlit as st
import pandas as pd

from emission_pipeline import (
    load_data,
    filter_by_keyword,
    parse_activity_id,
    aggregate_attributes,
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emission Factors Pipeline",
    page_icon="🌍",
    layout="wide",
)

st.markdown("""
<style>
.tag {
    display: inline-block;
    background: #1e3a2f;
    color: #00ff99;
    border-radius: 4px;
    padding: 2px 8px;
    margin: 2px;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ───────────────────────────────────────────────────────
if "tmp_path" not in st.session_state:
    st.session_state.tmp_path = None
if "upload_name" not in st.session_state:
    st.session_state.upload_name = None


# ── Helpers ──────────────────────────────────────────────────────────────────
CHUNK_BYTES = 8 * 1024 * 1024  # 8 MB write chunks


def save_upload_chunked(uploaded_file) -> str:
    """Write UploadedFile to disk in 8 MB chunks. Returns temp file path."""
    suffix = ".json.gz" if uploaded_file.name.endswith(".gz") else ".json"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        uploaded_file.seek(0)
        while True:
            chunk = uploaded_file.read(CHUNK_BYTES)
            if not chunk:
                break
            tmp.write(chunk)
    finally:
        tmp.close()
    return tmp.name


def render_attribute_values(values: list):
    """Safely render a list that may contain dicts, strings, or both."""
    if not values:
        return
    dicts   = [v for v in values if isinstance(v, dict)]
    strings = [str(v) for v in values if not isinstance(v, dict)]
    if dicts:
        all_keys = sorted({k for d in dicts for k in d})
        rows = [{k: d.get(k) for k in all_keys} for d in dicts]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    if strings:
        if dicts:
            st.caption("Also present as plain values:")
        PAGE = 50
        visible = strings[:PAGE]
        st.markdown(
            " ".join(f'<span class="tag">{v}</span>' for v in visible),
            unsafe_allow_html=True,
        )
        if len(strings) > PAGE:
            with st.expander(f"… {len(strings) - PAGE} more values"):
                st.markdown(
                    " ".join(f'<span class="tag">{v}</span>' for v in strings[PAGE:]),
                    unsafe_allow_html=True,
                )


def build_parsed_table(filtered_df: pd.DataFrame, max_rows: int = 500) -> pd.DataFrame:
    """
    Return a DataFrame where each row contains the original activity_id
    plus all parsed attribute columns side by side.

    Caps at max_rows for display performance.
    """
    subset = filtered_df["activity_id"].head(max_rows)
    rows = []
    for aid in subset:
        parsed = parse_activity_id(aid)
        row = {"activity_id": aid}
        row.update(parsed)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Header ───────────────────────────────────────────────────────────────────
st.title("🌍 Emission Factors Pipeline")
st.markdown("*Extract and explore structured attributes from large emission factor datasets.*")
st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    uploaded_file = st.file_uploader(
        "Upload dataset (.json.gz or .json)",
        type=["gz", "json", "jsonl"],
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.upload_name:
            if st.session_state.tmp_path and os.path.exists(st.session_state.tmp_path):
                try:
                    os.unlink(st.session_state.tmp_path)
                except OSError:
                    pass

            with st.spinner("Saving file to disk …"):
                st.session_state.tmp_path   = save_upload_chunked(uploaded_file)
                st.session_state.upload_name = uploaded_file.name

            st.success(f"Ready: {uploaded_file.name}")

    elif uploaded_file is None and st.session_state.tmp_path:
        try:
            if os.path.exists(st.session_state.tmp_path):
                os.unlink(st.session_state.tmp_path)
        except OSError:
            pass
        st.session_state.tmp_path    = None
        st.session_state.upload_name = None

    keyword_input = st.text_input(
        "Keyword(s)",
        placeholder="e.g. vehicle, freight, electricity",
        help="Comma-separated for multiple keywords",
    )

    st.subheader("Performance")
    chunk_size = st.select_slider(
        "Chunk size (records per batch)",
        options=[10_000, 25_000, 50_000, 100_000, 200_000],
        value=50_000,
        help="Larger = faster but uses more RAM. 50k is safe for most machines.",
    )

    run_btn = st.button("▶ Run Pipeline", use_container_width=True, type="primary")
    st.divider()

    if st.session_state.upload_name:
        st.caption(f"📂 Loaded: {st.session_state.upload_name}")
    else:
        st.caption("No file loaded yet.")


# ── Main pipeline ─────────────────────────────────────────────────────────────
if run_btn:
    if not st.session_state.tmp_path or not os.path.exists(st.session_state.tmp_path):
        st.error("Please upload a dataset file first.")
        st.stop()

    if not keyword_input.strip():
        st.error("Please enter at least one keyword.")
        st.stop()

    keywords = [k.strip() for k in keyword_input.split(",") if k.strip()]
    tmp_path = st.session_state.tmp_path

    # Step 1 — Load
    with st.spinner("Loading dataset …"):
        df = load_data(tmp_path, chunksize=chunk_size)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total records", f"{len(df):,}")

    # Step 2 — Filter
    with st.spinner("Filtering by keyword(s) …"):
        filtered = filter_by_keyword(df, keywords)

    col2.metric("Matched records", f"{len(filtered):,}")

    if filtered.empty:
        st.warning("No records matched the given keyword(s).")
        st.stop()

    # Step 3 — Parse + Aggregate
    with st.spinner("Parsing and aggregating …"):
        parsed = [parse_activity_id(aid) for aid in filtered["activity_id"]]
        parsed = [p for p in parsed if p]
        result = aggregate_attributes(parsed)

    col3.metric("Attributes found", f"{len(result):,}")
    st.success(f"Pipeline complete — {len(result)} attributes extracted.")
    st.divider()

    # ── Results tabs ────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Attributes", "🗂 Raw JSON", "🔍 Sample Records", "🆔 Activity IDs"])

    with tab1:
        summary_rows = [
            {
                "Attribute": attr,
                "# Unique Values": len(values),
                "Sample Values": ", ".join(str(v) for v in values[:5]),
            }
            for attr, values in sorted(result.items())
        ]
        st.subheader("Attribute Summary")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        st.divider()

        search = st.text_input("🔎 Filter attributes by name", placeholder="e.g. fuel, weight, type")
        filtered_attrs = {
            attr: values for attr, values in sorted(result.items())
            if not search or search.lower() in attr.lower()
        }

        if not filtered_attrs:
            st.info("No attributes match your filter.")
        else:
            st.caption(f"Showing {len(filtered_attrs)} of {len(result)} attributes")
            for attr, values in filtered_attrs.items():
                has_dicts = any(isinstance(v, dict) for v in values)
                badge = "🔢" if has_dicts else "🏷️"
                with st.expander(f"{badge} **{attr}** — {len(values)} unique value(s)"):
                    render_attribute_values(values)

    with tab2:
        st.download_button(
            "⬇ Download JSON",
            data=json.dumps(result, indent=2, default=str),
            file_name=f"{'_'.join(keywords)}_output.json",
            mime="application/json",
        )
        st.json(result)

    with tab3:
        st.subheader("Sample Records — activity_id + Parsed Attributes")

        MAX_DISPLAY = 500
        total_matched = len(filtered)
        display_count = min(total_matched, MAX_DISPLAY)

        st.caption(
            f"Showing {display_count:,} of {total_matched:,} matched records. "
            f"Each column is a parsed attribute extracted from the activity_id."
        )

        with st.spinner(f"Parsing {display_count:,} records for display …"):
            parsed_table = build_parsed_table(filtered, max_rows=MAX_DISPLAY)

        # Put activity_id first, then sort remaining columns alphabetically
        attr_cols = sorted([c for c in parsed_table.columns if c != "activity_id"])
        parsed_table = parsed_table[["activity_id"] + attr_cols]

        st.dataframe(parsed_table, use_container_width=True, hide_index=True)

        # Download the full parsed table as CSV
        csv_data = parsed_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download displayed records as CSV",
            data=csv_data,
            file_name=f"{'_'.join(keywords)}_sample_records.csv",
            mime="text/csv",
        )

    with tab4:
        st.subheader("🆔 Matched Activity IDs")
        st.caption(f"{len(filtered):,} activity_id strings matched your keyword(s).")

        # Live search within the matched IDs
        aid_search = st.text_input(
            "🔎 Search within activity IDs",
            placeholder="e.g. diesel, hgv, gt_20t",
            key="aid_search",
        )

        all_ids = filtered["activity_id"].dropna().tolist()

        if aid_search.strip():
            display_ids = [a for a in all_ids if aid_search.lower() in a.lower()]
            st.caption(f"Showing {len(display_ids):,} results matching '{aid_search}'")
        else:
            display_ids = all_ids
            st.caption(f"Showing all {len(display_ids):,} matched activity IDs")

        # Render as a numbered dataframe so IDs are easy to read and copy
        aid_df = pd.DataFrame({"#": range(1, len(display_ids) + 1), "activity_id": display_ids})
        st.dataframe(aid_df, use_container_width=True, hide_index=True, height=500)

        # Download button
        st.download_button(
            "⬇ Download all matched activity IDs as CSV",
            data="\n".join(display_ids).encode("utf-8"),
            file_name=f"{'_'.join(keywords)}_activity_ids.csv",
            mime="text/csv",
        )

else:
    st.info("Upload a dataset file and enter a keyword, then click **Run Pipeline**.")
    with st.expander("📌 See example output"):
        st.json({
            "vehicle_type": ["hgv", "lgv", "articulated_truck"],
            "fuel_source":  ["diesel", "petrol", "cng"],
            "vehicle_weight": [
                {"min": 7.5, "unit": "t"},
                {"min": 20, "unit": "t"},
                {"value": 3.5, "unit": "t"},
            ],
            "distance_basis": ["sfd", "tfd"],
        })