# Build with AI: AI-Powered Name Matching
# Dashboards with Streamlit
# Name Matching UI Building with Streamlit and Python

# Developed By Ambuj Kumar

import os
import base64
import json
import sqlite3
import xml.etree.ElementTree as ET
from io import BytesIO

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html

FAVICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "favicon.png.jpeg")



st.set_page_config(
    page_title="Name Matching",
    page_icon=FAVICON_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --nm-page-bg: #e8edf6;
        --nm-hero-blue-deep: #13347f;
        --nm-hero-blue-mid: #1b4aa8;
        --nm-hero-blue-soft: #2558be;
        --nm-ink: #132b54;
        --nm-muted: #4e6180;
        --nm-chip-bg: rgba(255,255,255,0.15);
        --nm-chip-border: rgba(255,255,255,0.35);
      }
      .stApp {
        background:
          radial-gradient(1100px 500px at 12% -10%, #f5f9ff 0%, transparent 58%),
          radial-gradient(900px 420px at 100% 0%, #dde8fb 0%, transparent 55%),
          linear-gradient(180deg, #edf2f9 0%, var(--nm-page-bg) 55%);
      }
      .block-container { padding-top: 0.9rem; padding-bottom: 1.6rem; }
      .nm-hero {
        padding: 0.95rem 1.05rem 0.9rem 1.05rem;
        border: 1px solid rgba(20,50,112,0.2);
        border-radius: 0.65rem;
        background: linear-gradient(135deg, var(--nm-hero-blue-deep), var(--nm-hero-blue-mid) 52%, var(--nm-hero-blue-soft));
        box-shadow: 0 14px 28px rgba(15, 38, 86, 0.25);
        margin-bottom: 1.15rem;
      }
      .nm-hero h1 {
        margin: 0 0 0.2rem 0;
        font-size: 1.45rem;
        color: #f7fbff;
        line-height: 1.15;
      }
      .nm-hero-title-row {
        display: flex;
        align-items: baseline;
        justify-content: flex-start;
        gap: 0.45rem;
        flex-wrap: nowrap;
      }
      .nm-powered-by {
        color: #ff4d4d;
        font-size: 1.45rem;
        font-weight: 700;
        line-height: 1.15;
        white-space: nowrap;
        animation: nm-rainbow-blink 3.4s linear infinite;
      }
      .nm-powered-by a {
        color: inherit;
        text-decoration: underline;
      }
      .nm-powered-by a:hover {
        color: #ffffff;
      }
      @keyframes nm-rainbow-blink {
        0%   { color: #ff4d4d; opacity: 1; }
        58%  { color: #ff4d4d; opacity: 1; }
        64%  { color: #ff9f43; opacity: 0.2; }
        70%  { color: #ffe66d; opacity: 1; }
        76%  { color: #2ed573; opacity: 0.2; }
        82%  { color: #1e90ff; opacity: 1; }
        88%  { color: #5352ed; opacity: 0.2; }
        94%  { color: #a55eea; opacity: 1; }
        100% { color: #ff4d4d; opacity: 0.2; }
      }
      .nm-hero p {
        margin: 0;
        color: rgba(245, 251, 255, 0.9);
        max-width: 72rem;
        font-size: 1.02rem;
      }
      .nm-chip-row {
        margin-top: 0.7rem;
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
      }
      .nm-chip {
        border: 1px solid var(--nm-chip-border);
        border-radius: 999px;
        padding: 0.2rem 0.6rem;
        background: var(--nm-chip-bg);
        color: #ecf5ff;
        font-weight: 600;
      }
      h2, h3, .st-emotion-cache-10trblm, .st-emotion-cache-16idsys p {
        color: var(--nm-ink);
      }
      div[data-testid="stMetric"] {
        border: 1px solid rgba(47, 77, 131, 0.18);
        border-radius: 0.6rem;
        padding: 0.9rem 1rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(245,249,255,0.8));
        box-shadow: 0 6px 16px rgba(16, 40, 84, 0.08);
      }
      div[data-testid="stDataFrame"] {
        border: 1px solid rgba(47, 77, 131, 0.18);
        border-radius: 0.5rem;
        overflow: hidden;
        box-shadow: 0 6px 16px rgba(16, 40, 84, 0.08);
      }
      div[data-testid="stDataFrame"] [role="columnheader"] {
        font-weight: 700 !important;
      }
      div[data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(180deg, #f9fbff 0%, #f2f6fc 100%);
        border: 1px solid #aec0dc;
        border-radius: 0.85rem;
      }
      div[data-testid="stFileUploaderDropzone"] section {
        color: #1b335e;
      }
      div[data-testid="stFileUploaderDropzone"] small {
        color: #657a99;
      }
      div[data-testid="stFileUploaderDropzone"] button {
        border-radius: 0.7rem;
        border: 1px solid #b5c2d6;
        background: linear-gradient(180deg, #f7faff, #edf2f8);
        color: #1b335e;
      }
      div[data-testid="stAlert"] {
        border-radius: 0.75rem;
        border: 1px solid #b7d2c4;
        box-shadow: 0 4px 12px rgba(31, 104, 77, 0.09);
      }
      div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 0.9rem;
        border: 1px solid #afc0d9 !important;
        background: linear-gradient(180deg, #f2f6fb 0%, #ebf1f8 100%);
        box-shadow: 0 8px 20px rgba(20, 46, 93, 0.07);
      }
      .nm-muted { opacity: 0.7; font-size: 0.88rem; }
      .nm-chat-shell {
        margin-top: 1rem;
        border: 1px solid rgba(20,40,80,0.12);
        border-radius: 1rem;
        padding: 0.95rem 1rem 1rem 1rem;
        background: linear-gradient(180deg, #f3f7fc 0%, #e8eef6 100%);
        box-shadow: 0 10px 20px rgba(20, 46, 93, 0.08);
      }
      .nm-chat-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.8rem;
        margin-bottom: 0.25rem;
      }
      .nm-chat-head h3 {
        margin: 0;
        color: #132b54;
        font-size: 1.05rem;
      }
      .nm-chat-badge {
        border: 1px solid #9eb2d6;
        border-radius: 999px;
        padding: 0.16rem 0.78rem;
        background: #dfe7f3;
        color: #2a4f8b;
        font-size: 0.86rem;
        font-weight: 600;
      }
      div[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: #343a4c;
        border-radius: 999px;
        padding: 0.55rem 0.9rem;
        margin-bottom: 0.7rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 6px 14px rgba(22, 28, 42, 0.18);
      }
      div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: #2d3342;
      }
      div[data-testid="stSidebar"] div[role="radiogroup"] label div {
        color: #f2f5fb !important;
        font-weight: 700;
        font-size: 0.98rem;
      }
      div[data-testid="stSidebar"] div[role="radiogroup"] input:checked + div {
        color: #ffffff !important;
      }
      div[data-testid="stSidebar"] .stButton > button {
        justify-content: flex-start;
        text-align: left;
        padding-left: 1rem;
      }
      div[data-testid="stSidebar"] .stButton > button span {
        width: 100%;
        text-align: left;
      }
      .nm-chat-quick {
        margin-top: 0.35rem;
        margin-bottom: 0.25rem;
        color: #132b54;
        font-size: 1.08rem;
        font-weight: 700;
      }
      .nm-muted { color: var(--nm-muted); opacity: 1; }
      .stButton > button[kind="primary"] {
        border: 0;
        border-radius: 0.6rem;
        background: linear-gradient(90deg, #143f98 0%, #1e54bc 100%);
        color: #f6faff;
        box-shadow: 0 8px 18px rgba(20, 63, 152, 0.28);
      }
      .stButton > button[kind="primary"]:hover {
        filter: brightness(1.03);
      }
      .nm-footer {
        margin-top: 1.2rem;
        padding-top: 0.6rem;
        border-top: 1px solid rgba(19, 43, 84, 0.2);
        text-align: center;
        color: #4e6180;
        font-size: 0.86rem;
      }
      /* Hide Streamlit footer/status, keep header so sidebar toggle is visible */
      footer,
      div[data-testid="stDecoration"],
      div[data-testid="stStatusWidget"] {
        visibility: hidden;
        height: 0;
        position: fixed;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="nm-hero">
      <div class="nm-hero-title-row">
        <h1 style="margin:0;">AI Powered Name Matching</h1>
        <div class="nm-powered-by">By BrainCal Tech Team <a href="https://braincal.com" target="_blank" rel="noopener noreferrer">https://braincal.com</a></div>
      </div>
      <p>Enterprise-grade name matching with exact, fuzzy, Soundex, Jaro-Winkler, Levenshtein, and AI Advance methods.</p>
      <div class="nm-chip-row">
        <span class="nm-chip">Operational Data Quality</span>
        <span class="nm-chip">Customer Record Matching</span>
        <span class="nm-chip">AI Assistant Enabled</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = os.path.dirname(__file__)
DEMO_SOURCE_PATH = os.path.join(BASE_DIR, "demo_data", "source_names.csv")
DEMO_TARGET_PATH = os.path.join(BASE_DIR, "demo_data", "target_names.csv")
DATA_UPLOAD_DB_PATH = os.path.join(BASE_DIR, "data_upload.db")
CONTROL_SETTINGS_XML_PATH = os.path.join(BASE_DIR, "control_settings.xml")

CONTROL_REGISTRY = [
    {"id": "data_upload_button", "type": "button", "label": "Data Upload"},
    {"id": "data_preview_button", "type": "button", "label": "Data Preview"},
    {"id": "run_name_matching_button", "type": "button", "label": "Run name matching"},
    {"id": "clear_chat_button", "type": "button", "label": "Clear chat"},
    {"id": "best_method_button", "type": "button", "label": "Best method?"},
    {"id": "tune_threshold_button", "type": "button", "label": "Tune threshold"},
    {"id": "explain_mismatch_button", "type": "button", "label": "Explain mismatch"},
    {"id": "show_data_previews_checkbox", "type": "checkbox", "label": "Show data previews"},
    {"id": "show_only_matches_checkbox", "type": "checkbox", "label": "Show only matched rows"},
    {"id": "include_location_checkbox", "type": "checkbox", "label": "Include Location"},
    {"id": "include_industry_checkbox", "type": "checkbox", "label": "Include Industry"},
    {"id": "include_app_context_checkbox", "type": "checkbox", "label": "Include current app context"},
]


def _default_control_settings() -> dict[str, bool]:
    return {control["id"]: True for control in CONTROL_REGISTRY}


def _load_control_settings(xml_path: str = CONTROL_SETTINGS_XML_PATH) -> dict[str, bool]:
    settings = _default_control_settings()
    if not os.path.exists(xml_path):
        _save_control_settings(settings, xml_path)
        return settings

    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        _save_control_settings(settings, xml_path)
        return settings

    for control_elem in root.findall("control"):
        control_id = control_elem.get("id", "").strip()
        enabled_value = str(control_elem.get("enabled", "true")).strip().lower()
        if control_id in settings:
            settings[control_id] = enabled_value in {"1", "true", "yes", "on"}
    return settings


def _save_control_settings(settings: dict[str, bool], xml_path: str = CONTROL_SETTINGS_XML_PATH) -> None:
    root = ET.Element("controls")
    for control in CONTROL_REGISTRY:
        control_id = control["id"]
        ET.SubElement(
            root,
            "control",
            attrib={
                "id": control_id,
                "type": control["type"],
                "label": control["label"],
                "enabled": "true" if bool(settings.get(control_id, True)) else "false",
            },
        )

    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


control_settings = _load_control_settings()


def _is_control_enabled(control_id: str) -> bool:
    return bool(control_settings.get(control_id, True))


def _file_mtime_iso(path: str) -> str:
    try:
        return pd.Timestamp.fromtimestamp(os.path.getmtime(path)).isoformat()
    except OSError:
        return "unknown"


def _clean_display_path(path: str) -> str:
    normalized = os.path.normpath(path)
    try:
        return os.path.relpath(normalized, start=os.getcwd())
    except ValueError:
        return normalized


def _get_openai_client():
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _chat_system_prompt() -> str:
    return (
        "You are a helpful assistant embedded in a Streamlit name matching app. "
        "Give concise and practical guidance about matching methods, thresholds, score interpretation, "
        "and troubleshooting. Do not request or reveal API keys."
    )


@st.cache_data(show_spinner=False)
def _read_table_from_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        return pd.read_csv(BytesIO(file_bytes))
    return pd.read_excel(BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def _read_demo_csv(path: str, file_mtime: float) -> pd.DataFrame:
    _ = file_mtime  # cache key includes mtime so updates invalidate cache
    return pd.read_csv(path)


def read_table(uploaded_file):
    if uploaded_file is None:
        return None
    return _read_table_from_bytes(uploaded_file.getvalue(), uploaded_file.name)


def _init_data_upload_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS data_upload_batches (
                upload_id TEXT PRIMARY KEY,
                source_filename TEXT NOT NULL,
                uploaded_at_utc TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                column_count INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS data_upload_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id TEXT NOT NULL,
                row_number INTEGER NOT NULL,
                row_data_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS data_upload_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id TEXT NOT NULL,
                source_column TEXT NOT NULL,
                destination_column TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _save_data_upload_to_db(
    source_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    source_filename: str,
    db_path: str = DATA_UPLOAD_DB_PATH,
) -> tuple[str, int]:
    _init_data_upload_db(db_path)
    upload_id = pd.Timestamp.utcnow().strftime("upload_%Y%m%d%H%M%S%f")
    upload_time = pd.Timestamp.utcnow().isoformat()

    safe_source_df = source_df.astype(object).where(pd.notna(source_df), None)
    row_payloads = [
        (upload_id, idx + 1, json.dumps(record, ensure_ascii=True))
        for idx, record in enumerate(safe_source_df.to_dict(orient="records"))
    ]
    mapping_payloads = [
        (upload_id, str(row["Source Column"]), str(row["Destination Column"]))
        for _, row in mapping_df.iterrows()
    ]

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO data_upload_batches (
                upload_id, source_filename, uploaded_at_utc, row_count, column_count
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (upload_id, source_filename, upload_time, len(source_df), len(source_df.columns)),
        )
        conn.executemany(
            """
            INSERT INTO data_upload_rows (upload_id, row_number, row_data_json)
            VALUES (?, ?, ?)
            """,
            row_payloads,
        )
        conn.executemany(
            """
            INSERT INTO data_upload_mappings (upload_id, source_column, destination_column)
            VALUES (?, ?, ?)
            """,
            mapping_payloads,
        )
        conn.commit()

    return upload_id, len(row_payloads)


def _compose_match_values(
    df: pd.DataFrame,
    primary_col: str,
    extra_cols: list[str] | None = None,
) -> pd.Series:
    primary_values = df[primary_col].fillna("").astype(str).str.strip()
    if not extra_cols:
        return primary_values

    combined_values = primary_values.copy()
    for col_name in extra_cols:
        col_values = df[col_name].fillna("").astype(str).str.strip()
        combined_values = combined_values.where(col_values.eq(""), combined_values + " " + col_values)
    return combined_values.str.replace(r"\s+", " ", regex=True).str.strip()


@st.cache_data(show_spinner=False)
def run_matching(
    left_values: tuple[str, ...],
    right_values: tuple[str, ...],
    method_value: str,
    fuzzy_threshold: int,
    levenshtein_max_distance: int,
    levenshtein_engine: str,
) -> pd.DataFrame:
    from Source.namematching import match_names

    return match_names(
        list(left_values),
        list(right_values),
        method=method_value,
        fuzzy_threshold=fuzzy_threshold,
        lev_max_distance=levenshtein_max_distance,
        lev_engine=levenshtein_engine,
    )


with st.sidebar:
    menu_options = {
        "Data Upload": "📥 Data Upload",
        "Name Matching": "🔎 Name Matching",
        "Tower Matching": "🗼 Tower Matching",
        "Admin": "⚙️ Admin",
    }
    if "sidebar_menu" not in st.session_state:
        st.session_state["sidebar_menu"] = "Name Matching"
    for key, label in menu_options.items():
        is_active = st.session_state["sidebar_menu"] == key
        if st.button(
            label,
            use_container_width=True,
            type="primary" if is_active else "secondary",
            key=f"menu_btn_{key}",
        ):
            st.session_state["sidebar_menu"] = key
            st.rerun()
    sidebar_menu = st.session_state["sidebar_menu"]
    st.divider()

    st.header("Matching Settings")
    st.caption("Configure matching and provide data.")

    use_demo_files = st.toggle("Use built-in demo files", value=True)

    with st.expander("Matching settings", expanded=True):
        method = st.selectbox(
            "Method",
            [
                "Exact Match",
                "Fuzzy Match",
                "Soundex Match",
                "Jaro-Winkler Distance Match",
                "Levenshtein Match",
                "AI Advance Match",
            ],
        )

        threshold = 75
        lev_max_distance = 2
        lev_engine = "auto"
        if method in {"Fuzzy Match", "Jaro-Winkler Distance Match", "AI Advance Match"}:
            threshold = st.slider("Fuzzy threshold", 0, 100, 75, 1)
        if method == "Levenshtein Match":
            lev_max_distance = st.slider("Levenshtein max distance", 0, 10, 2, 1)
            lev_engine = st.selectbox(
                "Levenshtein engine",
                ["Auto", "RapidFuzz", "Python"],
                help="Auto uses RapidFuzz when available, otherwise Python fallback.",
            ).lower()

        top_n = st.slider("Top matches to show", 1, 25, 10, 1)

    with st.expander("Advanced", expanded=False):
        show_previews = st.checkbox(
            "Show data previews",
            value=True,
            disabled=not _is_control_enabled("show_data_previews_checkbox"),
        )
        show_only_matches = st.checkbox(
            "Show only matched rows",
            value=False,
            disabled=not _is_control_enabled("show_only_matches_checkbox"),
        )
        max_rows_to_render = st.slider("Rows to render in UI tables", 100, 5000, 1000, 100)

    with st.expander("Debug", expanded=False):
        st.caption(f"Loaded: `{_clean_display_path(__file__)}`")
        st.caption(f"mtime: `{_file_mtime_iso(__file__)}`")

method_key = {
    "Exact Match": "exact",
    "Fuzzy Match": "fuzzy",
    "Soundex Match": "soundex",
    "Jaro-Winkler Distance Match": "jaro_winkler",
    "Levenshtein Match": "levenshtein",
    "AI Advance Match": "ai_advanced",
}[method]

if sidebar_menu == "Data Upload":
    st.subheader("Data Upload")
    source_upload_df = None
    mapping_df = pd.DataFrame(columns=["Source Column", "Destination Column"])
    preview_clicked = False
    uploaded_source_file = st.file_uploader(
        "Source file selector",
        type=["csv", "xlsx"],
        key="data_upload_source_file",
    )
    if uploaded_source_file is not None:
        source_upload_df = read_table(uploaded_source_file)
        if source_upload_df is None:
            st.warning("Could not read the selected file.")
        else:
            file_col, preview_col = st.columns([4, 1], gap="small")
            with file_col:
                st.success(f"Selected: `{uploaded_source_file.name}`")
            with preview_col:
                preview_clicked = st.button(
                    "Data Preview",
                    use_container_width=True,
                    disabled=not _is_control_enabled("data_preview_button"),
                )

            if preview_clicked:
                with st.expander("Uploaded Data Preview", expanded=True):
                    st.dataframe(source_upload_df.head(100), use_container_width=True, height=320)

            st.markdown("### Source and Destination Column Mapping")
            st.caption("Map each source column to a destination column using dropdowns.")

            source_columns = [str(col) for col in source_upload_df.columns.tolist()]
            header_col1, header_col2 = st.columns(2, gap="small")
            with header_col1:
                st.markdown("**Source Columns**")
            with header_col2:
                st.markdown("**Destination Columns**")

            mapped_rows: list[dict[str, str]] = []
            for idx, default_source in enumerate(source_columns):
                row_col1, row_col2 = st.columns(2, gap="small")
                with row_col1:
                    selected_source = st.selectbox(
                        f"Source Column {idx + 1}",
                        options=source_columns,
                        index=idx,
                        key=f"data_upload_source_col_{idx}",
                        label_visibility="collapsed",
                    )
                with row_col2:
                    default_dest_index = (
                        source_columns.index(default_source)
                        if default_source in source_columns
                        else 0
                    )
                    selected_destination = st.selectbox(
                        f"Destination Column {idx + 1}",
                        options=source_columns,
                        index=default_dest_index,
                        key=f"data_upload_dest_col_{idx}",
                        label_visibility="collapsed",
                    )
                mapped_rows.append(
                    {
                        "Source Column": selected_source,
                        "Destination Column": selected_destination,
                    }
                )

            mapping_df = pd.DataFrame(mapped_rows)
            st.session_state["data_upload_column_mapping"] = mapping_df
    else:
        st.info("Choose a CSV/XLSX file.")

    upload_button_clicked = st.button(
        "Data Upload",
        type="primary",
        use_container_width=True,
        disabled=(
            uploaded_source_file is None
            or source_upload_df is None
            or not _is_control_enabled("data_upload_button")
        ),
    )
    if upload_button_clicked and uploaded_source_file is not None and source_upload_df is not None:
        with st.spinner("Saving uploaded data to DB..."):
            upload_id, saved_row_count = _save_data_upload_to_db(
                source_upload_df,
                mapping_df,
                uploaded_source_file.name,
            )
        st.success(
            f"Data uploaded to DB successfully. Upload ID: `{upload_id}`. "
            f"Saved rows: {saved_row_count}. DB: `{DATA_UPLOAD_DB_PATH}`"
        )

    st.markdown("Switch to **Name Matching** from the left menu to run matching.")
    st.stop()

if sidebar_menu == "Tower Matching":
    st.subheader("Tower Matching")

    tower_template_path = os.path.join(BASE_DIR, "demo_data", "Tower_Matching_Template.xlsx")
    if os.path.exists(tower_template_path):
        with open(tower_template_path, "rb") as template_file:
            template_bytes = template_file.read()
        b64 = base64.b64encode(template_bytes).decode("ascii")
        href = (
            f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;'
            f'base64,{b64}" download="Tower_Matching_Template.xlsx"'
            ' style="text-decoration: underline;">'
            "Tower Matching Template</a>"
        )
        st.markdown(
            f"""
            <div style="display:flex; align-items:baseline; gap:0.5rem; flex-wrap:wrap;">
              <span>Upload a tower match file (CSV/XLSX). Use previews to sanity check the data.</span>
              {href}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("Upload a tower match file (CSV/XLSX). Use previews to sanity check the data.")

    tower_source_file = st.file_uploader(
        "**Tower Source File**",
        type=["csv", "xlsx"],
        key="tower_source_file",
    )

    tower_source_df = read_table(tower_source_file)

    if tower_source_file is not None and tower_source_df is None:
        st.warning("Could not read the tower source file.")

    def _render_tower_mapping(label: str, df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
        st.markdown(f"### {label} Column Mapping")
        st.caption("Map each source column to a destination column using dropdowns.")

        tower_columns = [str(col) for col in df.columns.tolist()]
        header_col1, header_col2 = st.columns(2, gap="small")
        with header_col1:
            st.markdown("**Source Columns**")
        with header_col2:
            st.markdown("**Destination Columns**")

        mapped_rows: list[dict[str, str]] = []
        for idx, default_source in enumerate(tower_columns):
            row_col1, row_col2 = st.columns(2, gap="small")
            with row_col1:
                selected_source = st.selectbox(
                    f"{label} Source Column {idx + 1}",
                    options=tower_columns,
                    index=idx,
                    key=f"{key_prefix}_source_col_{idx}",
                    label_visibility="collapsed",
                )
            with row_col2:
                default_dest_index = (
                    tower_columns.index(default_source)
                    if default_source in tower_columns
                    else 0
                )
                selected_destination = st.selectbox(
                    f"{label} Destination Column {idx + 1}",
                    options=tower_columns,
                    index=default_dest_index,
                    key=f"{key_prefix}_dest_col_{idx}",
                    label_visibility="collapsed",
                )
            mapped_rows.append(
                {
                    "Source Column": selected_source,
                    "Destination Column": selected_destination,
                }
            )

        mapping_df = pd.DataFrame(mapped_rows)
        st.session_state[f"{key_prefix}_column_mapping"] = mapping_df
        return mapping_df

    if tower_source_df is not None:
        st.markdown("**Tower Source Preview**")
        st.dataframe(tower_source_df.head(100), use_container_width=True, height=320)

    def _ensure_min_columns(df: pd.DataFrame, min_cols: int) -> pd.DataFrame:
        if df.shape[1] >= min_cols:
            return df
        extra_needed = min_cols - df.shape[1]
        extra_cols = []
        base_idx = 1
        while len(extra_cols) < extra_needed:
            name = f"_col_{df.shape[1] + base_idx}"
            if name not in df.columns:
                extra_cols.append(name)
            base_idx += 1
        for col_name in extra_cols:
            df[col_name] = ""
        return df

    def TM_Formula(source_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Port of the VBA TM_Formula logic using positional columns (A=0 ... O=14).
        Returns: (working_df, working_next_df, final_result_df)
        """
        df = _ensure_min_columns(source_df.copy(), 15)
        columns = list(df.columns)
        rows = df.to_numpy(copy=True).tolist()

        working_next_rows: list[list] = []
        final_rows: list[list] = []

        max_cycles = 1000
        max_rows = 30000

        # Column positions (0-based)
        col_c = 2
        col_d = 3
        col_e = 4
        col_f = 5
        col_h = 7
        col_j = 9
        col_k = 10
        col_l = 11
        col_m = 12
        col_n = 13
        col_o = 14

        for j in range(max_cycles + 1):
            ws_lr = len(rows)
            if ws_lr <= 1 or ws_lr == 1048576:
                break

            counts: dict[object, int] = {}
            i = 1  # 0-based index for Excel row 2
            while i < min(ws_lr, max_rows):
                row = rows[i]
                prev_row = rows[i - 1]

                current_c = row[col_c]
                k_val = counts.get(current_c, 0)
                row[col_k] = k_val
                counts[current_c] = k_val + 1

                row[col_l] = "primary" if row[col_f] == 0 else "non-primary"

                if row[col_k] == 0:
                    row[col_m] = 0
                else:
                    if (
                        row[col_f] == 0
                        and row[col_c] == prev_row[col_c]
                        and row[col_d] == prev_row[col_d]
                        and row[col_e] == prev_row[col_e]
                        and row[col_f] == prev_row[col_f]
                        and row[col_h] == prev_row[col_h]
                        and row[col_j] != prev_row[col_j]
                    ):
                        row[col_m] = 0
                    else:
                        row[col_m] = "next cycle"

                if row[col_k] == 0:
                    row[col_n] = 0
                else:
                    if row[col_f] >= (prev_row[col_e] + prev_row[col_f]):
                        row[col_n] = 0
                    else:
                        if row[col_e] == prev_row[col_e] and row[col_f] == prev_row[col_f]:
                            row[col_n] = 0
                        else:
                            row[col_n] = "next cycle"

                row[col_o] = row[col_m] if row[col_l] == "primary" else row[col_n]

                if row[col_k] == -1:
                    if i > 1:
                        for idx in range(1, i):
                            rows[idx][col_m:col_o + 1] = [j, j, j]
                        final_rows.extend(rows[1:i])
                    rows = rows[:1] + rows[i + 1 :]
                    if working_next_rows:
                        rows = working_next_rows + rows
                        working_next_rows = []
                    break

                if row[col_o] == "next cycle":
                    working_next_rows.append(row.copy())
                    rows.pop(i)
                    counts[current_c] = counts.get(current_c, 1) - 1
                    ws_lr = len(rows)
                    continue

                i += 1

        working_df = pd.DataFrame(rows, columns=columns)
        working_next_df = pd.DataFrame(working_next_rows, columns=columns)
        final_result_df = pd.DataFrame(final_rows, columns=columns)
        return working_df, working_next_df, final_result_df

    if tower_source_df is None:
        st.info("Upload the tower source file to get started.")
        st.stop()

    if st.button("Tower Match", type="primary", use_container_width=True):
        with st.spinner("Running Tower Match formula..."):
            working_df, working_next_df, final_result_df = TM_Formula(tower_source_df)

        st.success("Tower Match complete.")
        st.markdown("### Working Sheet (Processed)")
        st.dataframe(working_df.head(200), use_container_width=True, height=360)

        if not working_next_df.empty:
            st.markdown("### Working Sheet Next")
            st.dataframe(working_next_df.head(200), use_container_width=True, height=360)

        if not final_result_df.empty:
            st.markdown("### Final Result")
            st.dataframe(final_result_df.head(200), use_container_width=True, height=360)

    st.stop()

if sidebar_menu == "Admin":
    st.subheader("Admin")
    st.caption("Enable or disable app buttons and checkboxes, then save to XML.")

    button_controls = [c for c in CONTROL_REGISTRY if c["type"] == "button"]
    checkbox_controls = [c for c in CONTROL_REGISTRY if c["type"] == "checkbox"]
    updated_settings = dict(control_settings)

    st.markdown("### Buttons")
    for control in button_controls:
        updated_settings[control["id"]] = st.checkbox(
            control["label"],
            value=bool(control_settings.get(control["id"], True)),
            key=f"admin_control_{control['id']}",
        )

    st.markdown("### Checkboxes")
    for control in checkbox_controls:
        updated_settings[control["id"]] = st.checkbox(
            control["label"],
            value=bool(control_settings.get(control["id"], True)),
            key=f"admin_control_{control['id']}",
        )

    save_admin_controls = st.button("Save Admin Settings", type="primary", use_container_width=True)
    if save_admin_controls:
        _save_control_settings(updated_settings)
        st.success(f"Settings saved to `{CONTROL_SETTINGS_XML_PATH}`")
        st.rerun()

    st.stop()

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
      <h3 style="margin:0;">Provide data</h3>
      <div class="nm-muted" style="margin:0;">
        Upload two files (CSV/XLSX) or use the built-in demo files from the sidebar.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

up_col1, up_col2 = st.columns(2, gap="large")
with up_col1:
    left_file = st.file_uploader("**Source file**", type=["csv", "xlsx"])
with up_col2:
    right_file = st.file_uploader("**Reference File**", type=["csv", "xlsx"])

if use_demo_files:
    left_df = (
        _read_demo_csv(DEMO_SOURCE_PATH, os.path.getmtime(DEMO_SOURCE_PATH))
        if os.path.exists(DEMO_SOURCE_PATH)
        else None
    )
    right_df = (
        _read_demo_csv(DEMO_TARGET_PATH, os.path.getmtime(DEMO_TARGET_PATH))
        if os.path.exists(DEMO_TARGET_PATH)
        else None
    )
    st.success(f"Using demo files: `{DEMO_SOURCE_PATH}` and `{DEMO_TARGET_PATH}`")
else:
    left_df = read_table(left_file)
    right_df = read_table(right_file)

if left_df is None or right_df is None:
    st.info("Upload both files to start matching, or enable demo files in the sidebar.")
    st.stop()

if show_previews:
    with st.container(border=True):
        st.markdown("**Data Previews**")
        prev_col1, prev_col2 = st.columns(2, gap="large")
        with prev_col1:
            with st.expander("Preview: Source", expanded=False):
                st.dataframe(left_df.head(25), use_container_width=True, height=280)
        with prev_col2:
            with st.expander("Preview: Target", expanded=False):
                st.dataframe(right_df.head(25), use_container_width=True, height=280)

choose_col_header, include_loc_col, include_ind_col = st.columns([3, 1.4, 1.4], gap="small")
with choose_col_header:
    st.subheader("Choose columns")
with include_loc_col:
    include_location = st.checkbox(
        "Include Location",
        value=False,
        help="Append the location column to the primary name before matching when available.",
        disabled=not _is_control_enabled("include_location_checkbox"),
    )
with include_ind_col:
    include_industry = st.checkbox(
        "Include Industry",
        value=False,
        help="Append the industry column to the primary name before matching when available.",
        disabled=not _is_control_enabled("include_industry_checkbox"),
    )

col_sel_left, col_sel_right = st.columns(2, gap="large")
with col_sel_left:
    left_default = left_df.columns.tolist().index("full_name") if "full_name" in left_df.columns else 0
    left_name_col = st.selectbox("Source name column", options=left_df.columns.tolist(), index=left_default)
with col_sel_right:
    right_default = right_df.columns.tolist().index("name_in_system") if "name_in_system" in right_df.columns else 0
    right_name_col = st.selectbox("Target name column", options=right_df.columns.tolist(), index=right_default)

left_location_col: str | None = None
right_location_col: str | None = None
left_industry_col: str | None = None
right_industry_col: str | None = None

if include_location:
    location_left, location_right = st.columns(2, gap="large")
    with location_left:
        left_location_options = [c for c in left_df.columns.tolist() if c != left_name_col]
        if left_location_options:
            left_location_default = (
                left_location_options.index("location") if "location" in left_location_options else 0
            )
            left_location_col = st.selectbox(
                "Source location column",
                options=left_location_options,
                index=left_location_default,
            )
        else:
            st.caption("No source columns available for location selection.")
    with location_right:
        right_location_options = [c for c in right_df.columns.tolist() if c != right_name_col]
        if right_location_options:
            right_location_default = (
                right_location_options.index("location") if "location" in right_location_options else 0
            )
            right_location_col = st.selectbox(
                "Target location column",
                options=right_location_options,
                index=right_location_default,
            )
        else:
            st.caption("No target columns available for location selection.")

if include_industry:
    industry_left, industry_right = st.columns(2, gap="large")
    with industry_left:
        left_industry_options = [c for c in left_df.columns.tolist() if c != left_name_col]
        if left_industry_options:
            left_industry_default = (
                left_industry_options.index("industry") if "industry" in left_industry_options else 0
            )
            left_industry_col = st.selectbox(
                "Source industry column",
                options=left_industry_options,
                index=left_industry_default,
            )
        else:
            st.caption("No source columns available for industry selection.")
    with industry_right:
        right_industry_options = [c for c in right_df.columns.tolist() if c != right_name_col]
        if right_industry_options:
            right_industry_default = (
                right_industry_options.index("industry") if "industry" in right_industry_options else 0
            )
            right_industry_col = st.selectbox(
                "Target industry column",
                options=right_industry_options,
                index=right_industry_default,
            )
        else:
            st.caption("No target columns available for industry selection.")

left_extra_cols: list[str] = []
right_extra_cols: list[str] = []
if include_location and left_location_col and left_location_col != left_name_col:
    left_extra_cols.append(left_location_col)
if include_location and right_location_col and right_location_col != right_name_col:
    right_extra_cols.append(right_location_col)
if include_industry and left_industry_col and left_industry_col != left_name_col:
    if left_industry_col not in left_extra_cols:
        left_extra_cols.append(left_industry_col)
if include_industry and right_industry_col and right_industry_col != right_name_col:
    if right_industry_col not in right_extra_cols:
        right_extra_cols.append(right_industry_col)

left_names = _compose_match_values(left_df, left_name_col, left_extra_cols)
right_names = _compose_match_values(right_df, right_name_col, right_extra_cols)

run_header_col, run_button_col = st.columns([3, 1.4], gap="small")
with run_header_col:
    st.subheader("Run name matching")
with run_button_col:
    run_now = st.button(
        "Run name matching",
        type="primary",
        use_container_width=True,
        disabled=not _is_control_enabled("run_name_matching_button"),
    )

has_results = "result_df" in st.session_state
if run_now:
    with st.spinner("Matching names..."):
        st.session_state["result_df"] = run_matching(
            tuple(left_names.tolist()),
            tuple(right_names.tolist()),
            method_key,
            int(threshold),
            int(lev_max_distance),
            lev_engine,
        )
    st.session_state["scroll_to_top_matches"] = True
    has_results = True

if not has_results:
    st.info("Click **Run name matching** to generate results.")
    st.stop()

full_result_df = st.session_state.get("result_df")
if full_result_df is None:
    st.info("No results available yet. Run matching again.")
    st.stop()

result_df = full_result_df
if show_only_matches and "is_match" in full_result_df.columns:
    result_df = full_result_df[full_result_df["is_match"]].copy()

result_df_view = result_df.head(int(max_rows_to_render))
if len(result_df) > len(result_df_view):
    st.caption(f"Showing first {len(result_df_view):,} of {len(result_df):,} rows for faster rendering.")


def _style_matched_rows(df: pd.DataFrame):
    match_col = "is_match" if "is_match" in df.columns else ("Is Match" if "Is Match" in df.columns else None)
    if match_col is None:
        return df

    def _highlight_row(row: pd.Series) -> list[str]:
        if bool(row.get(match_col, False)):
            return ["background-color: #e8f5e9"] * len(row)
        return [""] * len(row)

    return df.style.apply(_highlight_row, axis=1)


def _to_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda col: str(col).replace("_", " ").title())


st.dataframe(
    _style_matched_rows(_to_display_columns(result_df_view)),
    use_container_width=True,
    height=420,
)

metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.metric("Source rows", len(full_result_df))
with metric_col2:
    matched_count = int(full_result_df["is_match"].sum()) if "is_match" in full_result_df.columns else 0
    st.metric("Matched rows", matched_count)
with metric_col3:
    if "is_match" in full_result_df.columns and len(full_result_df) > 0:
        match_rate = 100 * float(full_result_df["is_match"].mean())
    else:
        match_rate = 0.0
    st.metric("Match rate", f"{match_rate:.1f}%")

st.markdown('<div id="top-matches-section"></div>', unsafe_allow_html=True)
st.subheader("Top potential matches")
if "score" in result_df.columns:
    top_df = result_df.sort_values("score", ascending=False).head(int(top_n))
else:
    top_df = result_df.head(int(top_n))
st.dataframe(
    _style_matched_rows(_to_display_columns(top_df)),
    use_container_width=True,
    height=320,
)

if st.session_state.pop("scroll_to_top_matches", False):
    html(
        """
        <script>
          const scrollToMatches = () => {
            const el = window.parent.document.getElementById("top-matches-section");
            if (el && el.scrollIntoView) {
              el.scrollIntoView({behavior: "smooth", block: "start"});
            }
          };
          setTimeout(scrollToMatches, 120);
        </script>
        """,
        height=0,
    )

download_source_df = left_df.loc[result_df.index].reset_index(drop=True)
download_match_cols = [c for c in result_df.columns if c not in download_source_df.columns]
download_df = pd.concat(
    [download_source_df, result_df[download_match_cols].reset_index(drop=True)],
    axis=1,
)
csv_data = download_df.to_csv(index=False).encode("utf-8")
download_clicked = st.download_button(
    "Download Match Results (CSV)",
    data=csv_data,
    file_name="name_match_results.csv",
    mime="text/csv",
    use_container_width=True,
)
if download_clicked:
    html(
        """
        <script>
          window.open("https://www.linkedin.com/in/ambuj-kumar-60a20a5/", "_blank");
        </script>
        """,
        height=0,
    )

st.markdown('<div class="nm-chat-shell">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="nm-chat-head">
      <h3>Chat Assistant</h3>
      <span class="nm-chat-badge">AI Guidance</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="nm-muted">Ask about matching methods, thresholds, or troubleshooting. (Set `OPENAI_API_KEY` to enable.)</div>',
    unsafe_allow_html=True,
)

settings_col, clear_col = st.columns([4, 1], gap="small")
with settings_col:
    with st.expander("Assistant settings", expanded=False):
        chat_model = st.text_input("Model", value="gpt-4o-mini")
        chat_temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        include_app_context = st.checkbox(
            "Include current app context",
            value=True,
            disabled=not _is_control_enabled("include_app_context_checkbox"),
        )
with clear_col:
    clear_chat = st.button(
        "Clear chat",
        use_container_width=True,
        disabled=not _is_control_enabled("clear_chat_button"),
    )

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {
            "role": "assistant",
            "content": "Hi! Ask me which matching method to use, how to pick thresholds, or why a row did not match.",
        }
    ]

if clear_chat:
    st.session_state["chat_messages"] = [
        {"role": "assistant", "content": "Chat cleared. Ask a new question when you're ready."}
    ]
    st.rerun()

st.markdown('<div class="nm-chat-quick">Quick prompts</div>', unsafe_allow_html=True)
prompt_col1, prompt_col2, prompt_col3 = st.columns(3, gap="small")
if prompt_col1.button(
    "Best method?",
    use_container_width=True,
    disabled=not _is_control_enabled("best_method_button"),
):
    st.session_state["quick_prompt"] = "Which method is best for messy customer names?"
if prompt_col2.button(
    "Tune threshold",
    use_container_width=True,
    disabled=not _is_control_enabled("tune_threshold_button"),
):
    st.session_state["quick_prompt"] = "How should I tune the threshold to reduce false positives?"
if prompt_col3.button(
    "Explain mismatch",
    use_container_width=True,
    disabled=not _is_control_enabled("explain_mismatch_button"),
):
    st.session_state["quick_prompt"] = "Why might two similar names fail to match?"

for msg in st.session_state["chat_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_prompt = st.chat_input("Ask the assistant...")
if not user_prompt and "quick_prompt" in st.session_state:
    user_prompt = st.session_state.pop("quick_prompt")

if user_prompt:
    st.session_state["chat_messages"].append({"role": "user", "content": user_prompt})

    client = _get_openai_client()
    if client is None:
        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "content": (
                    "I cannot call the AI model yet. Set `OPENAI_API_KEY` in your environment "
                    "or `.streamlit/secrets.toml` and try again."
                ),
            }
        )
        st.rerun()

    context_lines: list[str] = []
    if include_app_context:
        context_lines = [
            f"Selected method: {method}",
            f"Method key: {method_key}",
            f"Fuzzy threshold: {int(threshold)}",
            f"Levenshtein max distance: {int(lev_max_distance)}",
            f"Levenshtein engine: {lev_engine}",
            f"Top matches: {int(top_n)}",
            f"Optional source columns: {', '.join(left_extra_cols) if left_extra_cols else 'None'}",
            f"Optional target columns: {', '.join(right_extra_cols) if right_extra_cols else 'None'}",
            f"Show only matches: {bool(show_only_matches)}",
            f"Rows in result: {len(result_df)}",
        ]

    user_content = user_prompt
    if context_lines:
        user_content = f"{user_prompt}\n\nApp context:\n" + "\n".join(context_lines)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model=chat_model,
                    messages=[
                        {"role": "system", "content": _chat_system_prompt()},
                        *st.session_state["chat_messages"][:-1],
                        {"role": "user", "content": user_content},
                    ],
                    temperature=float(chat_temperature),
                )
                answer = response.choices[0].message.content or ""
            except Exception as error:
                answer = f"Sorry, I hit an error calling the model: `{type(error).__name__}`."

            st.markdown(answer)
            st.session_state["chat_messages"].append({"role": "assistant", "content": answer})

st.markdown("</div>", unsafe_allow_html=True)
st.markdown(
    '<div class="nm-footer">&copy; 2026 braincal.com. All rights reserved</div>',
    unsafe_allow_html=True,
)

