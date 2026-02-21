# Build with AI: AI-Powered Name Matching
# Dashboards with Streamlit
# Name Matching UI Building with Streamlit and Python

# Developed By Ambuj Kumar

import os

import pandas as pd
import streamlit as st

from Source.name_matching import match_names


st.set_page_config(page_title="Name Matching", layout="wide", initial_sidebar_state="expanded")

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
        font-size: 2rem;
        color: #f7fbff;
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="nm-hero">
      <h1>AI Powered Name Matching</h1>
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


def read_table(uploaded_file):
    if uploaded_file is None:
        return None
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


@st.cache_data(show_spinner=False)
def run_matching(
    left_values: tuple[str, ...],
    right_values: tuple[str, ...],
    method_value: str,
    fuzzy_threshold: int,
    levenshtein_max_distance: int,
) -> pd.DataFrame:
    return match_names(
        list(left_values),
        list(right_values),
        method=method_value,
        fuzzy_threshold=fuzzy_threshold,
        lev_max_distance=levenshtein_max_distance,
    )


with st.sidebar:
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
        if method in {"Fuzzy Match", "Jaro-Winkler Distance Match", "AI Advance Match"}:
            threshold = st.slider("Fuzzy threshold", 0, 100, 75, 1)
        if method == "Levenshtein Match":
            lev_max_distance = st.slider("Levenshtein max distance", 0, 10, 2, 1)

        top_n = st.slider("Top matches to show", 1, 25, 10, 1)

    with st.expander("Advanced", expanded=False):
        show_previews = st.checkbox("Show data previews", value=True)
        show_only_matches = st.checkbox("Show only matched rows", value=False)

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

st.subheader("1) Provide data")
st.markdown(
    '<div class="nm-muted">Upload two files (CSV/XLSX) or use the built-in demo files from the sidebar.</div>',
    unsafe_allow_html=True,
)

up_col1, up_col2 = st.columns(2, gap="large")
with up_col1:
    left_file = st.file_uploader("Source file", type=["csv", "xlsx"])
with up_col2:
    right_file = st.file_uploader("Target file", type=["csv", "xlsx"])

if use_demo_files:
    left_df = pd.read_csv(DEMO_SOURCE_PATH) if os.path.exists(DEMO_SOURCE_PATH) else None
    right_df = pd.read_csv(DEMO_TARGET_PATH) if os.path.exists(DEMO_TARGET_PATH) else None
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

st.subheader("2) Choose columns")
col_sel_left, col_sel_right = st.columns(2, gap="large")
with col_sel_left:
    left_default = left_df.columns.tolist().index("full_name") if "full_name" in left_df.columns else 0
    left_name_col = st.selectbox("Source name column", options=left_df.columns.tolist(), index=left_default)
with col_sel_right:
    right_default = right_df.columns.tolist().index("name_in_system") if "name_in_system" in right_df.columns else 0
    right_name_col = st.selectbox("Target name column", options=right_df.columns.tolist(), index=right_default)

left_names = left_df[left_name_col].fillna("").astype(str)
right_names = right_df[right_name_col].fillna("").astype(str)

st.subheader("3) Run name matching")
run_now = st.button("Run name matching", type="primary", use_container_width=True)

has_results = "result_df" in st.session_state
if run_now:
    with st.spinner("Matching names..."):
        st.session_state["result_df"] = run_matching(
            tuple(left_names.tolist()),
            tuple(right_names.tolist()),
            method_key,
            int(threshold),
            int(lev_max_distance),
        )
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

st.subheader("Results")
st.dataframe(result_df, use_container_width=True, height=420)

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

st.subheader("Top potential matches")
if "score" in result_df.columns:
    top_df = result_df.sort_values("score", ascending=False).head(int(top_n))
else:
    top_df = result_df.head(int(top_n))
st.dataframe(top_df, use_container_width=True, height=320)

csv_data = result_df.to_csv(index=False).encode("utf-8")
is_premium_user = st.session_state.get("is_premium_user", False)
if is_premium_user:
    st.download_button(
        "Download Match Results (CSV)",
        data=csv_data,
        file_name="name_match_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    if st.button("Download Match Results (CSV)", use_container_width=True):
        st.warning("Premium required: upgrade your plan to download match results.")
        st.info("After payment, set `st.session_state['is_premium_user'] = True`.")

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
        include_app_context = st.checkbox("Include current app context", value=True)
with clear_col:
    clear_chat = st.button("Clear chat", use_container_width=True)

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
if prompt_col1.button("Best method?", use_container_width=True):
    st.session_state["quick_prompt"] = "Which method is best for messy customer names?"
if prompt_col2.button("Tune threshold", use_container_width=True):
    st.session_state["quick_prompt"] = "How should I tune the threshold to reduce false positives?"
if prompt_col3.button("Explain mismatch", use_container_width=True):
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
            f"Top matches: {int(top_n)}",
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
