import base64
import os
import streamlit as st
import streamlit.components.v1 as components

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
FAVICON_PATH = os.path.join(ASSETS_DIR, "favicon.png.jpeg")
LANDING_CSS_PATH = os.path.join(ASSETS_DIR, "landing.css")


def _safe_set_page_config(**kwargs):
    try:
        st.set_page_config(**kwargs)
    except Exception:
        pass


def _to_data_uri(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lower().replace(".", "")
    mime = "image/png" if ext == "png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def _load_css(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


_safe_set_page_config(
    page_title="ReMatch | Landing",
    page_icon=FAVICON_PATH,
    layout="wide",
    initial_sidebar_state="collapsed",
)

logo_data_uri = _to_data_uri(LOGO_PATH)
landing_css = _load_css(LANDING_CSS_PATH)

if landing_css:
    st.markdown(f"<style>{landing_css}</style>", unsafe_allow_html=True)

_contact_dialog = None
if hasattr(st, "dialog"):
    _contact_dialog = st.dialog
elif hasattr(st, "experimental_dialog"):
    _contact_dialog = st.experimental_dialog


def _contact_dialog_body() -> None:
    st.markdown(
        """
Use the details below to reach the BrainCal Tech team directly.

**Email**: [braincaldemo@gmail.com](mailto:braincaldemo@gmail.com)

**Alternative**: [surabhi.singh@braincal.com](mailto:surabhi.singh@braincal.com)

If you want, include your company name and the type of workflow you want to automate.
""".strip()
    )


_dialog_decorator = None
if hasattr(st, "dialog"):
    _dialog_decorator = st.dialog
elif hasattr(st, "experimental_dialog"):
    _dialog_decorator = st.experimental_dialog


def _demo_dialog_body() -> None:
    st.caption("Book a time directly in Calendly without leaving the page.")
    components.html(
        """
<style>
  .calendly-badge { display: none; }
</style>
<iframe
    src="https://cal.com/braincal-studies-gnvni3/30mins?overlayCalendar=true"
    width="100%"
    height="800"
    frameborder="0"
    style="border:0; border-radius: 12px; background: white;"
></iframe>
""".strip(),
        height=800,
    )


if _dialog_decorator is not None:

    @_dialog_decorator("Book a Demo", width="large")
    def _show_demo_dialog() -> None:
        _demo_dialog_body()
        if st.button("Close", key="demo_close"):
            st.session_state["show_demo_dialog"] = False
            st.rerun()

else:

    def _show_demo_dialog() -> None:
        _demo_dialog_body()


if _contact_dialog is not None:

    @_contact_dialog("Get in Touch", width="small")
    def _show_contact_dialog() -> None:
        _contact_dialog_body()
        if st.button("Close"):
            st.rerun()

else:

    def _show_contact_dialog() -> None:
        _contact_dialog_body()


# -------------------------------
# LANDING PAGE HTML
# -------------------------------

st.markdown(
    f"""
<section class="landing-shell-clean">

<nav class="landing-nav-clean">
    <div class="brand-wrap-clean">
        <img class="brand-logo-clean" src="{logo_data_uri}" alt="ReMatch" />
        <span class="brand-dot-clean">•</span>
        <span class="brand-tagline-clean">AI-Powered Insured Name Matching for Reinsurance</span>
    </div>

    <div class="nav-cta-clean">

        <a class="btn-touch-clean" href="?page=landing&demo=1" target="_self">Get in Touch</a>

        <a class="btn-nav-clean" href="https://cal.com/braincal-studies-gnvni3/30mins?overlayCalendar=true" target="_blank">Book a Demo</a>
        <a class="btn-nav-clean" href="?page=matcher">Tool Play</a>
    </div>
</nav>

<main class="hero-clean">
    <div class="hero-inner-clean">
        <div class="hero-chip-clean">✧ AI-Powered</div>

        <h1>Your exposure is only as good<br>as your data</h1>

        <p>
            ReMatch cleans and matches messy reinsurance bdx data using AI techniques and six intelligent
            matching methods, so the numbers you rely on are the numbers you can trust.
        </p>

        <div class="hero-cta-clean">
            <a class="btn-nav-clean" href="?page=whymatch">Why Match</a>
            <a class="btn-nav-clean" href="?page=rematchpricing">Price Model</a>
            <a class="btn-nav-clean" href="?page=howmatchworks">How it Works</a>
        </div>
    </div>
</main>

</section>
""",
    unsafe_allow_html=True,
)

_open_contact_dialog = str(st.query_params.get("contact", "")).strip() == "1"
if _open_contact_dialog:
    try:
        del st.query_params["contact"]
    except KeyError:
        pass

_open_demo_dialog = str(st.query_params.get("demo", "")).strip() == "1"
if _open_demo_dialog:
    st.session_state["show_demo_dialog"] = True
    try:
        del st.query_params["demo"]
    except KeyError:
        pass

if st.session_state.get("show_demo_dialog"):
    _show_demo_dialog()

if _open_contact_dialog:
    _show_contact_dialog()
