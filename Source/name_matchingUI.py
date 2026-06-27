import base64
import os
import runpy

import streamlit as st


BASE_DIR = os.path.dirname(__file__)
FAVICON_PATH = os.path.join(BASE_DIR, "assets", "favicon.png")


def _to_data_uri(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _safe_set_page_config(**kwargs) -> None:
    try:
        st.set_page_config(**kwargs)
    except Exception:
        pass


_safe_set_page_config(page_icon=FAVICON_PATH)

favicon_data_uri = _to_data_uri(FAVICON_PATH)
if favicon_data_uri:
    st.markdown(
        f"""
        <script>
        (function() {{
            const href = "{favicon_data_uri}";
            let icon = document.querySelector("link[rel='icon']")
                || document.querySelector("link[rel='shortcut icon']")
                || document.querySelector("link[rel*='icon']");
            if (!icon) {{
                icon = document.createElement("link");
                icon.setAttribute("rel", "icon");
                document.head.appendChild(icon);
            }}
            icon.setAttribute("type", "image/png");
            icon.setAttribute("href", href);
        }})();
        </script>
        """,
        unsafe_allow_html=True,
    )

runpy.run_path(os.path.join(BASE_DIR, "name_matchingaviationUI.py"), run_name="__main__")