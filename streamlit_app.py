import base64
import os
import runpy
import streamlit as st


BASE_DIR = os.path.dirname(__file__)
FAVICON_PATH = os.path.join(BASE_DIR, "Source", "assets", "favicon.png")


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
        # Child pages may already have set this during reruns.
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

page = st.query_params.get("page", "landing")

if page == "matcher":
    runpy.run_path(os.path.join(BASE_DIR, "Source", "name_matchingUI.py"), run_name="__main__")
elif page == "slm":
    runpy.run_path(os.path.join(BASE_DIR, "Source", "slm_ui.py"), run_name="__main__")
elif page == "whymatch":
    runpy.run_path(os.path.join(BASE_DIR, "Source", "whymatchUI.py"), run_name="__main__")
elif page in ("rematchpricing", "pricing"):
    runpy.run_path(os.path.join(BASE_DIR, "Source", "matchpricingUI.py"), run_name="__main__")
elif page == "howmatchworks":
    runpy.run_path(os.path.join(BASE_DIR, "Source", "howmatchworksUI.py"), run_name="__main__")
else:
    runpy.run_path(os.path.join(BASE_DIR, "Source", "landingUI.py"), run_name="__main__")
