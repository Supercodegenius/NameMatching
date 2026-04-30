import os
import csv
import hashlib
import json
import tempfile
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import streamlit as st


FAVICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "favicon.png.jpeg")


@lru_cache(maxsize=1)
def _pricing_paths() -> tuple[Path, Path, Path, bool]:
    """Return (data_dir, users_file, usage_file, persistent_ok).

    Streamlit Community Cloud can restrict writes in the repo directory, so we
    fall back to a temp directory when needed.
    """

    repo_dir = Path(__file__).resolve().parent.parent / "outputs" / "pricing"
    for candidate in (
        repo_dir,
        Path(tempfile.gettempdir()) / "namematching" / "pricing",
    ):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            users_file = candidate / "users.json"
            usage_file = candidate / "usage_history.csv"
            return candidate, users_file, usage_file, candidate == repo_dir
        except Exception:
            continue

    # Last resort: use temp dir but mark persistence unavailable (in-memory only).
    fallback = Path(tempfile.gettempdir()) / "namematching" / "pricing"
    return fallback, fallback / "users.json", fallback / "usage_history.csv", False


def _ensure_pricing_storage() -> bool:
    data_dir, users_file, usage_file, persistent_ok = _pricing_paths()
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        if not users_file.exists():
            users_file.write_text("{}", encoding="utf-8")
        if not usage_file.exists():
            with usage_file.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "timestamp_utc",
                        "email",
                        "billing_cycle",
                        "plan_name",
                        "included_records",
                        "usage_records",
                        "overage_records",
                        "base_subscription",
                        "overage_cost",
                        "estimated_total",
                        "price_per_record",
                    ],
                )
                writer.writeheader()
        st.session_state["pricing_persistence_ok"] = bool(persistent_ok)
        return True
    except Exception as exc:
        st.session_state["pricing_persistence_ok"] = False
        st.session_state["pricing_persistence_error"] = str(exc)
        return False


def _load_pricing_users() -> dict[str, dict[str, str]]:
    if not _ensure_pricing_storage():
        return st.session_state.get("_pricing_users", {})

    _, users_file, _, _ = _pricing_paths()
    try:
        raw = json.loads(users_file.read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def _save_pricing_users(users: dict[str, dict[str, str]]) -> None:
    if not _ensure_pricing_storage():
        st.session_state["_pricing_users"] = users
        return

    _, users_file, _, _ = _pricing_paths()
    users_file.write_text(json.dumps(users, indent=2), encoding="utf-8")


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _auth_or_register_user(email: str, password: str) -> tuple[bool, str]:
    users = _load_pricing_users()
    now_iso = datetime.now(timezone.utc).isoformat()
    password_hash = _hash_password(password)
    existing = users.get(email)

    if existing is None:
        users[email] = {
            "password_hash": password_hash,
            "created_at": now_iso,
            "last_login_at": now_iso,
        }
        _save_pricing_users(users)
        return True, "Account created and logged in."

    if str(existing.get("password_hash", "")) != password_hash:
        return False, "Invalid email or password."

    existing["last_login_at"] = now_iso
    users[email] = existing
    _save_pricing_users(users)
    return True, "Logged in successfully."


def _get_user_profile(email: str) -> dict[str, str]:
    users = _load_pricing_users()
    profile = users.get(email.strip().lower(), {})
    if not isinstance(profile, dict):
        return {}
    return {str(k): str(v) for k, v in profile.items()}


def _update_user_profile(email: str, updates: dict[str, str]) -> None:
    users = _load_pricing_users()
    key = email.strip().lower()
    existing = users.get(key, {})
    if not isinstance(existing, dict):
        existing = {}
    for update_key, update_value in updates.items():
        existing[str(update_key)] = str(update_value)
    users[key] = existing
    _save_pricing_users(users)


def _append_usage_snapshot(snapshot: dict[str, str]) -> None:
    if not _ensure_pricing_storage():
        history = st.session_state.get("_pricing_usage_rows", [])
        history.append({k: str(v) for k, v in snapshot.items()})
        st.session_state["_pricing_usage_rows"] = history
        return

    _, _, usage_file, _ = _pricing_paths()
    with usage_file.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp_utc",
                "email",
                "billing_cycle",
                "plan_name",
                "included_records",
                "usage_records",
                "overage_records",
                "base_subscription",
                "overage_cost",
                "estimated_total",
                "price_per_record",
            ],
        )
        writer.writerow(snapshot)


def _load_usage_snapshots(email: str, limit: int = 8) -> list[dict[str, str]]:
    if not _ensure_pricing_storage():
        rows = st.session_state.get("_pricing_usage_rows", [])
        filtered = [
            {str(k): str(v) for k, v in row.items()}
            for row in rows
            if str(row.get("email", "")).strip().lower() == email.strip().lower()
        ]
        return list(reversed(filtered[-limit:]))

    rows: list[dict[str, str]] = []
    _, _, usage_file, _ = _pricing_paths()
    with usage_file.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("email", "")).strip().lower() == email.strip().lower():
                rows.append({k: str(v) for k, v in row.items()})
    return list(reversed(rows[-limit:]))


def _safe_set_page_config(**kwargs) -> None:
    try:
        st.set_page_config(**kwargs)
    except Exception:
        pass


_safe_set_page_config(
    page_title="Pricing | ReMatch",
    page_icon=FAVICON_PATH,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    dedent(
        """
        <style>
          html,
          body {
            margin: 0;
          }

          .stApp {
            background:
              radial-gradient(55rem 30rem at 108% 92%, rgba(94, 157, 243, 0.14), transparent 62%),
              radial-gradient(40rem 24rem at -8% -8%, rgba(255, 255, 255, 0.72), transparent 60%),
              #f3f4f6;
          }

          div[data-testid="stHeader"],
          header[data-testid="stHeader"],
          div[data-testid="stToolbar"],
          div[data-testid="stDecoration"],
          #MainMenu,
          footer {
            display: none !important;
          }

          div[data-testid="stAppViewContainer"],
          section.main,
          section[data-testid="stMain"],
          div[data-testid="stMain"] {
            margin-top: 0 !important;
            padding-top: 0 !important;
          }

          div[data-testid="stAppViewContainer"] > .main,
          div[data-testid="stMainBlockContainer"],
          .block-container {
            max-width: 100% !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
          }

          .pricing-shell {
            max-width: min(74rem, 94vw);
            margin: 0 auto;
            padding: 1.1rem 0 1.8rem;
          }

          .pricing-header {
            text-align: center;
            margin-bottom: 1.25rem;
          }

          .pricing-header h1 {
            margin: 0;
            font-family: "Plus Jakarta Sans", sans-serif;
            font-size: clamp(2.2rem, 4.2vw, 3.6rem);
            letter-spacing: -0.02em;
            font-weight: 800;
            color: #12131a;
          }

          .pricing-header p {
            margin: 0.6rem auto 0;
            max-width: 52rem;
            font-family: "Manrope", sans-serif;
            font-size: 1.15rem;
            line-height: 1.35;
            color: #343741;
          }

          .pricing-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1.1rem;
            margin-top: 1.6rem;
          }

          .pricing-card {
            border-radius: 0.95rem;
            border: 1px solid rgba(18, 19, 26, 0.08);
            background: rgba(255, 255, 255, 0.88);
            padding: 1.4rem 1.4rem 1.3rem;
            box-shadow: 0 14px 24px rgba(21, 34, 55, 0.12);
          }

          .pricing-card h3 {
            margin: 0 0 0.6rem;
            font-family: "Plus Jakarta Sans", sans-serif;
            font-size: 1.2rem;
            color: #12131a;
            font-weight: 800;
          }

          .pricing-card .price {
            font-family: "Plus Jakarta Sans", sans-serif;
            font-weight: 800;
            font-size: 2.1rem;
            margin: 0.15rem 0 0.4rem;
            color: #0f62ff;
          }

          .pricing-card .meta {
            font-family: "Manrope", sans-serif;
            font-size: 0.95rem;
            color: #343741;
            margin: 0;
          }

          .pricing-list {
            margin: 1rem 0 0;
            padding-left: 1.2rem;
            font-family: "Manrope", sans-serif;
            color: #343741;
            line-height: 1.55;
            font-size: 0.98rem;
          }

          @media (max-width: 900px) {
            .pricing-grid {
              grid-template-columns: 1fr;
            }
          }
        </style>
        """
    ),
    unsafe_allow_html=True,
)

st.markdown(
    dedent(
        """
<section class="pricing-shell">
<header class="pricing-header">
<h1>Re<span style="color:#0f62ff;">Match</span> Pricing</h1>
<p>Easy to start, scalable later.</p>
</header>

<div class="pricing-grid">
<article class="pricing-card">
<h3>Pay As You Go</h3>
<p class="meta">Test with your own data</p>
<ul class="pricing-list">
<li>0.5 GBP/search</li>
<li>No integration required</li>
</ul>
</article>

<article class="pricing-card">
<h3>Subscription</h3>
<p class="meta">For teams ready to scale</p>
<ul class="pricing-list">
<li>Annual-usage based pricing</li>
<li>Lower effective rate per search</li>
<li>Agreed credit</li>
<li>20% upfront discount</li>
</ul>
</article>

<article class="pricing-card">
<h3>Enterprise</h3>
<p class="meta">Full platform integration</p>
<ul class="pricing-list">
<li>API access</li>
<li>Role-based access control</li>
<li>Audit logs</li>
</ul>
</article>
</div>
</section>
        """
    ),
    unsafe_allow_html=True,
)

if st.session_state.get("pricing_persistence_ok") is False:
    st.caption(
        "Note: pricing usage snapshots are running in session-only mode in this deployment "
        "(storage is not writable)."
    )

if "pricing_show_login_dialog" not in st.session_state:
    st.session_state["pricing_show_login_dialog"] = False
if "pricing_logged_in" not in st.session_state:
    st.session_state["pricing_logged_in"] = False
if "pricing_user_email" not in st.session_state:
    st.session_state["pricing_user_email"] = ""


def _open_login_flow(flow: str = "payg") -> None:
    st.query_params["page"] = "rematchpricing"
    st.query_params["pricing_flow"] = flow
    st.session_state["pricing_show_login_dialog"] = True
    st.rerun()


cta_col1, cta_col2, cta_col3 = st.columns(3, gap="small")
with cta_col1:
    if st.button("Get Started", type="primary", use_container_width=True, key="pricing_cta_payg"):
        _open_login_flow("payg")
with cta_col2:
    if st.button(
        "Get Started",
        type="secondary",
        use_container_width=True,
        key="pricing_cta_sub",
    ):
        _open_login_flow("subscription")
with cta_col3:
    if st.button("Get Started", type="secondary", use_container_width=True, key="pricing_cta_enterprise"):
        _open_login_flow("subscription")


pricing_flow = str(st.query_params.get("pricing_flow", "")).strip().lower()
if pricing_flow:
    st.query_params["page"] = "rematchpricing"


@st.dialog("Pricing setup", width="large")
def _pricing_login_dialog() -> None:
    st.markdown("#### Log in (or create an account)")
    email = st.text_input("Email", value=st.session_state.get("pricing_user_email", "")).strip().lower()
    password = st.text_input("Password", type="password").strip()

    action_col1, action_col2 = st.columns(2, gap="small")
    with action_col1:
        do_login = st.button("Continue", type="primary", use_container_width=True, key="pricing_login_continue")
    with action_col2:
        do_cancel = st.button("Cancel", use_container_width=True, key="pricing_login_cancel")

    if do_cancel:
        st.session_state["pricing_show_login_dialog"] = False
        if "pricing_flow" in st.query_params:
            del st.query_params["pricing_flow"]
        st.rerun()

    if do_login:
        ok, msg = _auth_or_register_user(email, password)
        if not ok:
            st.error(msg)
            st.stop()
        st.session_state["pricing_logged_in"] = True
        st.session_state["pricing_user_email"] = email
        st.success(msg)

    if not st.session_state.get("pricing_logged_in"):
        st.stop()

    user_email = st.session_state.get("pricing_user_email", "").strip().lower()
    profile = _get_user_profile(user_email)

    st.divider()
    st.markdown("#### Choose plan")

    plans = [
        {
            "name": "Starter",
            "included": 50000,
            "monthly_base": 399,
            "overage_price_per_record": 0.0020,
        },
        {
            "name": "Growth",
            "included": 250000,
            "monthly_base": 899,
            "overage_price_per_record": 0.0012,
        },
        {
            "name": "Enterprise",
            "included": 1000000,
            "monthly_base": 2999,
            "overage_price_per_record": 0.0008,
        },
    ]

    plan_names = [p["name"] for p in plans]
    default_plan = profile.get("plan_name") if profile.get("plan_name") in plan_names else "Starter"
    selected_plan_name = st.selectbox("Plan", plan_names, index=plan_names.index(default_plan))
    selected = next(p for p in plans if p["name"] == selected_plan_name)

    billing_cycle = st.radio(
        "Billing cycle",
        options=["Monthly", "Annual (20% discount)"],
        horizontal=True,
        index=0,
        key="pricing_billing_cycle",
    )

    included_records = int(selected["included"])
    monthly_base_fee = float(selected["monthly_base"])
    overage_price_per_record = float(selected["overage_price_per_record"])

    annual_discount_factor = 0.8 if billing_cycle.startswith("Annual") else 1.0
    base_fee_after_discount = monthly_base_fee * annual_discount_factor

    st.divider()
    st.markdown("#### Usage preview")

    usage_records = st.number_input(
        "Records processed this billing period",
        min_value=0,
        value=included_records,
        step=100,
        key="pricing_usage_records",
    )

    overage_records = max(0, int(usage_records) - included_records)
    overage_cost = overage_records * overage_price_per_record
    total_cost = base_fee_after_discount + overage_cost
    price_per_record = total_cost / max(1, int(usage_records))

    m1, m2, m3, m4 = st.columns(4, gap="small")
    with m1:
        st.metric("Included records", f"{included_records:,}")
    with m2:
        st.metric("Overage records", f"{overage_records:,}")
    with m3:
        st.metric("Estimated total", f"GBP {total_cost:,.2f}")
    with m4:
        st.metric("Price per record", f"GBP {price_per_record:.4f}")

    usage_ratio = min(1.0, int(usage_records) / max(1, included_records))
    st.progress(usage_ratio, text=f"Plan usage: {usage_ratio * 100:.1f}% of included records")

    st.divider()
    save_col, close_col = st.columns(2, gap="small")
    with save_col:
        if st.button("Save Usage Snapshot", type="primary", use_container_width=True, key="pricing_save_snapshot"):
            _update_user_profile(user_email, {"plan_name": selected_plan_name})
            _append_usage_snapshot(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "email": user_email,
                    "billing_cycle": billing_cycle,
                    "plan_name": selected_plan_name,
                    "included_records": str(included_records),
                    "usage_records": str(int(usage_records)),
                    "overage_records": str(overage_records),
                    "base_subscription": f"{base_fee_after_discount:.2f}",
                    "overage_cost": f"{overage_cost:.2f}",
                    "estimated_total": f"{total_cost:.2f}",
                    "price_per_record": f"{price_per_record:.6f}",
                }
            )
            st.success("Saved.")
    with close_col:
        if st.button("Close", use_container_width=True, key="pricing_close"):
            st.session_state["pricing_show_login_dialog"] = False
            if "pricing_flow" in st.query_params:
                del st.query_params["pricing_flow"]
            st.rerun()

    history_rows = _load_usage_snapshots(user_email, limit=8)
    if history_rows:
        st.markdown("#### Recent usage snapshots")
        st.dataframe(history_rows, use_container_width=True, height=260)


if st.session_state.get("pricing_show_login_dialog") or bool(pricing_flow):
    st.session_state["pricing_show_login_dialog"] = True
    _pricing_login_dialog()
