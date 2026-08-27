"""Supabase Google authentication entry screen for the Streamlit application."""

from __future__ import annotations

import secrets
import threading
import time
from typing import Any
from urllib.parse import urlencode

import streamlit as st
from supabase import Client, create_client
from supabase.lib.client_options import SyncClientOptions


AUTH_STYLES = """
<style>
    :root {
        --apple-red: #e94a3a;
        --apple-red-dark: #bf3028;
        --leaf: #2f7d4a;
        --ink: #25332b;
        --muted: #6f7d73;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 8% 12%, rgba(239, 107, 80, .16), transparent 28rem),
            radial-gradient(circle at 94% 82%, rgba(78, 139, 86, .18), transparent 30rem),
            linear-gradient(135deg, #fffdf8 0%, #f6f7e9 48%, #eef5e7 100%);
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"],
    [data-testid="stAppDeployButton"],
    [data-testid="stMainMenu"] { display: none; }

    .block-container {
        max-width: 1120px;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .block-container > [data-testid="stVerticalBlock"] {
        justify-content: center;
    }

    .st-key-auth-shell {
        background: rgba(255, 255, 255, .84);
        border: 1px solid rgba(63, 92, 68, .13);
        border-radius: 32px;
        box-shadow: 0 24px 70px rgba(43, 70, 49, .16);
        padding: 18px;
        overflow: hidden;
        backdrop-filter: blur(18px);
    }

    .auth-hero {
        min-height: 590px;
        border-radius: 24px;
        padding: 42px 40px 34px;
        color: white;
        background:
            radial-gradient(circle at 75% 12%, rgba(255,255,255,.18), transparent 13rem),
            linear-gradient(150deg, #215f3b 0%, #367b4b 54%, #6da45d 100%);
        position: relative;
        overflow: hidden;
    }

    .auth-hero::after {
        content: "";
        position: absolute;
        width: 280px;
        height: 280px;
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 50%;
        right: -120px;
        bottom: -110px;
        box-shadow: 0 0 0 38px rgba(255,255,255,.04), 0 0 0 78px rgba(255,255,255,.03);
    }

    .brand-kicker {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        padding: 8px 13px;
        border: 1px solid rgba(255,255,255,.24);
        background: rgba(255,255,255,.10);
        border-radius: 999px;
        font-size: .72rem;
        font-weight: 750;
        letter-spacing: .13em;
        text-transform: uppercase;
    }

    .hero-title {
        max-width: 440px;
        margin: 24px 0 10px;
        font-size: clamp(2.25rem, 4vw, 3.55rem);
        line-height: .98;
        letter-spacing: -.055em;
        font-weight: 850;
    }

    .hero-copy {
        max-width: 440px;
        color: rgba(255,255,255,.78);
        font-size: 1rem;
        line-height: 1.65;
    }

    .apple-stage {
        height: 220px;
        display: grid;
        place-items: center;
        position: relative;
    }

    .apple {
        width: 145px;
        height: 138px;
        position: relative;
        border-radius: 46% 48% 48% 46% / 54% 54% 44% 44%;
        transform: rotate(-4deg);
        background: radial-gradient(circle at 32% 23%, #ff9b72 0 8%, #ef5842 25%, #d6332d 72%);
        box-shadow: inset -18px -15px 30px rgba(117,17,20,.22), 0 25px 38px rgba(25,45,28,.30);
    }

    .apple::before {
        content: "";
        position: absolute;
        width: 48px;
        height: 28px;
        border-radius: 100% 0 100% 0;
        background: #9bc66f;
        top: -30px;
        left: 78px;
        transform: rotate(-18deg);
        box-shadow: inset -8px -3px 12px rgba(28,86,46,.26);
    }

    .apple::after {
        content: "";
        position: absolute;
        width: 9px;
        height: 42px;
        border-radius: 10px;
        background: #68452d;
        top: -29px;
        left: 66px;
        transform: rotate(11deg);
    }

    .scan-ring {
        position: absolute;
        width: 200px;
        height: 200px;
        border: 1px dashed rgba(255,255,255,.40);
        border-radius: 50%;
        animation: scan-spin 18s linear infinite;
    }

    .scan-ring::before, .scan-ring::after {
        content: "";
        position: absolute;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #dff3c0;
        box-shadow: 0 0 16px #dff3c0;
    }

    .scan-ring::before { top: 12px; left: 37px; }
    .scan-ring::after { right: 8px; bottom: 54px; }

    @keyframes scan-spin { to { transform: rotate(360deg); } }

    .ripeness-scale {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 7px;
        margin-top: 2px;
    }

    .ripeness-scale span {
        height: 9px;
        border-radius: 999px;
        background: var(--stage-color);
        box-shadow: 0 5px 12px rgba(20,40,25,.12);
    }

    .scale-labels {
        display: flex;
        justify-content: space-between;
        color: rgba(255,255,255,.62);
        font-size: .7rem;
        margin-top: 9px;
    }

    .auth-form-copy { padding: 56px 42px 14px; }

    .auth-eyebrow {
        color: var(--apple-red-dark);
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .auth-title {
        color: var(--ink);
        font-size: clamp(2rem, 3vw, 2.75rem);
        line-height: 1.06;
        letter-spacing: -.045em;
        font-weight: 850;
        margin-bottom: 14px;
    }

    .auth-subtitle {
        color: var(--muted);
        line-height: 1.65;
        margin-bottom: 28px;
    }

    .feature-row {
        display: flex;
        gap: 12px;
        align-items: flex-start;
        margin: 14px 0;
        color: #405047;
        font-size: .91rem;
    }

    .feature-icon {
        display: grid;
        place-items: center;
        min-width: 31px;
        height: 31px;
        border-radius: 10px;
        color: var(--leaf);
        background: #eaf3e4;
        font-weight: 800;
    }

    .st-key-login-actions { padding: 5px 42px 0; }

    .st-key-login-actions button,
    .st-key-login-actions a {
        min-height: 3.2rem;
        border-radius: 14px;
        font-weight: 750;
        transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }

    .st-key-login-actions button:hover,
    .st-key-login-actions a:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(43,70,49,.13);
    }

    .st-key-login-actions button[kind="primary"],
    .st-key-login-actions a[kind="primary"] {
        border: 0;
        background: linear-gradient(120deg, var(--apple-red), var(--apple-red-dark));
    }

    .privacy-note {
        text-align: center;
        padding: 14px 20px 34px;
        color: #8a958e;
        font-size: .72rem;
    }

    .st-key-destination-card {
        max-width: 650px;
        margin: 8vh auto 0;
        padding: 48px;
        text-align: center;
        background: rgba(255,255,255,.88);
        border: 1px solid rgba(63,92,68,.13);
        border-radius: 30px;
        box-shadow: 0 24px 70px rgba(43,70,49,.14);
    }

    .destination-apple {
        font-size: 4rem;
        filter: drop-shadow(0 13px 15px rgba(176,44,35,.18));
        margin-bottom: 8px;
    }

    .st-key-destination-card h1 { color: var(--ink); letter-spacing: -.04em; }

    .st-key-destination-card button { border-radius: 13px; min-height: 2.8rem; }

    @media (max-width: 780px) {
        .block-container {
            min-height: auto;
            display: block;
            padding: 1rem .8rem 2rem;
        }
        .block-container > [data-testid="stVerticalBlock"] {
            justify-content: flex-start;
        }
        .st-key-auth-shell { border-radius: 24px; padding: 10px; }
        .auth-hero { min-height: auto; padding: 25px; }
        .brand-kicker { font-size: .63rem; }
        .hero-title { font-size: 2.05rem; margin: 18px 0 8px; }
        .hero-copy { display: none; }
        .apple-stage { height: 142px; }
        .apple { transform: rotate(-4deg) scale(.72); }
        .scan-ring { transform: scale(.72); animation: none; }
        .auth-form-copy { padding: 35px 22px 10px; }
        .st-key-login-actions { padding: 4px 22px 0; }
        .privacy-note { padding-bottom: 22px; }
        .st-key-destination-card { margin-top: 2vh; padding: 32px 22px; }
    }

    @media (prefers-reduced-motion: reduce) {
        .scan-ring { animation: none; }
        .st-key-login-actions button { transition: none; }
    }
</style>
"""


AUTH_SESSION_KEY = "supabase_auth_session"
AUTH_ERROR_KEY = "supabase_auth_error"
LOGIN_URL_KEY = "supabase_google_login_url"
LOGIN_FLOW_KEY = "supabase_google_flow_id"
DEFAULT_REDIRECT_URL = "http://localhost:8000/"
FLOW_TTL_SECONDS = 10 * 60


class _OAuthFlowCache:
    """Hold short-lived PKCE data while the browser visits Google."""

    def __init__(self) -> None:
        self._flows: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def _prune(self) -> None:
        cutoff = time.time() - FLOW_TTL_SECONDS
        expired = [
            flow_id
            for flow_id, flow in self._flows.items()
            if flow["created_at"] < cutoff
        ]
        for flow_id in expired:
            self._flows.pop(flow_id, None)

    def get(self, flow_id: str, key: str) -> str | None:
        with self._lock:
            self._prune()
            flow = self._flows.get(flow_id)
            if flow is None:
                return None
            return flow["values"].get(key)

    def set(self, flow_id: str, key: str, value: str) -> None:
        with self._lock:
            self._prune()
            flow = self._flows.setdefault(
                flow_id,
                {"created_at": time.time(), "values": {}},
            )
            flow["values"][key] = value

    def remove(self, flow_id: str, key: str) -> None:
        with self._lock:
            flow = self._flows.get(flow_id)
            if flow is not None:
                flow["values"].pop(key, None)

    def discard(self, flow_id: str) -> None:
        with self._lock:
            self._flows.pop(flow_id, None)


class _OAuthFlowStorage:
    """Supabase-compatible storage scoped to one browser login flow."""

    def __init__(self, cache: _OAuthFlowCache, flow_id: str) -> None:
        self._cache = cache
        self._flow_id = flow_id

    def get_item(self, key: str) -> str | None:
        return self._cache.get(self._flow_id, key)

    def set_item(self, key: str, value: str) -> None:
        self._cache.set(self._flow_id, key, value)

    def remove_item(self, key: str) -> None:
        self._cache.remove(self._flow_id, key)


@st.cache_resource(show_spinner=False)
def _oauth_flow_cache() -> _OAuthFlowCache:
    return _OAuthFlowCache()


def _current_app_url() -> str:
    """Return the URL used by this browser session without query parameters."""
    try:
        current_url = str(st.context.url).strip()
    except (AttributeError, RuntimeError):
        current_url = ""

    if not current_url:
        return DEFAULT_REDIRECT_URL
    return f"{current_url.rstrip('/')}/"


def _supabase_settings() -> tuple[str, str, str]:
    """Return the Project URL, publishable key, and browser callback URL."""
    config = st.secrets.get("supabase")
    if config is None:
        raise RuntimeError(
            "Missing [supabase] configuration in .streamlit/secrets.toml."
        )

    project_url = str(config.get("url", "")).strip().rstrip("/")
    publishable_key = str(config.get("key", "")).strip()
    configured_redirect = str(config.get("redirect_url", "")).strip()
    redirect_url = configured_redirect or _current_app_url()

    if not project_url or not publishable_key:
        raise RuntimeError(
            "Supabase Project URL and publishable key are required."
        )
    if not project_url.endswith(".supabase.co"):
        raise RuntimeError(
            "The Supabase Project URL must end with '.supabase.co'."
        )
    if not redirect_url:
        raise RuntimeError("The Supabase redirect URL cannot be empty.")

    return project_url, publishable_key, redirect_url


def _create_supabase_client(*, storage: Any | None = None) -> Client:
    project_url, publishable_key, _ = _supabase_settings()
    options = SyncClientOptions(
        flow_type="pkce",
        storage=storage,
        persist_session=storage is None,
        auto_refresh_token=False,
    )
    return create_client(project_url, publishable_key, options)


def _callback_url(flow_id: str) -> str:
    _, _, redirect_url = _supabase_settings()
    separator = "&" if "?" in redirect_url else "?"
    return f"{redirect_url}{separator}{urlencode({'auth_flow': flow_id})}"


def _new_google_login_url() -> tuple[str, str]:
    flow_id = secrets.token_urlsafe(24)
    storage = _OAuthFlowStorage(_oauth_flow_cache(), flow_id)
    client = _create_supabase_client(storage=storage)
    response = client.auth.sign_in_with_oauth(
        {
            "provider": "google",
            "options": {"redirect_to": _callback_url(flow_id)},
        }
    )
    return response.url, flow_id


def _google_login_url() -> str:
    login_url = st.session_state.get(LOGIN_URL_KEY)
    if login_url:
        return str(login_url)

    login_url, flow_id = _new_google_login_url()
    st.session_state[LOGIN_URL_KEY] = login_url
    st.session_state[LOGIN_FLOW_KEY] = flow_id
    return login_url


def _store_session(session: Any) -> None:
    user = session.user
    metadata = user.user_metadata or {}
    display_name = (
        metadata.get("full_name")
        or metadata.get("name")
        or user.email
        or "Signed-in user"
    )
    st.session_state[AUTH_SESSION_KEY] = {
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
        "expires_at": session.expires_at,
        "user": {
            "id": user.id,
            "email": user.email,
            "name": display_name,
        },
    }


def _clear_callback_parameters() -> None:
    for parameter in (
        "code",
        "auth_flow",
        "error",
        "error_code",
        "error_description",
    ):
        if parameter in st.query_params:
            del st.query_params[parameter]


def _handle_oauth_callback() -> None:
    error_description = st.query_params.get("error_description")
    if error_description:
        st.session_state[AUTH_ERROR_KEY] = str(error_description)
        _clear_callback_parameters()
        return

    auth_code = st.query_params.get("code")
    if not auth_code:
        return

    flow_id = st.query_params.get("auth_flow")
    if not flow_id:
        st.session_state[AUTH_ERROR_KEY] = (
            "The login callback is missing its flow identifier. Confirm that "
            "this app's exact URL is listed under Supabase Authentication → "
            "URL Configuration → Redirect URLs, then try again."
        )
        _clear_callback_parameters()
        return

    cache = _oauth_flow_cache()
    storage = _OAuthFlowStorage(cache, str(flow_id))
    client = _create_supabase_client(storage=storage)

    try:
        response = client.auth.exchange_code_for_session(
            {"auth_code": str(auth_code)}
        )
        if response.session is None:
            raise RuntimeError("Supabase did not return a user session.")
        _store_session(response.session)
        st.session_state.guest_mode = False
        st.session_state.pop(AUTH_ERROR_KEY, None)
    except Exception as exc:
        st.session_state[AUTH_ERROR_KEY] = (
            "Google sign-in could not be completed. "
            f"Technical detail: {exc}"
        )
    finally:
        cache.discard(str(flow_id))
        st.session_state.pop(LOGIN_URL_KEY, None)
        st.session_state.pop(LOGIN_FLOW_KEY, None)
        _clear_callback_parameters()

    if AUTH_SESSION_KEY in st.session_state:
        st.rerun()


def _refresh_session_if_needed() -> None:
    auth_session = st.session_state.get(AUTH_SESSION_KEY)
    if not auth_session:
        return

    expires_at = auth_session.get("expires_at")
    if expires_at and int(expires_at) > int(time.time()) + 60:
        return

    try:
        client = _create_supabase_client()
        response = client.auth.refresh_session(auth_session["refresh_token"])
        if response.session is None:
            raise RuntimeError("Supabase could not refresh the login session.")
        _store_session(response.session)
    except Exception:
        st.session_state.pop(AUTH_SESSION_KEY, None)
        st.session_state[AUTH_ERROR_KEY] = (
            "Your login session expired. Please sign in again."
        )


def is_logged_in() -> bool:
    _refresh_session_if_needed()
    return AUTH_SESSION_KEY in st.session_state


def current_user() -> dict[str, str] | None:
    if not is_logged_in():
        return None
    return st.session_state[AUTH_SESSION_KEY]["user"]


def current_user_name() -> str:
    user = current_user()
    return user["name"] if user else "Signed-in user"


def current_user_id() -> str | None:
    user = current_user()
    return user["id"] if user else None


def authenticated_supabase_client() -> Client:
    """Return a client authorized as the current user for database/storage calls."""
    if not is_logged_in():
        raise RuntimeError("A signed-in user is required.")

    auth_session = st.session_state[AUTH_SESSION_KEY]
    client = _create_supabase_client()
    response = client.auth.set_session(
        auth_session["access_token"],
        auth_session["refresh_token"],
    )
    if response.session is not None:
        _store_session(response.session)
    return client


def logout() -> None:
    auth_session = st.session_state.get(AUTH_SESSION_KEY)
    if auth_session:
        try:
            client = _create_supabase_client()
            client.auth.set_session(
                auth_session["access_token"],
                auth_session["refresh_token"],
            )
            client.auth.sign_out({"scope": "local"})
        except Exception:
            pass

    st.session_state.pop(AUTH_SESSION_KEY, None)
    st.session_state.pop(LOGIN_URL_KEY, None)
    st.session_state.pop(LOGIN_FLOW_KEY, None)
    st.session_state.guest_mode = False


def _continue_as_guest() -> None:
    st.session_state.guest_mode = True


def _return_to_login() -> None:
    st.session_state.guest_mode = False


def require_login_or_guest() -> None:
    """Block the current page until Google login or guest mode is selected."""
    _handle_oauth_callback()

    if "guest_mode" not in st.session_state:
        st.session_state.guest_mode = False

    if is_logged_in() or st.session_state.guest_mode:
        return

    st.html(AUTH_STYLES)
    _show_login_page()
    st.stop()


def _show_login_page() -> None:
    with st.container(key="auth-shell"):
        hero_column, form_column = st.columns([1.08, 0.92], gap="small")

        with hero_column:
            st.html(
                """
                <section class="auth-hero">
                    <div class="brand-kicker"><span>●</span> Smart harvest vision</div>
                    <h1 class="hero-title">Know your apple.<br>Pick the perfect moment.</h1>
                    <p class="hero-copy">
                        Computer vision turns every apple image into a clear,
                        confident ripeness assessment—from early growth to overripe.
                    </p>
                    <div class="apple-stage" aria-label="Illustrated ripe apple">
                        <div class="scan-ring"></div>
                        <div class="apple"></div>
                    </div>
                    <div class="ripeness-scale" aria-label="Ripeness colour scale">
                        <span style="--stage-color:#91bd62"></span>
                        <span style="--stage-color:#bad05b"></span>
                        <span style="--stage-color:#e2c64d"></span>
                        <span style="--stage-color:#ed9c42"></span>
                        <span style="--stage-color:#e75b3f"></span>
                        <span style="--stage-color:#923c36"></span>
                    </div>
                    <div class="scale-labels"><span>20% ripe</span><span>Perfectly ripe</span><span>Overripe</span></div>
                </section>
                """
            )

        with form_column:
            st.html(
                """
                <section class="auth-form-copy">
                    <div class="auth-eyebrow">Apple Ripeness Assessment</div>
                    <div class="auth-title">Welcome to a smarter harvest.</div>
                    <p class="auth-subtitle">
                        Sign in to build your personal detection history, or explore
                        the classifier instantly as a guest.
                    </p>
                    <div class="feature-row">
                        <span class="feature-icon">✓</span>
                        <span><b>Private history</b><br>Return to every saved assessment.</span>
                    </div>
                    <div class="feature-row">
                        <span class="feature-icon">◎</span>
                        <span><b>Fast detection</b><br>Upload, capture, and classify with ease.</span>
                    </div>
                    <div class="feature-row">
                        <span class="feature-icon">↗</span>
                        <span><b>Guest friendly</b><br>No account is required to try the system.</span>
                    </div>
                </section>
                """
            )

            with st.container(key="login-actions"):
                auth_error = st.session_state.pop(AUTH_ERROR_KEY, None)
                if auth_error:
                    st.error(auth_error)

                try:
                    login_url = _google_login_url()
                except Exception as exc:
                    st.error("Supabase login is not configured correctly.")
                    st.caption(str(exc))
                else:
                    st.link_button(
                        "Continue with Google  →",
                        login_url,
                        type="primary",
                        use_container_width=True,
                    )
                st.button(
                    "Explore as guest",
                    use_container_width=True,
                    on_click=_continue_as_guest,
                )

            st.html(
                '<div class="privacy-note">Secure sign-in powered by Google and Supabase · '
                "Guest activity is not saved</div>"
            )


def _show_destination_page() -> None:
    with st.container(key="destination-card"):
        st.html('<div class="destination-apple">🍎</div>')
        st.title("You’re ready to assess")

        if is_logged_in():
            st.success(f"Signed in as {current_user_name()}")
            st.caption(
                "Your future classification history will be securely linked "
                "to this account."
            )
            st.button("Log out", use_container_width=True, on_click=logout)
        else:
            st.info("You’re exploring as a guest. Classification history will not be saved.")
            st.button(
                "Return to sign in",
                use_container_width=True,
                on_click=_return_to_login,
            )

        st.divider()
        st.subheader("Classification workspace coming next")
        st.caption("This temporary destination is ready for your classifier interface.")


def show_auth_entry() -> None:
    """Show login or the temporary destination page, then stop the app run.

    Remove the final ``st.stop()`` when the classification interface is ready
    to appear after authentication.
    """
    st.html(AUTH_STYLES)
    _handle_oauth_callback()

    if "guest_mode" not in st.session_state:
        st.session_state.guest_mode = False

    if not is_logged_in() and not st.session_state.guest_mode:
        _show_login_page()
        st.stop()

    _show_destination_page()
    st.stop()
