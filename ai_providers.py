"""
ai_providers.py
-----------------
Provider-agnostic abstraction for calling a language model to generate exam
questions. Supports:

  - "Anthropic (Cloud API)" — via the official `anthropic` Python SDK.
  - "Ollama (Local)"        — via Ollama's local REST API (default
                               http://localhost:11434).
  - "Open WebUI (Local)"    — via Open WebUI's OpenAI-compatible REST API
                               (default http://localhost:3000).

Keeping this logic separate from db_utils.py preserves a clean separation of
concerns: this module only knows how to talk to model providers, and returns
plain text/parsed data — it never touches SQLite directly.
"""

import os
import requests

try:
    import anthropic
    _ANTHROPIC_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully in the UI
    _ANTHROPIC_SDK_AVAILABLE = False

PROVIDER_ANTHROPIC = "Anthropic (Cloud API)"
PROVIDER_OLLAMA = "Ollama (Local)"
PROVIDER_OPENWEBUI = "Open WebUI (Local)"

PROVIDERS = [PROVIDER_ANTHROPIC, PROVIDER_OLLAMA, PROVIDER_OPENWEBUI]

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OPENWEBUI_BASE_URL = "http://localhost:3000"

# Curated fallback list used only if the Anthropic /models endpoint can't be reached.
_ANTHROPIC_FALLBACK_MODELS = [
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-haiku-4-5-20251001",
    "claude-fable-5",
]

_REQUEST_TIMEOUT_LIST = 8       # seconds, for listing models
_REQUEST_TIMEOUT_GENERATE = 180  # seconds, for a generation call (local models can be slow)


def provider_requires_api_key(provider: str) -> bool:
    return provider in (PROVIDER_ANTHROPIC, PROVIDER_OPENWEBUI)


def provider_requires_base_url(provider: str) -> bool:
    return provider in (PROVIDER_OLLAMA, PROVIDER_OPENWEBUI)


def default_base_url(provider: str) -> str:
    if provider == PROVIDER_OLLAMA:
        return DEFAULT_OLLAMA_BASE_URL
    if provider == PROVIDER_OPENWEBUI:
        return DEFAULT_OPENWEBUI_BASE_URL
    return ""


def is_provider_configured(provider: str, base_url: str = "", api_key: str = "") -> bool:
    """Quick, non-network check that the minimum required config is present."""
    if provider == PROVIDER_ANTHROPIC:
        if not _ANTHROPIC_SDK_AVAILABLE:
            return False
        return bool(api_key or os.environ.get("ANTHROPIC_API_KEY"))
    if provider == PROVIDER_OLLAMA:
        return bool(base_url)
    if provider == PROVIDER_OPENWEBUI:
        return bool(base_url) and bool(api_key)
    return False


# --------------------------------------------------------------------------
# Model listing
# --------------------------------------------------------------------------
def list_models(provider: str, base_url: str = "", api_key: str = "", custom_path: str = "") -> list[str]:
    """
    Return the list of model names/ids available from the given provider.
    Raises RuntimeError with a user-friendly message on failure.

    `custom_path` (Open WebUI only) overrides auto-detection with an exact endpoint
    path, e.g. "/api/models", for instances with a nonstandard API layout.
    """
    if provider == PROVIDER_ANTHROPIC:
        return _list_anthropic_models(api_key)
    if provider == PROVIDER_OLLAMA:
        return _list_ollama_models(base_url)
    if provider == PROVIDER_OPENWEBUI:
        return _list_openwebui_models(base_url, api_key, custom_path=custom_path)
    raise ValueError(f"Unknown provider: {provider}")


def _list_anthropic_models(api_key: str) -> list[str]:
    if not _ANTHROPIC_SDK_AVAILABLE:
        raise RuntimeError("The 'anthropic' package is not installed. Run: pip install anthropic")
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("No Anthropic API key provided (enter one, or set ANTHROPIC_API_KEY).")
    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.models.list()
        models = [m.id for m in response.data]
        return models if models else _ANTHROPIC_FALLBACK_MODELS
    except Exception:
        # The SDK/API may not support listing on all versions — fall back gracefully.
        return _ANTHROPIC_FALLBACK_MODELS


def _list_ollama_models(base_url: str) -> list[str]:
    base_url = (base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=_REQUEST_TIMEOUT_LIST)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", []) if "name" in m]
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {base_url}. Make sure Ollama is running "
            f"('ollama serve') and reachable at that address. ({exc})"
        )


def _list_openwebui_models(base_url: str, api_key: str, custom_path: str = "") -> list[str]:
    base_url = (base_url or DEFAULT_OPENWEBUI_BASE_URL).rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    paths = [custom_path] if custom_path else ["/api/models", "/api/v1/models", "/openai/v1/models", "/v1/models"]

    attempts = []  # (path, outcome) for diagnostics if every path fails
    for path in paths:
        if not path:
            continue
        try:
            resp = requests.get(f"{base_url}{path}", headers=headers, timeout=_REQUEST_TIMEOUT_LIST)
            if resp.status_code != 200:
                attempts.append(f"{path} -> HTTP {resp.status_code}")
                continue
            data = resp.json()
            items = data.get("data") or data.get("models") or []
            names = []
            for item in items:
                if isinstance(item, dict):
                    names.append(item.get("id") or item.get("name"))
                else:
                    names.append(str(item))
            names = [n for n in names if n]
            if names:
                return names
            attempts.append(f"{path} -> HTTP 200 but no models in response")
        except requests.exceptions.RequestException as exc:
            attempts.append(f"{path} -> {exc.__class__.__name__}: {exc}")
    raise RuntimeError(
        f"Could not list models from Open WebUI at {base_url}. Attempts:\n  "
        + "\n  ".join(attempts)
        + "\n\nTips: verify the base URL/port (the all-in-one Docker image usually serves "
        "the API on the same port as the UI, e.g. 3000; a `pip install open-webui` "
        "install typically serves on 8080). Also confirm an API key is generated under "
        "Open WebUI Settings → Account → API Keys, and that 'Enable API Key' is turned on "
        "under Admin Settings. If your instance uses a nonstandard path, set it under "
        "'Advanced: Custom API Path'."
    )


# --------------------------------------------------------------------------
# Generation call
# --------------------------------------------------------------------------
def call_model(
    provider: str,
    model: str,
    prompt: str,
    base_url: str = "",
    api_key: str = "",
    max_tokens: int = 4096,
    custom_path: str = "",
) -> str:
    """
    Send `prompt` to the selected provider/model and return the raw text response.
    Raises RuntimeError on any provider/network failure.

    `custom_path` (Open WebUI only) overrides auto-detection with an exact endpoint
    path, e.g. "/api/chat/completions", for instances with a nonstandard API layout.
    """
    if provider == PROVIDER_ANTHROPIC:
        return _call_anthropic(model, prompt, api_key, max_tokens)
    if provider == PROVIDER_OLLAMA:
        return _call_ollama(model, prompt, base_url)
    if provider == PROVIDER_OPENWEBUI:
        return _call_openwebui(model, prompt, base_url, api_key, custom_path=custom_path)
    raise ValueError(f"Unknown provider: {provider}")


def _call_anthropic(model: str, prompt: str, api_key: str, max_tokens: int) -> str:
    if not _ANTHROPIC_SDK_AVAILABLE:
        raise RuntimeError("The 'anthropic' package is not installed. Run: pip install anthropic")
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("No Anthropic API key provided (enter one, or set ANTHROPIC_API_KEY).")
    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    except Exception as exc:
        raise RuntimeError(f"Anthropic API call failed: {exc}")


def _call_ollama(model: str, prompt: str, base_url: str) -> str:
    base_url = (base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    try:
        resp = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=_REQUEST_TIMEOUT_GENERATE,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Ollama call failed ({base_url}): {exc}")


def _call_openwebui(model: str, prompt: str, base_url: str, api_key: str, custom_path: str = "") -> str:
    base_url = (base_url or DEFAULT_OPENWEBUI_BASE_URL).rstrip("/")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    paths = [custom_path] if custom_path else [
        "/api/chat/completions",
        "/api/v1/chat/completions",
        "/openai/v1/chat/completions",
        "/v1/chat/completions",
    ]

    attempts = []
    for path in paths:
        if not path:
            continue
        try:
            resp = requests.post(
                f"{base_url}{path}", headers=headers, json=payload, timeout=_REQUEST_TIMEOUT_GENERATE
            )
            if resp.status_code != 200:
                attempts.append(f"{path} -> HTTP {resp.status_code} {resp.reason}")
                continue
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as exc:
            attempts.append(f"{path} -> {exc.__class__.__name__}: {exc}")
        except (KeyError, IndexError, ValueError) as exc:
            attempts.append(f"{path} -> unexpected response shape: {exc}")

    raise RuntimeError(
        f"Open WebUI generation call failed at {base_url}. Attempts:\n  "
        + "\n  ".join(attempts)
        + "\n\nA '405 Method Not Allowed' usually means this port is only serving Open "
        "WebUI's frontend (not its API) — common when the API/backend runs on a "
        "different port (e.g. 8080 for a `pip install open-webui` deployment) than the "
        "one you configured here, or when a reverse proxy in front of it doesn't forward "
        "POST requests to the backend. A '401 Unauthorized' means the API key is missing "
        "or invalid — generate one under Open WebUI Settings → Account → API Keys, and "
        "ensure 'Enable API Key' is on under Admin Settings → General. If you know the "
        "exact working endpoint, set it under 'Advanced: Custom API Path'."
    )
