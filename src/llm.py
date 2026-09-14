"""
src/llm.py
----------
Handles the Large Language Model connection.

One client, many providers
    Groq, the Hugging Face "Inference Providers" router and a local Ollama
    server all speak the OpenAI chat-completions protocol
    (`POST /v1/chat/completions`). So instead of one class per vendor we use
    the official `openai` SDK with a different `base_url` + API key per
    provider. The `PROVIDERS` registry below holds those presets; the CLI's
    --provider flag and /provider command switch between them.

    To add another OpenAI-compatible provider (OpenAI itself, Gemini,
    Together, ...), add ONE entry to PROVIDERS -- nothing else changes:

        "openai": ProviderSpec(
            key="openai", label="OpenAI",
            base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            default_model="gpt-4o-mini", fallback_models=("gpt-4o-mini", "gpt-4o"),
            signup_url="https://platform.openai.com/api-keys",
        ),

    A provider marked `local=True` (Ollama) needs no key: it is "ready" when
    its server answers, and the SDK is given a placeholder key.

Public API
    get_llm(provider, model)  -> BaseLLM with .generate(system, user) -> str
    list_models(provider)     -> model ids for `main.py --list-models` (live, with fallback)
    provider_status()         -> which providers have an API key configured
    openai_client(provider)   -> raw openai.OpenAI client (used by RAGAS)

Secrets are read from src/config.py (which loads .env) and never logged.
"""

import os
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

from openai import APIStatusError, AsyncOpenAI, OpenAI, RateLimitError

from src import config


class MissingAPIKeyError(RuntimeError):
    """Raised when a provider is selected but its API key is not configured."""


# ---------------------------------------------------------------------- #
# Provider registry
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProviderSpec:
    key: str                       # registry name, also the value of LLM_PROVIDER in .env
    label: str                     # human-readable name for console output
    base_url: str                  # OpenAI-compatible endpoint
    api_key_env: str               # name of the .env variable holding the key
    api_key: str                   # the key's value (may be empty)
    default_model: str
    fallback_models: tuple[str, ...]   # shown when the live model list is unavailable
    signup_url: str                # where to get a key
    live_model_list: bool = True   # whether GET /models returns a useful list
    # Model ids containing any of these fragments are hidden from the dropdown
    # (audio, moderation and embedding models cannot answer questions).
    exclude_fragments: tuple[str, ...] = ("whisper", "tts", "guard", "embed", "moderation", "orpheus")
    local: bool = False            # runs on this machine: no key, "ready" = server reachable
    hint: str = ""                 # shown when the provider is not ready


PROVIDERS: dict[str, ProviderSpec] = {
    "groq": ProviderSpec(
        key="groq",
        label="Groq (free plan)",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        api_key=config.GROQ_API_KEY,
        default_model=config.GROQ_MODEL,
        # The chat models included in Groq's FREE plan (console.groq.com/docs/rate-limits,
        # Sept 2026): 30 requests/min, 1K requests/day, 8K tokens/min, 200K tokens/day.
        # They are listed first in the dropdown; other ids returned by the live
        # model list (Enterprise-only Llama models etc.) follow.
        fallback_models=(
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b",
            "qwen/qwen3.8-27b",
        ),
        signup_url="https://console.groq.com/keys",
        # "compound" = Groq's agentic system with built-in web search; it would
        # answer from the internet instead of our documents, so it is hidden.
        exclude_fragments=("whisper", "tts", "guard", "embed", "moderation", "orpheus", "compound"),
    ),
    "huggingface": ProviderSpec(
        key="huggingface",
        label="Hugging Face (Inference Providers)",
        base_url="https://router.huggingface.co/v1",
        api_key_env="HF_TOKEN",
        api_key=config.HF_TOKEN,
        default_model=config.HF_MODEL,
        # A curated handful that is listed first; the live list from the
        # router (140+ chat models, Sept 2026) follows it in /models.
        fallback_models=(
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen3-32B",
            "Qwen/Qwen2.5-7B-Instruct",
            "openai/gpt-oss-120b",
        ),
        signup_url="https://huggingface.co/settings/tokens",
        # Hide safety classifiers and non-chat models from the live list.
        exclude_fragments=("guard", "safeguard", "embed", "rerank", "whisper", "tts"),
    ),
    "ollama": ProviderSpec(
        key="ollama",
        label="Ollama (local, no key)",
        base_url=config.OLLAMA_BASE_URL,
        api_key_env="OLLAMA_API_KEY",   # only read if set; Ollama itself ignores keys
        api_key="",
        default_model=config.OLLAMA_MODEL,
        # What `ollama pull` users typically have; the live list shows what is installed.
        fallback_models=("llama3.2", "llama3.1", "qwen2.5", "mistral", "gemma2"),
        signup_url="https://ollama.com/download",
        exclude_fragments=("embed", "nomic", "bge", "minilm", "whisper"),
        local=True,
        hint="run `ollama serve` (then `ollama pull llama3.2`)",
    ),
}


def get_provider(name: str | None = None) -> ProviderSpec:
    """Look up a provider spec, defaulting to LLM_PROVIDER from .env."""
    name = (name or config.LLM_PROVIDER).lower()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider '{name}'. Available: {', '.join(PROVIDERS)}")
    return PROVIDERS[name]


def _resolve_api_key(spec: ProviderSpec) -> str:
    """Prefer a key set in the process environment over the .env value.

    Local servers (Ollama) ignore the key, but the openai SDK refuses to start
    without one, so they get a placeholder."""
    key = os.getenv(spec.api_key_env, "").strip() or spec.api_key
    return key or ("local" if spec.local else "")


def _server_reachable(base_url: str, timeout: float = 1.5) -> bool:
    """True if an OpenAI-compatible server answers GET /models at base_url."""
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def provider_status() -> dict[str, dict]:
    """{provider: {label, has_key, local, hint, ...}} for the CLI status lines.

    `has_key` means "ready to use": a key is present -- or, for a local
    provider, its server answers (checked with a short timeout)."""
    return {
        key: {
            "label": spec.label,
            "has_key": _server_reachable(spec.base_url) if spec.local else bool(_resolve_api_key(spec)),
            "env_var": spec.api_key_env,
            "signup_url": spec.signup_url,
            "default_model": spec.default_model,
            "local": spec.local,
            "hint": spec.hint,
            "base_url": spec.base_url,
        }
        for key, spec in PROVIDERS.items()
    }


def openai_client(provider: str | None = None, timeout: float = 60.0, max_retries: int = 3,
                  async_client: bool = False) -> "OpenAI | AsyncOpenAI":
    """Build an `openai.OpenAI` (or `AsyncOpenAI`) client for the provider's endpoint.

    Also used by evaluation/ragas_evaluation.py, because RAGAS's llm_factory
    accepts any OpenAI client -- which is exactly what makes Groq usable as
    the RAGAS judge for free. RAGAS's metrics are asynchronous internally, so
    they need the async flavour (`async_client=True`).
    """
    spec = get_provider(provider)
    api_key = _resolve_api_key(spec)
    if not api_key or api_key.startswith(("gsk_your", "hf_your")):
        raise MissingAPIKeyError(
            f"No API key for {spec.label}. Add {spec.api_key_env}=... to your .env file "
            f"(get one at {spec.signup_url})."
        )
    client_cls = AsyncOpenAI if async_client else OpenAI
    return client_cls(base_url=spec.base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)


@lru_cache(maxsize=8)
def list_models(provider: str | None = None) -> list[str]:
    """Model ids for `--list-models` / `/models`: live list from the API when possible.

    Cached per provider so repeated /models calls do not hit the network
    every time. Falls back to the static list on any error (no key, offline,
    provider does not support GET /models).
    """
    spec = get_provider(provider)
    models: list[str] = []
    if spec.live_model_list:
        try:
            response = openai_client(spec.key).models.list()
            models = sorted(
                m.id for m in response.data
                if not any(frag in m.id.lower() for frag in spec.exclude_fragments)
            )
        except Exception:
            models = []  # fall through to the static list
    if not models:
        models = list(spec.fallback_models)
    # Order: configured default, then the curated (free-plan) models, then
    # everything else the API reported.
    curated = [m for m in spec.fallback_models if m in models]
    rest = [m for m in models if m not in curated]
    ordered = curated + rest
    if spec.default_model in ordered:
        ordered.remove(spec.default_model)
    return [spec.default_model] + ordered


# ---------------------------------------------------------------------- #
# LLM classes
# ---------------------------------------------------------------------- #
class BaseLLM(ABC):
    """Minimal interface the rest of the project relies on."""

    provider: str
    model: str

    @abstractmethod
    def chat(self, messages: list[dict], temperature: float | None = None,
             max_tokens: int | None = None, json_mode: bool = False) -> str:
        """Send a full message list [{'role','content'}, ...] and return the reply text."""

    def generate(self, system: str, user: str, temperature: float | None = None,
                 max_tokens: int | None = None, json_mode: bool = False) -> str:
        """Convenience wrapper: one system prompt + one user prompt -> reply text."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)


def reasoning_kwargs(provider: str, model: str) -> dict:
    """Extra request fields that keep "reasoning" models cheap and concise.

    Groq's free-plan models (gpt-oss, qwen3) think before answering; those
    reasoning tokens count against the 8K tokens/min budget and are not
    needed for grounded Q&A over short passages. We therefore ask for the
    lowest effort and keep the reasoning out of the reply
    (see https://console.groq.com/docs/reasoning). Other providers ignore
    unknown fields or do not receive them at all.
    """
    if provider != "groq":
        return {}
    name = model.lower()
    if "gpt-oss" in name:
        return {"reasoning_effort": "low", "include_reasoning": False}
    if "qwen3" in name:
        return {"reasoning_effort": "none", "include_reasoning": False}
    return {}


class OpenAICompatibleLLM(BaseLLM):
    """Chat completions through any OpenAI-compatible endpoint (Groq, HF, ...)."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float = config.LLM_TEMPERATURE,
        max_tokens: int = config.LLM_MAX_TOKENS,
    ):
        self.spec = get_provider(provider)
        self.provider = self.spec.key
        self.model = model or self.spec.default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = openai_client(self.spec.key)  # raises MissingAPIKeyError early

    def chat(self, messages: list[dict], temperature: float | None = None,
             max_tokens: int | None = None, json_mode: bool = False) -> str:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        extra = reasoning_kwargs(self.provider, self.model)
        if extra:
            kwargs["extra_body"] = extra  # provider-specific fields the SDK passes through
        if json_mode:
            # Ask the provider to guarantee a JSON object reply (used by the
            # manual evaluation judge prompts). Not every model supports it,
            # so we retry without the flag if the API rejects it.
            kwargs["response_format"] = {"type": "json_object"}

        # The SDK already retries transient failures; this outer loop adds a
        # longer wait for free-tier rate limits (HTTP 429) so evaluation runs
        # survive bursts of requests instead of crashing.
        for attempt in range(4):
            try:
                response = self._client.chat.completions.create(**kwargs)
                return (response.choices[0].message.content or "").strip()
            except RateLimitError:
                wait = 5 * (attempt + 1)
                print(f"[llm] rate limited by {self.provider}; waiting {wait}s...")
                time.sleep(wait)
            except APIStatusError as exc:
                if json_mode and exc.status_code in (400, 422):
                    kwargs.pop("response_format", None)  # model lacks JSON mode
                    json_mode = False
                    continue
                raise
        raise RuntimeError(f"{self.provider} kept rate-limiting after several retries.")

    def __repr__(self) -> str:
        return f"OpenAICompatibleLLM(provider={self.provider!r}, model={self.model!r})"


def get_llm(provider: str | None = None, model: str | None = None, **kwargs) -> BaseLLM:
    """Factory used everywhere else in the project."""
    return OpenAICompatibleLLM(provider=provider, model=model, **kwargs)
