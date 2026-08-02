"""Shared utils: cached multi-provider LLM calls, cost + latency tracking."""
import hashlib, json, os, random, threading, time
from pathlib import Path

import yaml
import litellm
from litellm import completion, completion_cost

# Providers differ in which sampling params they accept (e.g. `seed` is not
# universal). Dropping unsupported params keeps one call signature working
# across all of them instead of special-casing per provider.
litellm.drop_params = True

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

SEED = 42


def _load_dotenv():
    """Read KEY=VALUE lines from cx-llm-bench/.env into the environment.

    litellm reads credentials from environment variables, and a shell export
    only reaches the process that made it. A .env file keeps the keys in one
    place that every run picks up. Real environment variables win, so an
    explicit export can still override the file.
    """
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip().removeprefix("export ").strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv()

# Free-tier providers rate-limit aggressively. Serialize per model and keep a
# minimum gap between that model's calls; different models still run at their
# own pace.
_model_locks: dict[str, threading.Lock] = {}
_last_call: dict[str, float] = {}
_locks_guard = threading.Lock()


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _provider_of(model: str) -> str:
    """Provider name for `model`, e.g. 'anthropic' for a bare claude-* id."""
    try:
        return litellm.get_llm_provider(model)[1]
    except Exception:
        # bare ids without a "/" prefix are OpenAI by litellm convention
        return model.split("/", 1)[0] if "/" in model else "openai"


def _model_delay(model: str, cfg: dict | None) -> float:
    """Seconds to wait between consecutive calls to `model`."""
    if not cfg:
        return 0.0
    delays = cfg.get("rate_limits", {}) or {}
    if model in delays:
        return float(delays[model])
    provider = _provider_of(model)
    if provider in delays:
        return float(delays[provider])
    return float(delays.get("default", 0.0))


def _throttle(model: str, delay: float):
    if delay <= 0:
        return
    now = time.time()
    prev = _last_call.get(model)
    if prev is not None:
        wait = delay - (now - prev)
        if wait > 0:
            time.sleep(wait)
    _last_call[model] = time.time()


def _lock_for(model: str) -> threading.Lock:
    with _locks_guard:
        if model not in _model_locks:
            _model_locks[model] = threading.Lock()
        return _model_locks[model]


def _key(model: str, prompt: str, max_tokens: int) -> Path:
    h = hashlib.sha256(
        f"{model}||{max_tokens}||{prompt}".encode()).hexdigest()[:24]
    return CACHE / f"{h}.json"


def _is_rate_limit(err: Exception) -> bool:
    name = type(err).__name__.lower()
    if "ratelimit" in name:
        return True
    s = str(err).lower()
    return "429" in s or "rate limit" in s or "quota" in s or "too many requests" in s


def call_llm(model: str, prompt: str, max_tokens: int = 512, retries: int = 6,
             cfg: dict | None = None):
    """Cached LLM call.

    Returns dict: text, latency_s, cost_usd, model_snapshot, cached, error.
    On persistent failure returns a record with error set and text "" rather
    than raising, so one bad model cannot abort an unattended multi-model run.
    Failed calls are NOT cached, so a rerun retries them.
    """
    p = _key(model, prompt, max_tokens)
    if p.exists():
        d = json.loads(p.read_text())
        d["cached"] = True
        d.setdefault("error", None)
        return d

    delay = _model_delay(model, cfg)
    last_err = None
    for attempt in range(retries):
        try:
            with _lock_for(model):
                _throttle(model, delay)
                t0 = time.time()
                resp = completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=max_tokens,
                    seed=SEED,
                )
                latency = time.time() - t0
            try:
                cost = completion_cost(completion_response=resp)
            except Exception:
                cost = None  # unknown pricing (e.g. local model)
            d = {
                "text": resp.choices[0].message.content or "",
                "latency_s": round(latency, 3),
                "cost_usd": cost,
                "model_snapshot": getattr(resp, "model", model),
                "cached": False,
                "error": None,
            }
            p.write_text(json.dumps(d))
            return d
        except Exception as e:
            last_err = e
            if attempt == retries - 1:
                break
            # Rate limits need a much longer, jittered backoff than other
            # errors — free tiers meter per minute.
            if _is_rate_limit(e):
                backoff = min(60.0, 5.0 * (2 ** attempt))
            else:
                backoff = min(30.0, 2.0 * (2 ** attempt))
            time.sleep(backoff + random.uniform(0, 1.0))

    return {
        "text": "",
        "latency_s": None,
        "cost_usd": None,
        "model_snapshot": model,
        "cached": False,
        "error": f"{type(last_err).__name__}: {str(last_err)[:300]}",
    }


def append_raw_log(name: str, record: dict):
    raw = ROOT / "results" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    with open(raw / f"{name}.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
