import json
import re
import time
import threading
from groq import Groq
import groq
from config import GROQ_API_KEYS, PRIMARY_MODEL, FALLBACK_MODEL

if not GROQ_API_KEYS:
    raise RuntimeError("No keys found. Set GROQ_API_KEYS in .env (comma-separated).")


class KeyPool:
    """Round-robin over API keys. A key that hits a rate limit goes on cooldown."""

    def __init__(self, keys):
        self.clients = [Groq(api_key=k) for k in keys]
        self.cooldown_until = [0.0] * len(keys)
        self.idx = 0
        self.lock = threading.Lock()

    def get(self):
        """Return (index, client) of the next available key, or (None, None) if all cooling down."""
        with self.lock:
            now = time.time()
            n = len(self.clients)
            for _ in range(n):
                i = self.idx
                self.idx = (self.idx + 1) % n
                if self.cooldown_until[i] <= now:
                    return i, self.clients[i]
            return None, None

    def cool(self, i, seconds):
        with self.lock:
            self.cooldown_until[i] = time.time() + seconds

    def soonest_available_in(self):
        with self.lock:
            return max(0.0, min(self.cooldown_until) - time.time())


pool = KeyPool(GROQ_API_KEYS)


def _retry_after(err, default=30):
    """Try to read how long Groq wants us to wait."""
    try:
        h = err.response.headers
        if "retry-after" in h:
            return float(h["retry-after"])
    except Exception:
        pass
    m = re.search(r"try again in ([\d.]+)s", str(err))
    if m:
        return float(m.group(1)) + 1
    return default


def ask_llm(system, user, max_tokens=2500, temperature=0.2, model=None, max_attempts=12):
    models = [model] if model else [PRIMARY_MODEL, FALLBACK_MODEL]
    last_err = None

    for m in models:
        for attempt in range(max_attempts):
            i, client = pool.get()
            if client is None:
                # every key is cooling down: wait for the soonest one
                wait = min(pool.soonest_available_in() + 0.5, 60)
                print(f"[llm] all {len(pool.clients)} keys rate-limited, waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = resp.choices[0].message.content or ""
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                if not content:
                    print(f"[llm] empty reply from {m} (likely ran out of tokens), retrying")
                    last_err = RuntimeError("empty response")
                    continue
                return content
            except groq.RateLimitError as e:
                wait = _retry_after(e)
                print(f"[llm] key {i} rate-limited on {m}, cooling {wait:.0f}s")
                pool.cool(i, wait)
                last_err = e
            except (groq.APIConnectionError, groq.APITimeoutError, groq.InternalServerError) as e:
                print(f"[llm] transient error on key {i}: {type(e).__name__}, retrying")
                time.sleep(2 * (attempt + 1))
                last_err = e
            except (groq.BadRequestError, groq.NotFoundError, groq.PermissionDeniedError) as e:
                # wrong/retired model name or no access: rotating keys won't help,
                # so skip to the next model in the list
                print(f"[llm] model {m} unavailable or bad request: {e}")
                last_err = e
                break
        print(f"[llm] giving up on {m}, trying fallback model if any")

    raise RuntimeError(f"All keys and models failed. Last error: {last_err}")


def ask_llm_json(system, user, max_tokens=2500):
    text = ask_llm(
        system + "\nRespond with ONLY valid JSON. No markdown fences, no commentary.",
        user, max_tokens=max_tokens, temperature=0.0,
    ).strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


if __name__ == "__main__":
    print(ask_llm("You are concise.", "Say hello in five words."))