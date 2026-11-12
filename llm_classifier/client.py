import os, re, time, random
import requests
from dotenv import dotenv_values


def load_env(path):
    return dotenv_values(os.environ.get("YANDEX_ENV_FILE") or path)


def mask(text, secrets):
    for s in secrets:
        if s:
            text = text.replace(s, "***")
    return re.sub(r"Api-Key\s+\S+", "Api-Key ***", text)


class YandexClient:
    def __init__(self, env, call_cfg):
        self.key = env["YANDEX_API_KEY"]
        self.folder = env["YANDEX_FOLDER_ID"]
        self.url = env["YANDEX_OPENAI_BASE_URL"].rstrip("/") + "/chat/completions"
        self.cfg = call_cfg

    def model_uri(self, template):
        return template.format(folder=self.folder)

    def chat(self, uri, system, user, max_tokens):
        """Один вызов с повторами на 429/5xx/сетевых ошибках. Возвращает dict ответа API + _latency."""
        payload = {"model": uri, "temperature": self.cfg["temperature"], "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        headers = {"Authorization": "Api-Key " + self.key}
        last = None
        for attempt in range(self.cfg["http_retries"]):
            t0 = time.time()
            try:
                r = requests.post(self.url, headers=headers, json=payload, timeout=self.cfg["timeout_s"])
                if r.status_code == 200:
                    d = r.json()
                    d["_latency"] = time.time() - t0
                    return d
                last = f"HTTP {r.status_code}: {mask(r.text[:300], [self.key, self.folder])}"
                if r.status_code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(last)
            except requests.RequestException as e:
                last = mask(f"{type(e).__name__}: {e}", [self.key, self.folder])
            time.sleep(min(60, 2 ** attempt + random.random()))
        raise RuntimeError(f"исчерпаны попытки: {last}")
