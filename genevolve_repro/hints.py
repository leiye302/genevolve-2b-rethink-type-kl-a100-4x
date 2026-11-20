"""One bounded request per trajectory with an Actor-routed native turn."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
import json
import os
import re
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

from .common import ROOT, read_json, required_env

MAX_HINT_CHARS = 400
MAX_BODY = 2 * 1024**2
MAX_TEXT_PER_MESSAGE = 3500
URL = re.compile(r"https?://[^\s<>\x22]+", re.IGNORECASE)
CREDENTIAL = re.compile(r"(?i)\b(?:oc_sk_|sk-|Bearer\s+)[A-Za-z0-9_+/.=-]{10,}")
LOCAL_PATH = re.compile(r"(?im)^\s*local_path:.*$")


@dataclass(frozen=True)
class HintSettings:
    endpoint: str
    model: str
    key: str = field(repr=False)

    @property
    def opencode_go(self):
        url = urlsplit(self.endpoint)
        return url.hostname == "opencode.ai" and url.path.startswith(("/zen/go/", "/inference/go/"))

    def identity(self):
        return {"endpoint": self.endpoint, "model": self.model,
                "client_session": "one-fresh-id-per-trajectory" if self.opencode_go else None}


def settings_from_env():
    endpoint = required_env("GENEVOLVE_HINT_API_URL")
    url = urlsplit(endpoint)
    if (url.scheme != "https" or not url.hostname or url.username is not None
            or url.password is not None or url.query or url.fragment
            or url.port not in (None, 443) or not url.path.endswith("/chat/completions")):
        raise ValueError("GENEVOLVE_HINT_API_URL must be a clean HTTPS /chat/completions endpoint.")
    r = read_json(ROOT / "configs" / "rethink.json")
    return HintSettings(endpoint, r["hint_model"], required_env("GENEVOLVE_HINT_API_KEY"))


def redact(text):
    text = CREDENTIAL.sub("[credential omitted]", str(text))
    text = URL.sub("[URL omitted]", text)
    text = LOCAL_PATH.sub("  local_path: [omitted]", text)
    if len(text) > MAX_TEXT_PER_MESSAGE:
        text = text[:2400] + "\n[excerpt omitted]\n" + text[-1000:]
    return text


@lru_cache(maxsize=1)
def system_prompt():
    return (ROOT / "configs" / "local_hint_system.txt").read_text(encoding="utf-8").strip()


def build_payload(trajectory, selected):
    # Native tool responses are user messages. Do not send paths, raw image
    # references, GT, image-Judge material, or other samples in the group.
    messages = trajectory.result.messages
    timeline, n = [], 0
    for message in messages[1:]:
        if message["role"] == "assistant":
            n += 1
        timeline.append({"role": message["role"], "turn": n, "text": redact(message["content"])})
    return {"selected_turns": selected, "termination": trajectory.result.termination, "timeline": timeline}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def post(settings, payload, timeout, session):
    headers = {"Authorization": "Bearer " + settings.key, "Content-Type": "application/json",
               "Accept": "application/json", "User-Agent": "genevolve-rethink-hints/1.0"}
    if settings.opencode_go and session:
        headers["x-opencode-session"] = session
    body = json.dumps(payload, ensure_ascii=False).encode()
    if len(body) > MAX_BODY:
        raise ValueError("oversized request")
    request = Request(settings.endpoint, data=body, headers=headers, method="POST")
    with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
        body = response.read(MAX_BODY + 1)
        if len(body) > MAX_BODY:
            raise ValueError("oversized response")
        return json.loads(body)


def parse_reply(reply, selected):
    choice = reply["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("incomplete response")
    content = choice["message"]["content"].strip()
    if content.startswith("```") and content.endswith("```"):
        content = content.split("\n", 1)[1][:-3].strip()

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    hints = json.loads(content, object_pairs_hook=unique)["hints"]
    if not isinstance(hints, dict):
        raise ValueError("expected a hints object")
    out = {}
    for n in selected:
        value = hints.get(str(n))
        if isinstance(value, str):
            value = value.strip()
            if (0 < len(value) <= MAX_HINT_CHARS and not any(c in value for c in "<>{}\r\n")
                    and not URL.search(value) and not CREDENTIAL.search(value)):
                out[n] = value
    return out


def request_one(settings, cfg, trajectory, selected, deadline):
    payload = {"model": settings.model, "temperature": 0.0, "max_tokens": cfg["hint_request_tokens"],
               "messages": [{"role": "system", "content": system_prompt()},
                            {"role": "user", "content": json.dumps(build_payload(trajectory, selected), ensure_ascii=False)}]}
    payload["reasoning_effort"] = cfg["hint_reasoning_effort"]
    session = uuid4().hex if settings.opencode_go else None
    count, taken = 0, {}
    for attempt in range(cfg["hint_attempts"]):
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        try:
            count += 1
            taken.update(parse_reply(post(settings, payload, min(remaining, cfg["hint_timeout"]), session), selected))
            if taken:
                break
        except HTTPError as error:
            error.close()
            if error.code not in (408, 429) and error.code < 500:
                break
        except (TimeoutError, URLError, OSError, ValueError, AttributeError, IndexError, KeyError, TypeError):
            pass
        if attempt + 1 < cfg["hint_attempts"]:
            sleep(max(0, min(0.5, deadline - monotonic())))
    return taken, count


def request_trajectories(settings, cfg, jobs):
    if not jobs:
        return []
    deadline = monotonic() + cfg["hint_update_budget_s"]
    with ThreadPoolExecutor(max_workers=min(len(jobs), cfg["hint_parallel_per_rank"])) as pool:
        return list(pool.map(lambda item: request_one(settings, cfg, *item, deadline), jobs))
