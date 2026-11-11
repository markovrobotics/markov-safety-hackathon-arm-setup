"""Run the agent policy on a logged-in Claude Code or Codex plan instead of an API key.

Serves an OpenAI-style /v1/chat/completions on localhost and answers each turn by
running `claude -p` or `codex exec` on this machine's login. The model picks the CLI:

    uv run llm_bridge.py &
    uv run inspect-robots run --instruction "pick up the cup" --embodiment nero --policy agent \
        -P model=claude/opus -P base_url=http://127.0.0.1:8765/v1      # or model=codex/ (the login's default model)

Each turn is one fresh CLI call carrying the whole conversation; camera frames are
written to files the CLI looks at. Expect a few seconds per turn.
"""

import base64
import hashlib
import json
import subprocess
import sys
import tempfile
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
WORK = Path(tempfile.mkdtemp(prefix="llm-bridge-"))
REPLY = {
    "type": "object",
    "properties": {
        "tool": {"type": "string"},
        "arguments_json": {"type": "string"},
        "text": {"type": "string"},
    },
    "required": ["tool", "arguments_json", "text"],
    "additionalProperties": False,
}


def image_file(url: str) -> Path:
    data = base64.b64decode(url.split(",", 1)[1])
    path = WORK / f"{hashlib.sha1(data).hexdigest()[:12]}.png"
    path.write_bytes(data)
    return path


def render(body: dict) -> tuple[str, list[Path]]:
    """The conversation as one prompt, plus the images of its newest message."""
    lines, images = [], []
    for m in body["messages"]:
        content = m.get("content")
        parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        text, imgs = [], []
        for p in parts:
            if p.get("type") == "text":
                text.append(p["text"])
            elif p.get("type") == "image_url":
                imgs.append(image_file(p["image_url"]["url"]))
        role = m["role"].upper()
        if m["role"] == "tool":
            role = f"TOOL RESULT ({m.get('tool_call_id')})"
        lines.append(f"### {role}\n" + "\n".join(text))
        lines += [f"[image: {i}]" for i in imgs]
        for c in m.get("tool_calls") or []:
            lines.append(f"[you called {c['function']['name']} {c['function']['arguments']}]")
        if imgs:
            images = imgs
    tools = [t["function"] for t in body.get("tools", [])]
    lines.append(
        "### HOW TO ANSWER\nYou are the policy above. Answer with exactly one call of one of these tools, "
        "as `tool` (its name) and `arguments_json` (its arguments as a JSON object string); `text` is "
        "anything you want to say (may be empty).\n" + json.dumps(tools)
    )
    return "\n\n".join(lines), images


def ask(model: str, prompt: str, images: list[Path]) -> dict:
    cli, _, name = model.partition("/")
    if cli == "claude":
        if images:
            prompt += "\n\nThe newest camera frames are these files; look at them with Read first: " + ", ".join(map(str, images))
        cmd = ["claude", "-p", "--output-format", "json", "--json-schema", json.dumps(REPLY),
               "--allowedTools", "Read", "--no-session-persistence"] + (["--model", name] if name else [])
        out = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=WORK, check=True)
        return json.loads(out.stdout)["structured_output"]
    if cli == "codex":
        schema, last = WORK / "reply.schema.json", WORK / f"{uuid.uuid4().hex}.json"
        schema.write_text(json.dumps(REPLY))
        cmd = ["codex", "exec", "--skip-git-repo-check", "-s", "read-only", "--output-schema", str(schema),
               "-o", str(last)] + (["-m", name] if name else [])
        for i in images:
            cmd += ["-i", str(i)]
        subprocess.run(cmd + ["-"], input=prompt, capture_output=True, text=True, cwd=WORK, check=True)
        return json.loads(last.read_text())
    raise ValueError(f"model must be claude/<model> or codex/<model>, got {model!r}")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            reply = ask(body["model"], *render(body))
            call = {"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                    "function": {"name": reply["tool"], "arguments": reply["arguments_json"] or "{}"}}
            msg = {"role": "assistant", "content": reply["text"] or None, "tool_calls": [call]}
            out, code = {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": msg}]}, 200
        except subprocess.CalledProcessError as e:
            out, code = {"error": {"message": (e.stderr or e.stdout)[-2000:]}}, 502
        except Exception as e:  # report to the agent rather than drop the connection
            out, code = {"error": {"message": repr(e)}}, 500
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        print(code, body["model"], out["choices"][0]["message"]["tool_calls"][0]["function"] if code == 200 else out, flush=True)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"llm bridge on http://127.0.0.1:{PORT}/v1 (scratch {WORK})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
