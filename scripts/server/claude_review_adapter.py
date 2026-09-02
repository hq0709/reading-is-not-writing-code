#!/usr/bin/env python3
"""Locked, read-only Claude reviewer MCP transport for Concept Flow."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HOME = Path("/home/qingchan")
UTC = timezone.utc  # noqa: UP017 - local verification still supports Python 3.10.
CLAUDE = HOME / ".local/bin/claude"
REPO = HOME / "work/concept-flow"
STATE = HOME / ".codex/state/claude-review-concept-flow"
MODEL = "claude-fable-5-1"
ARIS_SHA = "94d8093ed21d20a790830318190095b9f5036ce8"
TOOL = {
    "name": "review",
    "description": "Run a pinned, read-only cross-model review of the supplied evidence.",
    "inputSchema": {
        "type": "object",
        "properties": {"prompt": {"type": "string", "minLength": 1}},
        "required": ["prompt"],
        "additionalProperties": False,
    },
}


def git_fingerprint() -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", os.fspath(REPO), *args], text=True, stderr=subprocess.STDOUT
        ).strip()

    return {"head": git("rev-parse", "HEAD"), "status": git("status", "--porcelain=v1")}


def inside_home(path: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(HOME)
    except (FileNotFoundError, ValueError):
        return False
    return True


def canonical_models(payload: Any) -> set[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("modelUsage"), dict):
        return set()
    found = set()
    for usage in payload["modelUsage"].values():
        if isinstance(usage, dict) and isinstance(usage.get("canonicalModel"), str):
            found.add(usage["canonicalModel"])
    return found


def review_argv(prompt: str) -> list[str]:
    return [
        os.fspath(CLAUDE),
        "-p",
        "--name",
        "pinned-review",
        "--no-session-persistence",
        "--model",
        MODEL,
        "--effort",
        "medium",
        "--permission-mode",
        "plan",
        "--tools",
        "",
        "--disallowedTools",
        "mcp__*",
        "--safe-mode",
        "--output-format",
        "json",
        prompt,
    ]


def review_env() -> dict[str, str]:
    return {
        "CLAUDE_REVIEW_MODEL": MODEL,
        "CLAUDE_CODE_EFFORT_LEVEL": "medium",
        "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS": "1",
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    }


def run_review(prompt: str) -> str:
    aris = HOME / "aris_repo"
    if Path.home() != HOME or not all(inside_home(path) for path in (REPO, CLAUDE, aris)):
        raise RuntimeError("reviewer paths are not installed below the authorized home")
    if subprocess.check_output(["git", "-C", os.fspath(aris), "rev-parse", "HEAD"], text=True).strip() != ARIS_SHA:
        raise RuntimeError("review refused: ARIS pin mismatch")
    before = git_fingerprint()
    if before["status"]:
        raise RuntimeError("review refused: checkout is dirty")
    argv = review_argv(prompt)
    completed = subprocess.run(
        argv,
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
        env={**os.environ, **review_env()},
    )
    after = git_fingerprint()
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Claude did not return valid JSON (exit {completed.returncode})") from error
    models = canonical_models(payload)
    valid = completed.returncode == 0 and models == {MODEL} and before == after
    STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    receipt = {
        "utc": datetime.now(UTC).isoformat(),
        "modelExpected": MODEL,
        "modelsObserved": sorted(models),
        "effort": "medium",
        "readOnly": before == after,
        "checkoutBefore": before,
        "checkoutAfter": after,
        "promptSha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "wrapperSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arisFullSha": ARIS_SHA,
        "argvWithoutPrompt": argv[:-1],
        "environmentPins": review_env(),
        "exitCode": completed.returncode,
        "valid": valid,
    }
    receipt_path = STATE / f"review-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    receipt_path.chmod(0o600)
    if not valid:
        raise RuntimeError(f"review identity/read-only gate failed; receipt: {receipt_path}")
    result = payload.get("result") if isinstance(payload, dict) else None
    return result if isinstance(result, str) else json.dumps(payload, ensure_ascii=False)


def respond(request_id: Any, result: Any = None, error: str | None = None) -> None:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is None:
        payload["result"] = result
    else:
        payload["error"] = {"code": -32000, "message": error}
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def serve() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        method = request.get("method")
        request_id = request.get("id")
        if request_id is None:
            continue
        try:
            if method == "initialize":
                respond(
                    request_id,
                    {
                        "protocolVersion": request.get("params", {}).get(
                            "protocolVersion", "2025-06-18"
                        ),
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "claude-review-concept-flow", "version": "1"},
                    },
                )
            elif method == "tools/list":
                respond(request_id, {"tools": [TOOL]})
            elif method == "tools/call":
                params = request.get("params", {})
                args = params.get("arguments", {})
                if params.get("name") != "review" or set(args) != {"prompt"}:
                    raise ValueError("only review(prompt) is accepted")
                text = run_review(args["prompt"])
                respond(request_id, {"content": [{"type": "text", "text": text}]})
            else:
                respond(request_id, error=f"unsupported method: {method}")
        except Exception as error:  # noqa: BLE001 - MCP must structure every failure.
            respond(request_id, error=str(error))


if __name__ == "__main__":
    if sys.argv[1:] == ["--print-tool-schema"]:
        print(json.dumps(TOOL["inputSchema"], sort_keys=True))
    elif sys.argv[1:] == ["--probe"]:
        print(run_review("Reply with exactly READY."))
    elif sys.argv[1:]:
        raise SystemExit("usage: claude_review_adapter.py [--print-tool-schema|--probe]")
    else:
        serve()
