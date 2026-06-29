from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Any, Dict, List

import requests


TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get simple mock weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. Beijing"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers and return the sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
]


def _parse_arguments(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _tool_get_weather(args: Dict[str, Any]) -> Dict[str, Any]:
    city = str(args.get("city", "")).strip() or "unknown"
    unit = str(args.get("unit", "celsius")).lower()
    seed = sum(ord(ch) for ch in city)
    condition = ["sunny", "cloudy", "rainy", "windy"][seed % 4]
    temp_c = 15 + (seed % 15)
    if unit == "fahrenheit":
        temperature = round(temp_c * 9 / 5 + 32, 1)
        return {"city": city, "condition": condition, "temperature": temperature, "unit": "fahrenheit"}
    return {"city": city, "condition": condition, "temperature": float(temp_c), "unit": "celsius"}


def _tool_add_numbers(args: Dict[str, Any]) -> Dict[str, Any]:
    a = float(args.get("a", 0))
    b = float(args.get("b", 0))
    return {"a": a, "b": b, "sum": a + b}


def _run_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "get_weather":
        return _tool_get_weather(args)
    if name == "add_numbers":
        return _tool_add_numbers(args)
    return {"error": f"unknown tool: {name}", "args": args}


def _resolve_tool_choice(choice: str) -> str | Dict[str, Any]:
    value = choice.strip()
    if value in {"auto", "required", "none"}:
        return value
    return {"type": "function", "function": {"name": value}}


def _build_system_prompt(force_tool_first: bool) -> str:
    base = "You are a helpful assistant."
    if not force_tool_first:
        return base + " Use tools when needed and provide concise answers."
    return (
        base
        + " You must call at least one tool before giving the final answer. "
        + "Do not answer directly in round 1 without tool_calls. "
        + "After tool results are provided, produce a concise final answer."
    )


def _stream_chat(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    print("\n[request payload]")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    with requests.post(url, json=payload, stream=True, timeout=300) as response:
        print(f"\n[http] status={response.status_code}")
        if response.status_code != 200:
            print(response.text)
            return {"ok": False, "assistant_text": "", "tool_calls": None}

        assistant_chunks: List[str] = []
        tool_calls: List[Dict[str, Any]] | None = None
        finish_reason: str | None = None
        matched_stop: str | None = None
        tool_call_missing = False

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data: "):
                continue

            data = line[6:].strip()
            print(f"[sse] {data}")
            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
            except Exception:
                continue

            choices = chunk.get("choices")
            if not isinstance(choices, list) or len(choices) == 0:
                continue
            choice = choices[0] if isinstance(choices[0], dict) else {}
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                delta = {}

            content = delta.get("content")
            if content is not None:
                token = str(content)
                assistant_chunks.append(token)
                print(token, end="", flush=True)

            raw_tool_calls = delta.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                tool_calls = raw_tool_calls
                print("\n[assistant.tool_calls]")
                print(json.dumps(tool_calls, ensure_ascii=False, indent=2))

            if choice.get("finish_reason") is not None:
                finish_reason = str(choice.get("finish_reason"))
                if choice.get("matched_stop") is not None:
                    matched_stop = str(choice.get("matched_stop"))
                if choice.get("tool_call_missing") is True:
                    tool_call_missing = True

        if len(assistant_chunks) > 0:
            print()
        print(
            f"[finish] finish_reason={finish_reason}, matched_stop={matched_stop}, "
            f"tool_call_missing={tool_call_missing}"
        )
        return {
            "ok": True,
            "assistant_text": "".join(assistant_chunks),
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "matched_stop": matched_stop,
            "tool_call_missing": tool_call_missing,
        }


def run_agentic_session(
    *,
    url: str,
    model: str,
    user_prompt: str,
    max_rounds: int,
    tool_choice: str,
    require_tool_call: bool,
    force_tool_prompt: bool,
) -> bool:
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": _build_system_prompt(force_tool_prompt),
        },
        {"role": "user", "content": user_prompt},
    ]

    for round_id in range(1, max_rounds + 1):
        print(f"\n========== Round {round_id} ==========")
        payload = {
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": _resolve_tool_choice(tool_choice),
            "stream": True,
            "max_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
        }
        result = _stream_chat(url, payload)
        if not result.get("ok"):
            return

        assistant_text = str(result.get("assistant_text") or "")
        tool_calls = result.get("tool_calls")
        assistant_message: Dict[str, Any] = {"role": "assistant", "content": assistant_text}
        if isinstance(tool_calls, list) and len(tool_calls) > 0:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)

        if not isinstance(tool_calls, list) or len(tool_calls) == 0:
            if require_tool_call and round_id == 1:
                print("\n[error] tool-call check failed: round 1 has no assistant.tool_calls.")
                print("[hint] Try --tool-choice get_weather or keep --force-tool-prompt enabled.")
                print(f"\n[All Messages]: {messages}")
                return False
            print("\n[final assistant text]")
            print(assistant_text)
            print(f"\n[All Messages]: {messages}")
            return True

        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function", {})
            if not isinstance(function, dict):
                function = {}
            name = str(function.get("name", "")).strip()
            args = _parse_arguments(function.get("arguments"))
            tool_result = _run_tool(name, args)
            call_id = str(call.get("id") or f"call_{uuid.uuid4().hex[:24]}")

            print(f"\n[tool.execute] {name}({json.dumps(args, ensure_ascii=False)})")
            print(f"[tool.result] {json.dumps(tool_result, ensure_ascii=False)}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
            )

    print("\n[warning] Reached max_rounds without final text-only assistant answer.")
    print(f"\n[All Messages]: {messages}")
    return not require_tool_call

    

def main() -> None:
    parser = argparse.ArgumentParser(description="Simple Agentic tool-call client for /v1/chat/completions.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="default")
    parser.add_argument(
        "--prompt",
        type=str,
        default="How's the weather in Beijing?",
    )
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument(
        "--tool-choice",
        type=str,
        default="auto",
        help='tool_choice: auto / required / none / function_name (e.g. "get_weather").',
    )
    parser.add_argument(
        "--require-tool-call",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, round 1 must contain assistant.tool_calls or script exits with failure.",
    )
    parser.add_argument(
        "--force-tool-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, system prompt explicitly requires tool call before final answer.",
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    ok = run_agentic_session(
        url=url,
        model=args.model,
        user_prompt=args.prompt,
        max_rounds=args.max_rounds,
        tool_choice=args.tool_choice,
        require_tool_call=args.require_tool_call,
        force_tool_prompt=args.force_tool_prompt,
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
