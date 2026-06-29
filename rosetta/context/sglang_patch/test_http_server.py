import asyncio
import json
from typing import List, Dict, Any, Optional

import httpx
import pytest
import requests

# We assume the API server is running locally on the default host/port defined in launch.py.
# Adjust BASE_URL if you run the server elsewhere.
BASE_URL = "http://0.0.0.0:8000"

# ---------------------------------------------------------------------------
# Helper to build request payloads similar to the examples in test_system.py
# ---------------------------------------------------------------------------

def build_generate_payload(prompt: str) -> dict:
    return {
        "prompt": prompt,
        "max_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "enable_thinking": False,
    }


def build_chat_payload(example: dict, round: int, all_messages: List[Dict[str, Any]]) -> dict:
    
    if round == 0 and example.get("system_prompt"):
        all_messages.append({"role": "system", "content": example["system_prompt"]})
    
    all_messages.append({"role": "user", "content": example["user_messages"][round]})
    payload = {
        "model": "default",
        "messages": all_messages,
        "max_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "stream": False,
        "drop_message": example.get("drop_messages"),
        "enable_thinking": False,
    }
    print(f"\npayload: {payload}\n")
    return payload


def _normalize_assistant_content(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    if content is None:
        return ""
    return str(content)


def _extract_assistant_text(response: requests.Response) -> Optional[str]:
    # minisgl /v1/chat/completions usually returns SSE-style chunks:
    # data: {"choices":[{"delta":{"content":"..."}}], ...}
    content_type = (response.headers.get("content-type") or "").lower()
    if "text/event-stream" in content_type:
        tokens: List[str] = []
        print(f"Response \n{response.status_code}: ", end="", flush=True)
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                continue
            delta = choices[0].get("delta", {})
            if not isinstance(delta, dict):
                continue

            token = delta.get("content")
            if token is None:
                continue
            token = str(token)
            tokens.append(token)
            print(token, end="", flush=True)
        print("")
        return "".join(tokens)

    raw_body = response.text.strip()
    if raw_body.startswith("data: "):
        tokens: List[str] = []
        print(f"Response \n{response.status_code}: ", end="", flush=True)
        for line in raw_body.splitlines():
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                continue
            delta = choices[0].get("delta", {})
            if not isinstance(delta, dict):
                continue

            token = delta.get("content")
            if token is None:
                continue
            token = str(token)
            tokens.append(token)
            print(token, end="", flush=True)
        print("")
        return "".join(tokens)

    if not raw_body:
        print(f"Response \n{response.status_code}: <empty body>")
        return None

    try:
        response_json = response.json()
    except ValueError:
        print(f"Response \n{response.status_code} (non-JSON): {raw_body}")
        return None

    assistant_text = _normalize_assistant_content(
        response_json.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    print(f"Response \n{response.status_code}: {assistant_text}")
    return assistant_text

# ---------------------------------------------------------------------------
# Example payloads – borrowed from test_system.py
# ---------------------------------------------------------------------------

EXAMPLE_1 = {
    "name": "math_followup",
    "system_prompt": "You are a helpful assistant.",
    "user_messages": ["What is 15 + 27?", "Now multiply that result by 3."],
    "drop_messages": {3: [1, 2]},
}

EXAMPLE_2 = {
    "name": "memory_test",
    "system_prompt": "You are a friendly chatbot.",
    "user_messages": ["My favorite color is blue and my lucky number is 7.", "What is my lucky number?", "What is my favorite color?"],
    "drop_messages": {3: [1, 2]},
}

EXAMPLE = [EXAMPLE_1, EXAMPLE_2]
port = 8000

url = f"http://localhost:{port}/v1/chat/completions"

for example in EXAMPLE:
    all_messages = []
    for round, msg in enumerate(example["user_messages"]):
        data = build_chat_payload(example, round, all_messages)
        with requests.post(url, json=data, stream=True) as response:
            if response.status_code != 200:
                err_body = response.text.strip()
                print(f"Response \n{response.status_code}: {err_body if err_body else '<empty body>'}")
                break
            assistant_text = _extract_assistant_text(response)
        if assistant_text is None:
            break
        all_messages.append({"role": "assistant", "content": assistant_text})


# python -m minisgl --model-path /share/public/public_models/Qwen3-1.7B --host 0.0.0.0 --port 8000 --cuda-graph-max-bs 0


"""
# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_endpoint():
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        payload = build_generate_payload("Hello, introduce yourself.")
        resp = await client.post("/generate", json=payload)
        assert resp.status_code == 200, f"Unexpected status {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "text" in data
        assert isinstance(data["text"], str)
        assert len(data["text"]) > 0


@pytest.mark.asyncio
async def test_parallel_chat_completions():
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        payload1 = build_chat_payload(EXAMPLE_1)
        payload2 = build_chat_payload(EXAMPLE_2)

        task1 = client.post("/v1/chat/completions", json=payload1)
        task2 = client.post("/v1/chat/completions", json=payload2)

        resp1, resp2 = await asyncio.gather(task1, task2)

        for resp in (resp1, resp2):
            assert resp.status_code == 200, f"Failed request: {resp.text}"
            js = resp.json()
            assert "id" in js
            assert "choices" in js and len(js["choices"]) == 1
            choice = js["choices"][0]
            assert choice["message"]["role"] == "assistant"
            assert isinstance(choice["message"]["content"], str)

        # Ensure the responses differ (different user inputs)
        assert resp1.json() != resp2.json()
        """
