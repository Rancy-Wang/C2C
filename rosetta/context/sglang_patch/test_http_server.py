import asyncio
from typing import List, Dict, Any

import httpx
import pytest
import requests

# We assume the API server is running locally on the default host/port defined in launch.py.
# Adjust BASE_URL if you run the server elsewhere.
BASE_URL = "http://127.0.0.1:8000"

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


def build_chat_payload(example: dict) -> dict:
    messages: List[Dict[str, Any]] = []
    if example.get("system_prompt"):
        messages.append({"role": "system", "content": example["system_prompt"]})
    for um in example.get("user_messages", []):
        messages.append({"role": "user", "content": um})
    return {
        "messages": messages,
        "max_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "stream": False,
        "drop_message": example.get("drop_messages"),
        "enable_thinking": False,
    }

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
    "name": "greeting",
    "system_prompt": "You are a friendly chatbot.",
    "user_messages": ["Hello!", "Can you tell me a joke?"],
    "drop_messages": None,
}

EXAMPLE = [EXAMPLE_1, EXAMPLE_2]
port = 8000

url = f"http://localhost:{port}/v1/chat/completions"

for example in EXAMPLE:
    data = build_chat_payload(example)
    response = requests.post(url, json=data)
    print(f"Response \n{response.status_code}: {response.text}")





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