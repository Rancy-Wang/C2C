"""Utility functions for CAMEL message conversion."""

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Any
import json
import copy
from dotenv import load_dotenv, find_dotenv

from camel.agents import ChatAgent
from camel.messages import BaseMessage, FunctionCallingMessage
from camel.memories import MemoryRecord, ContextRecord
from camel.toolkits import FunctionTool
from camel.types import OpenAIBackendRole, RoleType, ModelPlatformType, ModelType, ChatCompletion
from camel.models import ModelFactory, BaseModelBackend
from camel.configs import ChatGPTConfig
import re
import uuid as _uuid

from pathlib import Path

from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk


def read_jsonl(path: Path) -> list[dict]:
    """Read records from a JSONL file."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records to a JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def extract_logprobs(response) -> Optional[list[dict]]:
    """Extract per-token logprobs from a ChatCompletion response.

    Works with both:
    - Standard chat completions responses (logprobs.content is a list of
      TopLogprob objects)
    - Completions endpoint responses (logprobs._raw_logprobs is a list of dicts,
      set by ``_completions_stream_run``)

    Args:
        response: ChatCompletion object.

    Returns:
        List of dicts with 'token', 'logprob', 'top_logprobs' keys,
        or None if no logprobs in the response.
    """
    choice = response.choices[0]
    if not choice.logprobs:
        return None

    # Check for raw logprobs from completions endpoint
    raw = getattr(choice.logprobs, '_raw_logprobs', None)
    if raw:
        return raw

    # Standard chat completions format
    if not choice.logprobs.content:
        return None

    result = []
    for entry in choice.logprobs.content:
        top = []
        if entry.top_logprobs:
            for tlp in entry.top_logprobs:
                d = {"token": tlp.token, "logprob": tlp.logprob}
                alt_id = getattr(tlp, "token_id", None)
                if alt_id is None and hasattr(tlp, "model_extra"):
                    alt_id = (tlp.model_extra or {}).get("token_id")
                if alt_id is not None:
                    d["token_id"] = alt_id
                top.append(d)
        d = {
            "token": entry.token,
            "logprob": entry.logprob,
            "top_logprobs": top,
        }
        # Fireworks returns token_id as an extension field
        tid = getattr(entry, "token_id", None)
        if tid is None and hasattr(entry, "model_extra"):
            tid = (entry.model_extra or {}).get("token_id")
        if tid is not None:
            d["token_id"] = tid
        result.append(d)
    return result


def _parse_oss_completions_output(text: str) -> tuple[str, str, Optional[list]]:
    """Parse raw gpt-oss completions endpoint output into structured response.

    The gpt-oss model outputs special tokens in the completions endpoint:
    - ``<|channel|>analysis<|message|>REASONING<|end|>`` for thinking
    - ``<|channel|>final<|message|>CONTENT`` for final answer
    - ``<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>JSON`` for tool calls

    Args:
        text: Raw text from completions endpoint.

    Returns:
        Tuple of (content, reasoning, tool_calls).
        tool_calls is a list of OpenAI-format dicts or None.
    """
    content = ""
    reasoning = ""
    tool_calls = None

    # Split into sections by <|start|>assistant (the generation starts mid-section)
    # The first section doesn't have <|start|>assistant prefix (continues from prompt)
    parts = re.split(r"<\|start\|>assistant", text)

    for part in parts:
        if not part.strip():
            continue

        # Extract channel type and message content
        # Use \w+ (not \S+) so we stop at <|message|> boundary
        ch_match = re.search(r"<\|channel\|>(\w+)", part)
        msg_match = re.search(r"<\|message\|>(.*?)(?:<\|end\|>|<\|call\|>|<\|return\|>|$)", part, re.DOTALL)

        if not ch_match or not msg_match:
            # Fallback: treat entire part as content
            content += part.strip()
            continue

        channel = ch_match.group(1)
        msg_text = msg_match.group(1).strip()

        if channel == "analysis":
            reasoning += msg_text
        elif channel == "final":
            content += msg_text
        elif channel.startswith("commentary"):
            # Tool call: extract tool name and arguments
            # Format: "commentary to=functions.NAME <|constrain|>json"
            # Use \w+ (not \S+) so we stop at <|constrain|> / <|channel|> boundary
            # when HF decodes special tokens without leading spaces.
            tool_match = re.search(r"to=functions\.(\w+)", part)
            if tool_match:
                tool_name = tool_match.group(1)
                try:
                    json.loads(msg_text)  # validate only
                    raw_args = msg_text  # preserve original formatting
                except json.JSONDecodeError:
                    raw_args = msg_text

                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "id": f"call_{_uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": raw_args,
                    },
                })
            else:
                content += msg_text

    return content, reasoning, tool_calls


def _convert_legacy_logprobs(logprobs_chunks: list) -> list[dict]:
    """Convert SDK completions legacy logprobs to standard format.

    The completions API returns logprobs in legacy format::

        choice.logprobs.tokens = ["token1", ...]
        choice.logprobs.token_logprobs = [0.0, ...]
        choice.logprobs.top_logprobs = [{"alt1": -1.0, ...}, ...]

    This converts to the standard format used by ``extract_logprobs()``::

        [{"token": "token1", "logprob": 0.0,
          "top_logprobs": [{"token": "alt1", "logprob": -1.0}, ...]}, ...]

    Args:
        logprobs_chunks: List of (tokens, token_logprobs, top_logprobs) tuples
            collected from streaming chunks.

    Returns:
        List of per-token logprob dicts.
    """
    result = []
    for tokens, token_logprobs, top_logprobs_list in logprobs_chunks:
        for j, tok in enumerate(tokens):
            lp = token_logprobs[j] if token_logprobs and j < len(token_logprobs) else None
            top = []
            if top_logprobs_list and j < len(top_logprobs_list) and top_logprobs_list[j]:
                for alt_tok, alt_lp in top_logprobs_list[j].items():
                    top.append({"token": alt_tok, "logprob": alt_lp})
            result.append({"token": tok, "logprob": lp, "top_logprobs": top})
    return result


# Cache for tokenizers used by completions endpoint
_completions_tokenizers: dict[str, Any] = {}


def _get_completions_tokenizer(tokenizer_name: str):
    """Get or lazily load a tokenizer for completions endpoint.

    Args:
        tokenizer_name: HuggingFace tokenizer name/path.

    Returns:
        Loaded tokenizer.
    """
    if tokenizer_name not in _completions_tokenizers:
        from transformers import AutoTokenizer
        _completions_tokenizers[tokenizer_name] = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
    return _completions_tokenizers[tokenizer_name]


def _completions_stream_run(
    api_key: str,
    base_url: str,
    model_name: str,
    tokenizer_name: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    temperature: float = 0.0,
    max_tokens: int = 32768,
    top_logprobs: int = 5,
) -> ChatCompletion:
    """Run model via completions endpoint with streaming + logprobs.

    Applies chat template locally, sends token IDs to the completions endpoint,
    parses the raw output back into a structured ChatCompletion.

    This is used for models (like gpt-oss-120b) where the chat completions
    endpoint does not return logprobs in streaming mode, but the completions
    endpoint does.

    Args:
        api_key: API key for the provider.
        base_url: Base URL for the API.
        model_name: Model identifier.
        tokenizer_name: HuggingFace tokenizer name for chat template.
        messages: List of message dicts.
        tools: Optional list of tool schemas.
        drop_message: Optional drop schedule for minisgl contextual system.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        top_logprobs: Number of top logprob alternatives.

    Returns:
        ChatCompletion with logprobs attached.
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    tokenizer = _get_completions_tokenizer(tokenizer_name)

    # Preprocess messages: convert reasoning fields for chat template
    # gpt-oss expects 'thinking', others may use 'reasoning_content'
    processed_messages = []
    for msg in messages:
        m = dict(msg)
        reasoning = m.pop("_reasoning", None) or m.pop("reasoning_content", None)
        if reasoning:
            m["thinking"] = reasoning
        processed_messages.append(m)

    # Apply chat template to get token IDs
    template_kwargs = {}
    if tools:
        template_kwargs["tools"] = tools
    prompt_ids = tokenizer.apply_chat_template(
        processed_messages, add_generation_prompt=True, **template_kwargs
    )

    # Stream from completions endpoint
    stream = client.completions.create(
        model=model_name,
        prompt=prompt_ids,
        max_tokens=max_tokens,
        stream=True,
        logprobs=top_logprobs,
        temperature=temperature,
    )

    all_text = ""
    logprobs_chunks = []
    finish_reason = None
    model = None
    completion_id = None
    created = None
    usage = None

    for chunk in stream:
        if completion_id is None:
            completion_id = chunk.id
            created = chunk.created
            model = chunk.model

        if hasattr(chunk, 'usage') and chunk.usage is not None:
            usage = chunk.usage

        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        all_text += choice.text or ""

        if choice.logprobs:
            lp = choice.logprobs
            if hasattr(lp, 'tokens') and lp.tokens:
                logprobs_chunks.append((
                    list(lp.tokens),
                    list(lp.token_logprobs) if lp.token_logprobs else [],
                    list(lp.top_logprobs) if lp.top_logprobs else [],
                ))

        if choice.finish_reason:
            finish_reason = choice.finish_reason

    # Parse output into structured response
    content, reasoning, tool_calls = _parse_oss_completions_output(all_text)

    # Convert logprobs to standard format
    converted_logprobs = _convert_legacy_logprobs(logprobs_chunks) if logprobs_chunks else []

    # Build ChatCompletion
    from openai.types.chat import ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
    from openai.types.completion_usage import CompletionUsage

    message = ChatCompletionMessage(
        role="assistant",
        content=content or None,
        tool_calls=tool_calls if tool_calls else None,
    )
    if reasoning:
        message.reasoning_content = reasoning

    choice_logprobs = None
    if converted_logprobs:
        # Store as list in a ChoiceLogprobs-compatible wrapper
        # Since ChoiceLogprobs expects TopLogprob objects, we store raw list
        # and let extract_logprobs handle both formats
        choice_logprobs = ChoiceLogprobs(content=None, refusal=None)
        # Attach raw logprobs as custom attribute for extract_logprobs
        choice_logprobs._raw_logprobs = converted_logprobs

    choice_obj = Choice(
        index=0,
        message=message,
        finish_reason=finish_reason or "stop",
        logprobs=choice_logprobs,
    )

    if usage is None:
        usage = CompletionUsage(
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(converted_logprobs),
            total_tokens=len(prompt_ids) + len(converted_logprobs),
        )

    result = ChatCompletion(
        id=completion_id or "completions_stream",
        choices=[choice_obj],
        created=created or 0,
        model=model or model_name,
        object="chat.completion",
        usage=usage,
    )
    # Store prompt token IDs for reconstruction verification.
    # These are the exact IDs sent to the API via apply_chat_template.
    result._prompt_ids = prompt_ids
    return result


def _extract_oss_reasoning(content: str) -> tuple[str, str]:
    """Extract gpt-oss reasoning from special tokens in content.

    sglang's gpt-oss detector doesn't separate reasoning into
    reasoning_content, so raw tokens like ``<|channel|>analysis<|message|>``
    leak into content. This parses them out.

    Returns:
        Tuple of (clean_content, reasoning).
    """
    if not content or "<|" not in content:
        return content, ""
    # Extract analysis sections: <|channel|>analysis<|message|>...<|end|>
    reasoning_parts = re.findall(
        r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)", content, re.DOTALL
    )
    reasoning = "\n".join(p.strip() for p in reasoning_parts if p.strip())
    # Extract final content: <|channel|>final<|message|>...
    final_match = re.search(
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)", content, re.DOTALL
    )
    if final_match:
        clean = final_match.group(1).strip()
    elif reasoning:
        # Remove all special-token sections, keep any remaining plain text
        clean = re.sub(
            r"<\|(?:start\|>assistant|channel\|>\w+|message\|>|constrain\|>\w+|end\|>|call\|>)",
            "", content
        ).strip()
        # If after stripping we only have the reasoning text back, return empty
        if clean == reasoning:
            clean = ""
    else:
        return content, ""
    return clean, reasoning


def _extract_think_tags(content: str) -> tuple[str, str]:
    """Extract <think>...</think> tags from content.

    Some providers (e.g. Fireworks) embed reasoning inline in the content
    field instead of providing a separate reasoning_content field.
    Handles two formats:
      1. ``<think>reasoning</think>answer`` — standard format
      2. ``reasoning</think>answer`` — Fireworks strips the opening tag

    Args:
        content: Raw content string that may contain think tags.

    Returns:
        Tuple of (clean_content, reasoning). If no tags found, reasoning
        is empty and content is returned unchanged.
    """
    if not content:
        return content, ""
    # Case 1: Both <think> and </think> present
    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        clean = content[:match.start()] + content[match.end():]
        return clean.strip(), reasoning
    # Case 2: Only </think> present (Fireworks strips opening <think>)
    match = re.search(r"</think>", content)
    if match:
        reasoning = content[:match.start()].strip()
        clean = content[match.end():].strip()
        if reasoning:
            return clean, reasoning
    return content, ""


def setup_env():
    """Setup environment variables."""
    load_dotenv(find_dotenv())

def create_model(
    provider: str,
    model_type: Optional[str] = None,
    model_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: int = 32768,
    stream: Optional[bool] = None,
    chat_template_kwargs: Optional[dict] = None,
    **kwargs: Any,
):
    """Create a model based on the provider.

    Args:
        provider: Model provider, one of "local", "openai", "gemini".
        model_type: Model type/name string. If None, uses provider defaults:
            - local: "local"
            - openai: GPT_4O_MINI
            - gemini: "gemini-3-flash-preview"
        model_url: API URL for local/compatible models.
        api_key: API key (uses env var if not provided).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        chat_template_kwargs: Custom chat template kwargs for local models.
            Passed through to the server's extra_body if provided.
        **kwargs: Additional model config parameters.

    Returns:
        Configured CAMEL model instance.

    Raises:
        ValueError: If provider is not supported.
    """
    if provider == "local":
        model_type = model_type or "local"
        extra_body: dict = {}
        if chat_template_kwargs is not None:
            extra_body["chat_template_kwargs"] = chat_template_kwargs
        opt_tools = kwargs.pop("opt_tools", None)
        if opt_tools is not None:
            extra_body["opt_tools"] = opt_tools
        local_config: dict = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": extra_body,
            **kwargs,
        }
        if top_p is not None:
            local_config["top_p"] = top_p
        if top_k is not None:
            extra_body["top_k"] = top_k
        if stream:
            local_config["stream"] = True
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model_type,
            model_config_dict=local_config,
            api_key=api_key or "not-needed",
            url=model_url or "http://localhost:30000/v1",
        )
    elif provider == "openai":
        model_type = model_type or ModelType.GPT_5_MINI
        config = ChatGPTConfig(max_tokens=max_tokens, temperature=temperature)
        model_config_dict = config.as_dict()
        if stream:
            model_config_dict["stream"] = True
        model_config_dict.update(kwargs)
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=model_type,
            model_config_dict=model_config_dict,
            api_key=api_key,
        )
    elif provider == "gemini":
        model_type = model_type or "gemini-3-flash-preview"
        config = ChatGPTConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=kwargs.get("reasoning_effort", "medium"),
        )
        model_config_dict = config.as_dict()
        if stream:
            model_config_dict["stream"] = True
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model_type,
            model_config_dict=model_config_dict,
            api_key=api_key or os.getenv("GEMINI_API_KEY"),
            url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    elif provider == "fireworks":
        model_type = model_type or "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507"
        config = ChatGPTConfig(
            max_tokens=max_tokens,
            temperature=temperature,
        )
        model_config_dict = config.as_dict()
        if top_p is not None:
            model_config_dict["top_p"] = top_p
        if top_k is not None:
            extra = model_config_dict.setdefault("extra_body", {})
            extra["top_k"] = top_k
        if stream:
            model_config_dict["stream"] = True

        # Merge extra kwargs (e.g., logprobs, top_logprobs).
        # 'echo' is Fireworks-specific and must go via extra_body.
        # 'use_completions' and 'tokenizer_name' are handled separately.
        echo = kwargs.pop("echo", None)
        use_completions = kwargs.pop("use_completions", False)
        tokenizer_name = kwargs.pop("tokenizer_name", None)
        model_config_dict.update(kwargs)
        if echo is not None:
            extra = model_config_dict.setdefault("extra_body", {})
            extra["echo"] = echo

        fw_api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        fw_base_url = "https://api.fireworks.ai/inference/v1"

        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model_type,
            model_config_dict=model_config_dict,
            api_key=fw_api_key,
            url=fw_base_url,
        )

        # When use_completions=True, store config for completions endpoint.
        # This enables streaming logprobs for models where chat completions
        # doesn't return them (e.g., gpt-oss-120b).
        if use_completions:
            model._completions_config = {
                "api_key": fw_api_key,
                "base_url": fw_base_url,
                "model_name": model_type,
                "tokenizer_name": tokenizer_name or model_type,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_logprobs": model_config_dict.get("top_logprobs", 5),
            }

        return model
    elif provider == "hf":
        from rosetta.workflow.hf_backend import CacheOptBackend, HFBackend

        hf_model = kwargs.pop("model", None)
        hf_tokenizer = kwargs.pop("tokenizer", None)
        opt_model = kwargs.pop("opt_model", None)
        output_parser = kwargs.pop("output_parser", None)
        enable_thinking = kwargs.pop("enable_thinking", True)
        if hf_model is None or hf_tokenizer is None:
            raise ValueError(
                "provider='hf' requires 'model' and 'tokenizer' kwargs"
            )
        common_kwargs = dict(
            tokenizer=hf_tokenizer,
            output_parser=output_parser,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
        if top_p is not None:
            common_kwargs["top_p"] = top_p
        if top_k is not None:
            common_kwargs["top_k"] = top_k
        if opt_model is not None:
            return CacheOptBackend(opt_model, **common_kwargs)
        return HFBackend(hf_model, **common_kwargs)
    else:
        raise ValueError(
            f"Unsupported model provider: {provider}. "
            f"Choose from: local, openai, gemini, fireworks, hf"
        )


def collect_stream_response(
    stream: Stream[ChatCompletionChunk],
) -> ChatCompletion:
    """Consume a streaming response and return a complete ChatCompletion.

    This function iterates through all chunks in a streaming response,
    accumulates the content, reasoning_content, and tool calls, and constructs
    a complete ChatCompletion object that matches the non-streaming API format.

    Args:
        stream: A Stream of ChatCompletionChunk objects from the model.

    Returns:
        ChatCompletion: A complete response object with all accumulated content,
            including reasoning_content if the model provides it.
    """
    collected_content = ""
    collected_reasoning = ""
    collected_tool_calls = {}  # index -> {id, type, function: {name, arguments}}
    collected_logprobs = []  # Accumulated per-token logprobs from chunks
    finish_reason = None
    model = None
    completion_id = None
    created = None
    usage = None

    for chunk in stream:
        # Capture metadata from first chunk
        if completion_id is None:
            completion_id = chunk.id
            created = chunk.created
            model = chunk.model

        # Check for usage in chunk (some providers send it in final chunk)
        if hasattr(chunk, 'usage') and chunk.usage is not None:
            usage = chunk.usage

        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        delta = choice.delta

        # Accumulate logprobs (available on some models, e.g. Kimi K2, DeepSeek)
        if hasattr(choice, 'logprobs') and choice.logprobs:
            lp_content = getattr(choice.logprobs, 'content', None)
            if lp_content:
                collected_logprobs.extend(lp_content)

        # Accumulate content
        if delta.content:
            collected_content += delta.content

        # Accumulate reasoning content (various field names used by different providers)
        # DeepSeek, Qwen3 use reasoning_content; some use thinking_content or reasoning
        reasoning_delta = None
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            reasoning_delta = delta.reasoning_content
        elif hasattr(delta, 'thinking_content') and delta.thinking_content:
            reasoning_delta = delta.thinking_content
        elif hasattr(delta, 'reasoning') and delta.reasoning:
            reasoning_delta = delta.reasoning

        if reasoning_delta:
            collected_reasoning += reasoning_delta

        # Accumulate tool calls
        if delta.tool_calls:
            for tc in delta.tool_calls:
                tc_index = tc.index

                if tc_index not in collected_tool_calls:
                    collected_tool_calls[tc_index] = {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }

                # Update id if we get it (usually only in first chunk for this tool call)
                if tc.id:
                    collected_tool_calls[tc_index]["id"] = tc.id

                # Accumulate function name and arguments
                if tc.function:
                    if tc.function.name:
                        collected_tool_calls[tc_index]["function"]["name"] += tc.function.name
                    if tc.function.arguments:
                        collected_tool_calls[tc_index]["function"]["arguments"] += tc.function.arguments

        # Capture finish reason
        if choice.finish_reason:
            finish_reason = choice.finish_reason

    # Build tool_calls list if any were collected
    tool_calls = None
    if collected_tool_calls:
        # Sort by index and filter out any with missing id
        tool_calls = [
            collected_tool_calls[i]
            for i in sorted(collected_tool_calls.keys())
            if collected_tool_calls[i]["id"]
        ]

    # If provider didn't supply separate reasoning fields, extract from content
    if not collected_reasoning and collected_content:
        collected_content, collected_reasoning = _extract_oss_reasoning(collected_content)
        if not collected_reasoning:
            collected_content, collected_reasoning = _extract_think_tags(collected_content)

    # Construct the ChatCompletion object
    from openai.types.chat import ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    message = ChatCompletionMessage(
        role="assistant",
        content=collected_content or None,
        tool_calls=tool_calls if tool_calls else None,
    )

    # Add reasoning_content as attribute if present
    if collected_reasoning:
        message.reasoning_content = collected_reasoning

    # Build logprobs for the Choice if any were collected from the stream
    choice_logprobs = None
    if collected_logprobs:
        from openai.types.chat.chat_completion import ChoiceLogprobs
        choice_logprobs = ChoiceLogprobs(
            content=collected_logprobs,
            refusal=None,
        )

    choice_obj = Choice(
        index=0,
        message=message,
        finish_reason=finish_reason or "stop",
        logprobs=choice_logprobs,
    )

    # Use collected usage from stream or create placeholder
    if usage is None:
        usage = CompletionUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

    return ChatCompletion(
        id=completion_id or "stream_collected",
        choices=[choice_obj],
        created=created or 0,
        model=model or "unknown",
        object="chat.completion",
        usage=usage,
    )


def model_run_sync(
    model: BaseModelBackend,
    messages: List[dict],
    tools: Optional[List[dict]] = None,
    drop_message: Optional[dict[int, list[int]]] = None,
) -> ChatCompletion:
    """Run a model and return a complete ChatCompletion, handling streaming transparently.

    This wrapper function calls model.run() and handles both streaming and
    non-streaming responses. If the model is configured for streaming, it
    consumes the entire stream and returns a complete ChatCompletion object.

    If the model has ``_completions_config`` (set by ``create_model`` with
    ``use_completions=True``), it uses the completions endpoint instead of
    chat completions to get streaming logprobs.

    Args:
        model: A CAMEL BaseModelBackend instance.
        messages: List of message dicts in OpenAI format.
        tools: Optional list of tool schemas.

    Returns:
        ChatCompletion: A complete response object, regardless of streaming mode.
    """
    # Use completions endpoint for models that need it for streaming logprobs
    cc = getattr(model, '_completions_config', None)
    if cc:
        return _completions_stream_run(
            api_key=cc["api_key"],
            base_url=cc["base_url"],
            model_name=cc["model_name"],
            tokenizer_name=cc["tokenizer_name"],
            messages=messages,
            tools=tools,
            temperature=cc.get("temperature", 0.0),
            max_tokens=cc.get("max_tokens", 32768),
            top_logprobs=cc.get("top_logprobs", 5),
        )

    # Inject per-request drop_message via extra_body for OpenAI-compatible backends.
    # Keep the model object unchanged after each call.
    old_cfg = getattr(model, "model_config_dict", None)
    restore_cfg = old_cfg
    if drop_message is not None and isinstance(old_cfg, dict):
        cfg = copy.deepcopy(old_cfg)
        extra_body = cfg.setdefault("extra_body", {})
        extra_body["drop_message"] = {str(int(k)): [int(x) for x in v] for k, v in drop_message.items()}
        model.model_config_dict = cfg
    try:
        response = model.run(messages, tools=tools)
    finally:
        if drop_message is not None and isinstance(old_cfg, dict):
            model.model_config_dict = restore_cfg

    # Check if response is a stream (handles both openai.Stream and _SyncStreamWrapper)
    if isinstance(response, Stream):
        return collect_stream_response(response)

    # Check for other streaming wrapper types (e.g., _SyncStreamWrapper)
    # These are iterable and don't have .choices attribute
    if hasattr(response, '__iter__') and not hasattr(response, 'choices'):
        return collect_stream_response(response)

    # Non-streaming response - extract reasoning if not provided separately
    msg = response.choices[0].message
    if not getattr(msg, 'reasoning_content', None) and msg.content:
        clean, reasoning = _extract_oss_reasoning(msg.content)
        if not reasoning:
            clean, reasoning = _extract_think_tags(msg.content)
        if reasoning:
            msg.content = clean
            msg.reasoning_content = reasoning

    # Filter spurious tool calls from echo mode.
    # When echo=True, the API may parse echoed prompt tokens as extra tool
    # calls with names like "<|start|>system".  Valid function names never
    # contain "<|", so these can be safely removed.
    echo_enabled = getattr(model, "model_config_dict", {}).get("extra_body", {}).get("echo", False)
    if echo_enabled and msg.tool_calls:
        filtered = [tc for tc in msg.tool_calls if "<|" not in tc.function.name]
        msg.tool_calls = filtered or None

    return response


def context_records_to_memory_records(
    records: List[ContextRecord]
) -> List[MemoryRecord]:
    """Convert ContextRecord list to standard message format.
    
    Args:
        records: List of MemoryRecord objects.
    """
    return [ctx_record.memory_record for ctx_record in records]

def memoryRecord_flip_role(message: MemoryRecord) -> MemoryRecord:
    """Flip the role of a message."""
    if message.message.role_type == RoleType.USER:
        message.message.role_type = RoleType.ASSISTANT
    elif message.message.role_type == RoleType.ASSISTANT:
        message.message.role_type = RoleType.USER
    elif message.message.role_type == RoleType.SYSTEM:
        message.message.role_type = RoleType.SYSTEM
    elif message.message.role_type == RoleType.FUNCTION:
        message.message.role_type = RoleType.FUNCTION
    elif message.message.role_type == RoleType.TOOL:
        message.message.role_type = RoleType.TOOL
    else:
        raise ValueError(f"Unsupported role type: {message.message.role_type}.")
    return message

def messages_to_memoryRecords(
    chat_history: List[dict],
    skip_system: bool = False
) -> List[MemoryRecord]:
    """Convert standard message format to CAMEL MemoryRecord list.

    Args:
        chat_history: List of dictionaries with 'role' and 'content' keys.
                     Roles can be 'user', 'assistant', 'system', 'function',
                     'tool', or 'developer'.
        skip_system: Whether to skip system messages. Default is True.

    Returns:
        List of MemoryRecord objects suitable for CAMEL agents.

    Example:
        >>> chat_history = [
        ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
        ...     {'role': 'user', 'content': 'Hello'},
        ...     {'role': 'assistant', 'content': 'Hi there!'}
        ... ]
        >>> message_list = convert_to_camel_messages(chat_history)
        >>> len(message_list)  # System message skipped by default
        2
    """
    message_list = []

    # Build a mapping of tool_call_id -> function_name for tool messages
    # that don't have func_name specified
    tool_call_map = {}
    for msg in chat_history:
        if msg.get('role') == 'assistant' and 'tool_calls' in msg:
            for tc in msg['tool_calls']:
                tool_call_map[tc['id']] = tc['function']['name']
    
    for message in chat_history:
        role = message['role']
        content = message['content']
        
        if role == 'user':
            message_list.append(
                MemoryRecord(
                    message=BaseMessage.make_user_message(
                        role_name="user", 
                        content=content
                    ),
                    role_at_backend=OpenAIBackendRole.USER
                )
            )
        elif role == 'assistant':
            # Check if this assistant message has tool_calls
            tool_calls = message.get('tool_calls')
            if tool_calls:
                # Use FunctionCallingMessage for assistant messages with tool calls
                # Extract function name and arguments from first tool_call
                first_call = tool_calls[0]
                func_name = first_call.get('function', {}).get('name')
                args_str = first_call.get('function', {}).get('arguments', '{}')
                import json
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}

                base_msg = FunctionCallingMessage(
                    role_name="assistant",
                    role_type=RoleType.ASSISTANT,
                    content=content,
                    meta_dict={'tool_calls': tool_calls},
                    func_name=func_name,
                    args=args,
                    tool_call_id=first_call.get('id')
                )
            else:
                base_msg = BaseMessage.make_assistant_message(
                    role_name="assistant",
                    content=content
                )
            message_list.append(
                MemoryRecord(
                    message=base_msg,
                    role_at_backend=OpenAIBackendRole.ASSISTANT
                )
            )
        elif role == 'system':
            if not skip_system:
                message_list.append(
                    MemoryRecord(
                        message=BaseMessage.make_system_message(
                            role_name="System",
                            content=content
                        ),
                        role_at_backend=OpenAIBackendRole.SYSTEM
                    )
                )
        elif role == 'function':
            message_list.append(
                MemoryRecord(
                    message=BaseMessage(
                        role_name="function",
                        role_type=RoleType.DEFAULT,
                        content=content,
                        meta_dict=None
                    ),
                    role_at_backend=OpenAIBackendRole.FUNCTION
                )
            )
        elif role == 'tool':
            # Tool messages use FunctionCallingMessage with FUNCTION role
            tool_call_id = message.get('tool_call_id')
            func_name = message.get('func_name')

            # If func_name not provided, try to look it up from tool_call_map
            if not func_name and tool_call_id:
                func_name = tool_call_map.get(tool_call_id)

            message_list.append(
                MemoryRecord(
                    message=FunctionCallingMessage(
                        role_name="tool",
                        role_type=RoleType.DEFAULT,
                        content=content,
                        meta_dict=None,
                        result=content,
                        tool_call_id=tool_call_id,
                        func_name=func_name
                    ),
                    role_at_backend=OpenAIBackendRole.FUNCTION
                )
            )
        elif role == 'developer':
            message_list.append(
                MemoryRecord(
                    message=BaseMessage(
                        role_name="developer",
                        role_type=RoleType.DEFAULT,
                        content=content,
                        meta_dict=None
                    ),
                    role_at_backend=OpenAIBackendRole.DEVELOPER
                )
            )
        else:
            raise ValueError(f"Unsupported role: {role}.")
    
    return message_list



def memoryRecords_to_messages(
    records: List[MemoryRecord]
) -> List[dict]:
    """Convert MemoryRecord list to standard message format.
    
    Args:
        records: List of MemoryRecord objects.
    """
    return [record.to_openai_message() for record in records]

def add_tool_requests_to_chat_history(
    chat_history: List[dict],
    tool_request,
) -> List[dict]:
    """Add tool requests to chat history."""
    last_msg = chat_history[-1]
    if last_msg.get("role") == "assistant":
        # Format tool_calls according to what record_interaction expects
        # Arguments must be JSON string for messages_to_memoryRecords
        args = tool_request.args or {}
        args_str = json.dumps(args) if isinstance(args, dict) else args
        last_msg["tool_calls"] = [
            {
                "id": tool_request.tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_request.tool_name,
                    "arguments": args_str,
                },
            }
        ]
    return chat_history


@dataclass
class StepResult:
    """Result from ExternalToolAgent.step().

    Attributes:
        content: Final response text from the agent.
        num_tool_calls: Number of tool calls made.
        tools_used: List of unique tool names used.
        terminated_early: Whether execution stopped before natural completion.
        termination_reason: Reason for early termination (if any).
        usage: Accumulated token usage across all internal calls.
    """
    content: str
    num_tool_calls: int = 0
    tools_used: List[str] = field(default_factory=list)
    terminated_early: bool = False
    termination_reason: str = ""
    usage: Optional[dict] = None


class ExternalToolAgent:
    """Wrapper for ChatAgent with external tools and context limit handling.

    Executes tools externally with explicit control flow, checking context
    length between each iteration. Stops gracefully when approaching token limit.

    Example:
        >>> agent = ExternalToolAgent(
        ...     system_message="You are a helpful assistant.",
        ...     model=worker_model,
        ...     tools=worker_tools,
        ...     reserved_tokens=2048,
        ... )
        >>> result = agent.step("Search for information about X")
        >>> print(result.content, result.num_tool_calls)

    With live logging:
        >>> from rosetta.workflow.display import ConvLogger
        >>> logger = ConvLogger()
        >>> agent = ExternalToolAgent(..., logger=logger)
        >>> result = agent.step("Search for X")  # Shows live updates
    """

    def __init__(
        self,
        system_message: str,
        model: BaseModelBackend,
        tools: List[FunctionTool],
        reserved_tokens: int = 2048,
        token_limit: Optional[int] = None,
        logger: Optional[Any] = None,
    ):
        """Initialize ExternalToolAgent.

        Args:
            system_message: System prompt for the agent.
            model: Model backend to use.
            tools: List of tools available to the agent.
            reserved_tokens: Tokens to reserve; stops when limit approached.
            token_limit: Context window size. If None, defaults to 128000.
            logger: Optional ConvLogger for live message display. Must have
                start(), stop(), and update(messages) methods.
        """
        self.agent = ChatAgent(
            system_message=system_message,
            model=model,
            external_tools=tools,
            summarize_threshold=None,
        )
        self.tool_map = {tool.get_function_name(): tool for tool in tools}
        self.token_limit = token_limit if token_limit is not None else 128000
        self.token_threshold = self.token_limit - reserved_tokens
        self.logger = logger
        self._accumulated_usage: dict = {}

    def _accumulate_usage(self, response) -> None:
        """Accumulate token usage from a response.

        Args:
            response: ChatAgentResponse or raw model response with usage info.
        """
        usage = None
        if hasattr(response, 'info') and response.info:
            usage = response.info.get("usage")
        elif hasattr(response, 'usage') and response.usage:
            usage = response.usage

        if usage is None:
            return

        # Convert usage object to dict if needed
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif not isinstance(usage, dict):
            # Try extracting attributes
            usage_dict = {}
            for key in ["completion_tokens", "prompt_tokens", "total_tokens"]:
                if hasattr(usage, key):
                    usage_dict[key] = getattr(usage, key)
            usage = usage_dict if usage_dict else None

        if usage is None:
            return

        # Accumulate
        for key in ["completion_tokens", "prompt_tokens", "total_tokens"]:
            if key in usage:
                self._accumulated_usage[key] = self._accumulated_usage.get(key, 0) + usage[key]

    def _get_accumulated_usage(self) -> Optional[dict]:
        """Get accumulated usage and reset the accumulator."""
        if not self._accumulated_usage:
            return None
        result = dict(self._accumulated_usage)
        self._accumulated_usage = {}
        return result

    @property
    def memory(self):
        """Access underlying agent's memory."""
        return self.agent.memory

    @property
    def chat_history(self):
        """Access underlying agent's chat history."""
        return self.agent.chat_history

    def _generate_summary(self, task: str) -> str:
        """Generate summary when exiting due to token limit."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            result = self.agent.summarize(filename=None, include_summaries=False)
        summary = result.get("summary", "")
        if summary:
            return f"[Context limit - summary]\nTask: {task}\n\n{summary}"
        return f"[Context limit]\nTask: {task}\nUnable to generate summary."

    def _check_context_limit(self) -> bool:
        """Check if context length exceeds threshold.

        Returns:
            True if over threshold, False otherwise.
        """
        try:
            _, num_tokens = self.agent.memory.get_context()
            return num_tokens >= self.token_threshold
        except Exception:
            return False

    def _execute_tool(self, request) -> str:
        """Execute a tool request and return result."""
        tool = self.tool_map.get(request.tool_name)
        if tool is None:
            return f"Error: Tool '{request.tool_name}' not found"
        try:
            return tool(**request.args)
        except Exception as e:
            return f"Error executing tool '{request.tool_name}': {e}"

    def _continue_from_tool_result(self):
        """Continue conversation after tool execution without adding user message.

        Returns:
            ChatCompletion from the model (streaming handled transparently).
        """
        openai_messages, num_tokens = self.agent.memory.get_context()
        response = model_run_sync(self.agent.model_backend, openai_messages)

        # Update memory with assistant response
        from camel.messages import BaseMessage
        from camel.types import OpenAIBackendRole
        content = response.choices[0].message.content or ""
        assistant_msg = BaseMessage.make_assistant_message(role_name="assistant", content=content)
        self.agent.update_memory(assistant_msg, OpenAIBackendRole.ASSISTANT)

        return response

    def step(self, message: str, max_iterations: Optional[int] = None) -> StepResult:
        """Execute task with external tool handling.

        Runs until the agent completes (no tool call) or a limit is reached.
        If a logger is configured, shows live message updates during execution.

        Args:
            message: Initial message/task to send to the agent.
            max_iterations: Maximum number of tool calls. None for unlimited.

        Returns:
            StepResult with response content and execution metadata.
        """
        num_tool_calls = 0
        tools_used = []
        is_first_call = True
        self._accumulated_usage = {}  # Reset for this step call

        # Start live logging if logger is configured
        if self.logger:
            self.logger.start()

        def _finish(result: StepResult) -> StepResult:
            """Stop logger, add usage, and return result."""
            if self.logger:
                self.logger.stop()
            result.usage = self._get_accumulated_usage()
            return result

        while True:
            # Check context length before each call
            if self._check_context_limit():
                return _finish(StepResult(
                    content=self._generate_summary(message),
                    num_tool_calls=num_tool_calls,
                    tools_used=tools_used,
                    terminated_early=True,
                    termination_reason="Token limit reached. Summarized.",
                ))

            # Call agent
            try:
                if is_first_call:
                    response = self.agent.step(message)
                    self._accumulate_usage(response)
                    is_first_call = False
                else:
                    # Continue without adding user message
                    response = self._continue_from_tool_result()
                    self._accumulate_usage(response)
            except Exception as e:
                error_str = str(e).lower()
                if any(x in error_str for x in ["too long", "context length", "maximum context"]):
                    return _finish(StepResult(
                        content=self._generate_summary(message),
                        num_tool_calls=num_tool_calls,
                        tools_used=tools_used,
                        terminated_early=True,
                        termination_reason="Context exceeded. Summarized.",
                    ))
                if self.logger:
                    self.logger.stop()
                raise

            # Update logger after agent response
            if self.logger:
                self.logger.update(self.chat_history)

            # Check for external tool request
            if hasattr(response, 'info'):
                tool_requests = response.info.get("external_tool_call_requests", [])
            else:
                # Direct model response - check for tool_calls
                tool_calls = getattr(response.choices[0].message, 'tool_calls', None)
                if tool_calls:
                    # Build tool request from response
                    tc = tool_calls[0]
                    from camel.agents._types import ToolCallRequest
                    args = tc.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except json.JSONDecodeError:
                            args = {}
                    tool_requests = [ToolCallRequest(
                        tool_name=tc.function.name,
                        args=args if isinstance(args, dict) else {},
                        tool_call_id=tc.id,
                    )]
                else:
                    tool_requests = []

            if not tool_requests:
                # No tool call - task complete
                content = response.msg.content if hasattr(response, 'msg') else response.choices[0].message.content
                return _finish(StepResult(
                    content=content or "",
                    num_tool_calls=num_tool_calls,
                    tools_used=tools_used,
                ))

            # Execute external tool
            request = tool_requests[0]
            if request.tool_name not in tools_used:
                tools_used.append(request.tool_name)

            result = self._execute_tool(request)

            # Fix the assistant message recorded by step() - it doesn't have
            # tool_calls properly encoded for the OpenAI API. We need to:
            # 1. Remove the incorrectly formatted assistant message
            # 2. Add a proper FunctionCallingMessage with tool_calls
            # 3. Add the tool result message
            self.agent.memory.pop_records(1)

            # Get assistant content from response
            if hasattr(response, 'msg'):
                assistant_content = response.msg.content or ""
            else:
                assistant_content = response.choices[0].message.content or ""

            # Add proper assistant message with tool_calls
            assist_msg = FunctionCallingMessage(
                role_name="assistant",
                role_type=RoleType.ASSISTANT,
                meta_dict=None,
                content=assistant_content,
                func_name=request.tool_name,
                args=request.args,
                tool_call_id=request.tool_call_id,
            )
            self.agent.update_memory(assist_msg, OpenAIBackendRole.ASSISTANT)

            # Add tool result message
            func_msg = FunctionCallingMessage(
                role_name="assistant",
                role_type=RoleType.ASSISTANT,
                meta_dict=None,
                content="",
                func_name=request.tool_name,
                result=result,
                tool_call_id=request.tool_call_id,
            )
            self.agent.update_memory(func_msg, OpenAIBackendRole.FUNCTION)

            # Update logger after tool execution
            if self.logger:
                self.logger.update(self.chat_history)

            num_tool_calls += 1

            # Check max iterations
            if max_iterations and num_tool_calls >= max_iterations:
                content = response.msg.content if hasattr(response, 'msg') else response.choices[0].message.content
                return _finish(StepResult(
                    content=content or "Max iterations reached",
                    num_tool_calls=num_tool_calls,
                    tools_used=tools_used,
                    terminated_early=True,
                    termination_reason="Max iterations reached.",
                ))