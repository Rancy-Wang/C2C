from typing import Any, Dict, List, Optional, Tuple

from rosetta.workflow.basic_utils import msg_assistant, msg_tool, execute_tool, parse_tool_arguments, _clean_for_api
from rosetta.workflow.camel_utils import model_run_sync, extract_logprobs
from rosetta.workflow.track import InteractionTracker, record_interaction


def run_with_tools(
    messages: List[Dict[str, Any]],
    generate_fn,
    tools: List,
    max_iterations: int = 10,
    on_generation=None,
    logger=None,
    ctx_manager=None,
    unknown_tool_fn=None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Run model with tools until text response or done.

    Handles only the assistant turn: generate -> execute all tool calls -> repeat.
    The caller manages the outer conversation loop (user sim, etc.).

    Args:
        messages: Conversation messages so far.
        generate_fn: Callable(messages, tool_schemas) -> assistant message dict.
        tools: List of tool objects (FunctionTool-like).
        max_iterations: Maximum number of LLM call rounds.
        on_generation: Optional callback(completion) called after each round
            (generation + tool execution). Return True to stop the loop.
        logger: Optional conversation logger for display.
        ctx_manager: Manages context compression and history config.
        unknown_tool_fn: Optional callable(name, args) -> str for handling
            tool calls to tools not in the tool_map. If None, returns a
            generic error message.
    """
    tool_map = {t.get_function_name(): t for t in tools}
    tool_schemas = [t.get_openai_tool_schema() for t in tools]

    if logger:
        logger.start()
        logger.update(messages)

    completion = {}
    for _ in range(max_iterations):
        completion = generate_fn(_clean_for_api(messages), tool_schemas)
        messages.append(completion)

        tool_calls = completion.get("tool_calls")
        if not tool_calls:
            logger and logger.update(messages)
            break

        for tc in tool_calls:
            name = tc["function"]["name"]
            args_raw = tc["function"]["arguments"]
            args = parse_tool_arguments(args_raw)
            result = execute_tool(tool_map, name, args, unknown_tool_fn=unknown_tool_fn)
            messages.append(msg_tool(tc["id"], result))
        logger and logger.update(messages)

        if on_generation is not None and on_generation(completion):
            break

        if ctx_manager:
            messages = ctx_manager.apply(messages)
            logger and logger.update(messages)

    logger and logger.stop()
    return completion.get("content", ""), messages


def make_generate_fn(
    model,
    tracker: Optional[InteractionTracker] = None,
    drop_message: Optional[Dict[int, List[int]]] = None,
):
    """Build generate_fn from a CAMEL model backend.

    Args:
        model: CAMEL BaseModelBackend instance.
        tracker: Optional InteractionTracker. When provided, each generation
            records usage, logprobs, and messages into the tracker.
    """

    def fn(msgs, tools):
        response = model_run_sync(model, msgs, tools=tools, drop_message=drop_message)
        msg = response.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None)
        d = msg_assistant(msg.content or "", tool_calls=msg.tool_calls, reasoning=reasoning)
        if tracker is not None:
            msgs.append(d)
            logprobs = extract_logprobs(response)
            record_interaction(tracker, msgs, llm_id=0, usage=response.usage, logprobs=logprobs)
            msgs.pop()
            tracker.rounds = (tracker.rounds or 0) + 1
        return d

    return fn
