"""Core evaluation logic for tau2-bench multi-turn tool-calling tasks.

Provides UserSimulator (LLM-simulated user with device-side tool calls),
solve_task (agent loop), and calculate_reward (via tau2's evaluate_simulation).

The tau2 architecture has three parties: Agent, User, and Environment.
The user simulator can make tool calls on their device (e.g., toggle_data,
check_speed_test) which are routed to the environment with requestor="user".
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.user.base import STOP, TRANSFER, OUT_OF_SCOPE
from tau2.user.user_simulator import get_global_user_sim_guidelines

from rosetta.workflow.basic_utils import (
    msg_assistant,
    msg_system,
    msg_user,
    parse_tool_arguments as _parse_args,
)
from rosetta.workflow.camel_utils import model_run_sync
from rosetta.workflow.singletool import make_generate_fn, run_with_tools


STOP_TOKENS = {STOP, TRANSFER, OUT_OF_SCOPE}
FIRST_AGENT_MSG = "Hi! How can I help you today?"

# Tools that indicate the agent wants to terminate the conversation
TERMINATE_TOOLS = ["transfer_to_human_agents"]


class UserSimulator:
    """LLM-simulated user for tau2-bench multi-turn evaluation.

    Supports user-side device tool calls: when the LLM generates tool calls,
    they are executed on the environment with requestor="user" and the results
    are fed back to the LLM until it produces a text response.
    """

    def __init__(self, model, env: Optional[Environment] = None):
        self.model = model
        self.env = env
        self.messages: List[Dict[str, Any]] = []
        self._user_tool_schemas: Optional[List[dict]] = None

    def _build_system_prompt(self, task: Task) -> str:
        """Build system prompt from tau2 guidelines + task.user_scenario."""
        has_tools = self._user_tool_schemas is not None
        guidelines = get_global_user_sim_guidelines(use_tools=has_tools)
        scenario = str(task.user_scenario)
        return f"{guidelines}\n\n<scenario>\n{scenario}\n</scenario>"

    def _init_user_tools(self) -> None:
        """Load user tool schemas from environment."""
        if self.env is None:
            return
        try:
            self._user_tool_schemas = [
                t.openai_schema for t in self.env.get_user_tools()
            ]
        except (ValueError, AttributeError):
            self._user_tool_schemas = None

    def reset(self, task: Task, first_agent_msg: str) -> Tuple[str, List[dict]]:
        """Initialize and get first user response to agent's greeting.

        Returns:
            (text, trajectory_events) where trajectory_events contains any
            user tool call messages for the evaluation trajectory.
        """
        self._init_user_tools()
        self.messages = [
            {"role": "system", "content": self._build_system_prompt(task)},
            {"role": "user", "content": first_agent_msg},
        ]
        return self._generate()

    def step(self, agent_text: str) -> Tuple[str, List[dict]]:
        """Get next user message given agent response.

        Returns:
            (text, trajectory_events) where trajectory_events contains any
            user tool call messages for the evaluation trajectory.
        """
        self.messages.append({"role": "user", "content": agent_text})
        return self._generate()

    def _generate(self, max_rounds: int = 10) -> Tuple[str, List[dict]]:
        """Generate user response, executing any user tool calls on the env.

        The user sim LLM may generate tool calls (e.g., toggle_data,
        check_speed_test). These are executed on the environment with
        requestor="user", and results are fed back until the LLM produces
        a text-only response.

        Returns:
            (text, trajectory_events) — text is the user's message to the
            agent; trajectory_events is a list of trajectory dicts for user
            tool calls and their results (tagged with _requestor="user").
        """
        trajectory_events: List[dict] = []

        for _ in range(max_rounds):
            response = model_run_sync(
                self.model, self.messages, tools=self._user_tool_schemas
            )
            msg = response.choices[0].message
            tool_calls = msg.tool_calls or []

            if not tool_calls:
                # Text-only response — done
                content = msg.content or ""
                self.messages.append({"role": "assistant", "content": content})
                return content, trajectory_events

            # User tool calls — execute on env and feed results back
            tc_dicts = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
            self.messages.append(
                {"role": "assistant", "content": msg.content, "tool_calls": tc_dicts}
            )
            # Record in trajectory as user tool call
            trajectory_events.append(
                {
                    "role": "user",
                    "content": msg.content,
                    "tool_calls": tc_dicts,
                    "_requestor": "user",
                }
            )

            for tc in tool_calls:
                result = self._exec_user_tool(tc.function.name, tc.function.arguments)
                self.messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
                trajectory_events.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                        "_requestor": "user",
                    }
                )

        # Fallback: max rounds exceeded, return whatever content we have
        return msg.content or "", trajectory_events

    def _exec_user_tool(self, name: str, raw_args: Any) -> str:
        """Execute a single user tool call on the environment."""
        args = _parse_args(raw_args)
        try:
            result = self.env.make_tool_call(name, requestor="user", **args)
            self.env.sync_tools()
            return Environment.to_json_str(result)
        except Exception as e:
            return f"Error: {e}"


def _last_was_terminate(messages: List[Dict[str, Any]]) -> bool:
    """Check if the last assistant message contained a terminate tool call."""
    for msg in reversed(messages):
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc["function"]["name"] in TERMINATE_TOOLS:
                    return True
            return False
        if msg["role"] == "assistant":
            return False
    return False


def solve_task(
    model,
    user_sim: UserSimulator,
    env: Environment,
    task: Task,
    tools: list,
    system_prompt: str,
    domain: str,
    max_steps: int = 30,
    max_tool_rounds: int = 10,
    drop_messages: Optional[Dict[int, List[int]]] = None,
    **rw_kwargs,
) -> Dict[str, Any]:
    """Run the agent loop for one tau2-bench task.

    Maintains two message lists:
    - messages: what the agent model sees (system, assistant, user text, tools)
    - trajectory: full history including user device tool calls, for evaluation

    Args:
        model: Agent model backend.
        user_sim: UserSimulator instance (must have env set).
        env: tau2 Environment instance (fresh per task).
        task: Task with user_scenario and initial_state.
        tools: List of ToolWrapper objects for agent-side tools.
        system_prompt: Agent system prompt.
        domain: Domain name for reward calculation.
        max_steps: Maximum number of user-agent turns.
        max_tool_rounds: Maximum tool-calling iterations per agent turn.
        **rw_kwargs: Extra kwargs passed to run_with_tools.

    Returns:
        Dict with reward, reward_info, messages (full trajectory),
        termination_reason.
    """
    t_start = time.time()

    # Route unknown tool calls through the tau2 environment so error messages
    # match what the evaluator produces on replay (e.g. "Tool 'X' not found.").
    def _env_unknown_tool(name, args):
        try:
            result = env.make_tool_call(name, requestor="assistant", **args)
            env.sync_tools()
            return Environment.to_json_str(result)
        except Exception as e:
            return f"Error: {e}"

    # 1. Init messages (agent view) and trajectory (full, for evaluation)
    messages = [msg_system(system_prompt), msg_assistant(FIRST_AGENT_MSG)]
    trajectory = list(messages)

    # 2. Init env state from task.initial_state
    if task.initial_state:
        env.set_state(
            task.initial_state.initialization_data,
            task.initial_state.initialization_actions,
            task.initial_state.message_history or [],
        )

    # 3. Get first user response (may include user tool calls)
    user_text, user_events = user_sim.reset(task, FIRST_AGENT_MSG)
    trajectory.extend(user_events)

    if any(tok in user_text for tok in STOP_TOKENS):
        user_msg = msg_user(user_text)
        messages.append(user_msg)
        trajectory.append(user_msg)
        return _finish(trajectory, env, task, domain, "user_stop", t_start)
    user_msg = msg_user(user_text)
    messages.append(user_msg)
    trajectory.append(user_msg)

    # 4. Agent loop
    # drop_messages is applied only to agent-side generation calls.
    generate_fn = make_generate_fn(model, drop_message=drop_messages)
    termination = "max_steps"
    for step in range(max_steps):
        prev_len = len(messages)
        text, messages = run_with_tools(
            messages, generate_fn, tools, max_iterations=max_tool_rounds,
            unknown_tool_fn=_env_unknown_tool, **rw_kwargs
        )
        # Copy new agent messages to trajectory
        trajectory.extend(messages[prev_len:])

        # Check if agent used a terminate tool
        if _last_was_terminate(messages):
            termination = "agent_stop"
            break

        # Route text to user sim (may trigger user device tool calls)
        user_text, user_events = user_sim.step(text)
        trajectory.extend(user_events)

        if any(tok in user_text for tok in STOP_TOKENS):
            termination = "user_stop"
            break
        user_msg = msg_user(user_text)
        messages.append(user_msg)
        trajectory.append(user_msg)

    return _finish(trajectory, env, task, domain, termination, t_start)


def _finish(
    trajectory: List[Dict[str, Any]],
    env: Environment,
    task: Task,
    domain: str,
    termination_reason: str,
    t_start: float,
) -> Dict[str, Any]:
    """Compute reward and return result dict."""
    t_end = time.time()
    try:
        reward, reward_info = calculate_reward(
            trajectory, env, task, domain, termination_reason, t_start, t_end
        )
    except Exception as e:
        # tau2 evaluator may raise ValueError when replayed tool call results
        # don't match the trajectory (e.g. different error message formats for
        # unknown tools). Treat as reward=0 but preserve the trajectory.
        reward, reward_info = 0.0, {"error": f"{type(e).__name__}: {e}"}
    return {
        "reward": reward,
        "reward_info": reward_info,
        "messages": trajectory,
        "termination_reason": termination_reason,
    }


def calculate_reward(
    trajectory: List[Dict[str, Any]],
    env: Environment,
    task: Task,
    domain: str,
    termination_reason: str,
    t_start: float,
    t_end: float,
) -> Tuple[float, Dict]:
    """Convert trajectory to tau2 format and call evaluate_simulation."""
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType

    tau2_messages = convert_messages_to_tau2(trajectory)

    reason_map = {
        "user_stop": TerminationReason.USER_STOP,
        "agent_stop": TerminationReason.AGENT_STOP,
        "max_steps": TerminationReason.MAX_STEPS,
    }

    sim = SimulationRun(
        id=str(uuid.uuid4()),
        task_id=str(task.id),
        start_time=str(t_start),
        end_time=str(t_end),
        duration=t_end - t_start,
        termination_reason=reason_map[termination_reason],
        messages=tau2_messages,
    )

    reward_info = evaluate_simulation(
        simulation=sim,
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=domain,
    )
    return reward_info.reward, reward_info.model_dump()


def convert_messages_to_tau2(messages: List[Dict[str, Any]]) -> list:
    """Convert dict messages to tau2 Pydantic message objects.

    Handles both agent-side messages (requestor="assistant") and user-side
    device tool calls (tagged with _requestor="user").
    """
    from tau2.data_model.message import (
        AssistantMessage,
        UserMessage,
        ToolMessage,
        ToolCall,
    )

    result = []
    for msg in messages:
        role = msg["role"]
        requestor = msg.get("_requestor", "assistant")

        if role == "system":
            continue
        elif role == "user" and requestor == "user":
            # User-side device tool call
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=_parse_args(tc["function"]["arguments"]),
                    requestor="user",
                )
                for tc in msg.get("tool_calls", [])
            ]
            result.append(
                UserMessage(
                    role="user",
                    content=msg.get("content"),
                    tool_calls=tool_calls or None,
                )
            )
        elif role == "user":
            # Regular user text message
            result.append(UserMessage(role="user", content=msg["content"]))
        elif role == "assistant":
            tool_calls = None
            if msg.get("tool_calls"):
                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=_parse_args(tc["function"]["arguments"]),
                        requestor="assistant",
                    )
                    for tc in msg["tool_calls"]
                ]
            result.append(
                AssistantMessage(
                    role="assistant",
                    content=msg.get("content"),
                    tool_calls=tool_calls,
                )
            )
        elif role == "tool":
            result.append(
                ToolMessage(
                    id=msg.get("tool_call_id", ""),
                    role="tool",
                    content=msg.get("content"),
                    requestor=requestor,
                )
            )
    return result


