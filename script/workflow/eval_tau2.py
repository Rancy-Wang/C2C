"""
Tau2-bench multi-turn tool-calling evaluation.

Evaluates agents on airline (50), retail (115), telecom (114+) domains
using LLM-simulated users and stateful tool execution with tau2's evaluator.

Usage:
    # Single task dry run
    python script/workflow/eval_tau2.py --domain airline --limit 1 --num-workers 1

    # 5 airline tasks
    python script/workflow/eval_tau2.py --domain airline --limit 5 --num-workers 1

    # Full airline (50 tasks)
    python script/workflow/eval_tau2.py --domain airline --num-workers 4

    # Telecom with small task set
    python script/workflow/eval_tau2.py --domain telecom --task-set telecom_small --num-workers 4

    # Resume interrupted run
    python script/workflow/eval_tau2.py --domain airline --resume

    # Multiple trials per task
    python script/workflow/eval_tau2.py --domain airline --num-trials 3
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

from loguru import logger

logger.disable("tau2")

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from rosetta.benchmark.tau2.interface import load_tasks
from rosetta.workflow.camel_utils import create_model, read_jsonl, setup_env, write_jsonl

# Errors that are transient and worth retrying on --resume.
# Non-transient errors (logic bugs, assertion failures, etc.) are kept as-is.
RETRYABLE_ERRORS = (
    "ReadTimeout",
    "APIConnectionError",
    "APITimeoutError",
    "RemoteProtocolError",
    "ConnectionError",
    "TimeoutError",
    "RateLimitError",
)


def _solve_task_once(
    queue: mp.Queue,
    model,
    user_sim,
    env,
    task,
    tools,
    system_prompt: str,
    domain: str,
    max_steps: int,
    drop_messages: dict[int, list[int]] | None,
) -> None:
    """Run solve_task in a child process and put result/error into queue."""
    try:
        from rosetta.benchmark.tau2.evaluate import solve_task

        result = solve_task(
            model=model,
            user_sim=user_sim,
            env=env,
            task=task,
            tools=tools,
            system_prompt=system_prompt,
            domain=domain,
            max_steps=max_steps,
            drop_messages=drop_messages,
        )
        queue.put(("ok", result))
    except Exception as e:
        queue.put(("err", f"{type(e).__name__}: {e}"))


def _parse_drop_messages(raw: str | None) -> dict[int, list[int]] | None:
    """Parse drop_messages JSON string into normalized int-key dict.

    Expected CLI format:
        '{"4":[2,3]}'
    """
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid --drop-messages JSON: {e}") from e

    if not isinstance(obj, dict):
        raise ValueError("--drop-messages must be a JSON object")

    out: dict[int, list[int]] = {}
    for k, v in obj.items():
        key = int(k)
        if not isinstance(v, list):
            raise ValueError(f"--drop-messages[{k}] must be a list of message ids")
        out[key] = [int(x) for x in v]
    return out


def _resolve_api_key(value: str | None, env_name: str | None) -> str | None:
    """Resolve an API key from an explicit value or environment variable."""
    if value:
        return value
    if env_name:
        return os.getenv(env_name)
    return None


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker(
    worker_id: int,
    work_items: list[tuple[int, int]],
    args: argparse.Namespace,
    process_dir: Path,
) -> None:
    """Worker process: create models, loop over (task_id, trial) pairs, write results."""
    setup_env()

    # Lazy imports inside worker to avoid tau2 import overhead in main process
    from rosetta.benchmark.tau2.evaluate import UserSimulator, solve_task
    from rosetta.benchmark.tau2.interface import (
        load_tasks,
        get_environment,
        get_system_prompt,
        get_tools_info,
        make_function_tools,
    )

    # --- Create agent model ---
    if args.model_provider == "hf":
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible:
            gpu_ids = visible.split(",")
            assigned = gpu_ids[worker_id % len(gpu_ids)]
            os.environ["CUDA_VISIBLE_DEVICES"] = assigned

        hf_tokenizer = AutoTokenizer.from_pretrained(args.model_type)
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_type, dtype=torch.bfloat16, device_map="auto",
        )

        if args.lora_checkpoint:
            from peft import PeftModel
            hf_model = PeftModel.from_pretrained(hf_model, args.lora_checkpoint)
            hf_model.eval()

        opt_model = None
        if args.opt_checkpoint or args.use_cache_opt:
            from rosetta.optimize.wrapper import CacheOptimizeModel
            opt_model = CacheOptimizeModel(hf_model)
            if args.opt_checkpoint:
                opt_model.load_pretrained(args.opt_checkpoint)

        agent_model = create_model(
            provider="hf",
            model=hf_model,
            tokenizer=hf_tokenizer,
            opt_model=opt_model,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            enable_thinking=not args.no_thinking,
        )
    else:
        opt_model = None
        hf_tokenizer = None
        extra_kwargs = {}
        if args.no_thinking:
            if args.model_provider == "fireworks":
                extra_kwargs["reasoning_effort"] = "none"
            elif args.model_provider == "local":
                extra_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        agent_model = create_model(
            provider=args.model_provider,
            model_type=args.model_type,
            model_url=args.model_url,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            stream=True,
            **extra_kwargs,
        )

    # --- Create user simulator model ---
    # If not explicitly provided, let user model reuse agent model URL so
    # local/openai-compatible setups work out of the box.
    user_model_url = args.user_model_url or args.model_url
    user_model_provider = (
        "local" if args.user_model_provider == "api" else args.user_model_provider
    )
    user_model_api_key = _resolve_api_key(
        args.user_model_api_key,
        args.user_model_api_key_env,
    )
    user_model = create_model(
        provider=user_model_provider,
        model_type=args.user_model_type,
        model_url=user_model_url,
        api_key=user_model_api_key,
        temperature=0.0,
        max_tokens=2048,
        chat_template_kwargs={"enable_thinking": False},
    )
    user_sim = UserSimulator(user_model)

    # --- Load tasks ---
    tasks = load_tasks(args.domain, task_set=args.task_set)

    # Pre-create one environment to get tool info for CacheOptimizeModel registration
    sample_env = get_environment(args.domain)
    tools_info = get_tools_info(sample_env)
    system_prompt = get_system_prompt(sample_env)

    # Register full domain tools on CacheOptimizeModel
    if opt_model is not None:
        tmpl_kwargs = {"enable_thinking": False} if args.no_thinking else {}
        system_msg = {"role": "system", "content": system_prompt}
        opt_model.register_tools(hf_tokenizer, tools_info, system_msg, **tmpl_kwargs)

    out_file = process_dir / f"worker_{worker_id}.jsonl"
    traj_file = process_dir / f"worker_{worker_id}_traj.jsonl"

    file_mode = "a" if args.resume else "w"
    with out_file.open(file_mode, encoding="utf-8") as fout, \
         traj_file.open(file_mode, encoding="utf-8") as ftraj:
        for task_idx, trial in work_items:
            t0 = time.time()
            err = None
            result = None

            try:
                # Fresh environment per task
                env = get_environment(args.domain)
                tools = make_function_tools(env)
                sp = get_system_prompt(env)
                user_sim.env = env

                q: mp.Queue = mp.Queue(maxsize=1)
                p_eval = mp.Process(
                    target=_solve_task_once,
                    args=(
                        q,
                        agent_model,
                        user_sim,
                        env,
                        tasks[task_idx],
                        tools,
                        sp,
                        args.domain,
                        args.max_steps,
                        args.drop_messages,
                    ),
                )
                p_eval.start()
                p_eval.join(timeout=args.item_timeout_seconds)

                if p_eval.is_alive():
                    p_eval.terminate()
                    p_eval.join(timeout=5)
                    err = (
                        f"ItemTimeoutError: solve_task exceeded "
                        f"{args.item_timeout_seconds}s"
                    )
                else:
                    if p_eval.exitcode == 0 and not q.empty():
                        status, payload = q.get_nowait()
                        if status == "ok":
                            result = payload
                        else:
                            err = payload
                    else:
                        err = (
                            f"WorkerProcessError: child exited with code "
                            f"{p_eval.exitcode}"
                        )
            except Exception as e:
                err = f"{type(e).__name__}: {e}"

            seconds = time.time() - t0

            # Lightweight record
            record = {
                "task_id": task_idx,
                "trial": trial,
                "reward": result["reward"] if result else 0.0,
                "reward_info": result.get("reward_info") if result else {},
                "termination_reason": result.get("termination_reason") if result else None,
                "seconds": round(seconds, 2),
                "error": err,
                "no_thinking": args.no_thinking,
                "drop_messages": args.drop_messages,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            # Heavy trajectory (messages + tools)
            saved_msgs = [
                {k: v for k, v in m.items() if not k.startswith("_")}
                for m in (result["messages"] if result else [])
            ]
            trajectory = {
                "task_id": task_idx,
                "trial": trial,
                "no_thinking": args.no_thinking,
                "drop_messages": args.drop_messages,
                "messages": saved_msgs,
                "tools": tools_info,
            }
            ftraj.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
            ftraj.flush()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_pass_at_k(records: list[dict], num_trials: int) -> dict:
    """Compute pass rate and pass^k metrics.

    pass^k = fraction of tasks passing in at least 1 of k trials.
    """
    by_task: dict[int, list[float]] = {}
    for rec in records:
        tid = rec["task_id"]
        by_task.setdefault(tid, []).append(rec.get("reward", 0.0))

    num_tasks = len(by_task)
    if num_tasks == 0:
        return {"num_tasks": 0, "pass_rate": 0.0}

    pass_1 = sum(max(rewards) for rewards in by_task.values()) / num_tasks
    metrics = {"num_tasks": num_tasks, "pass_rate": pass_1}

    if num_trials > 1:
        for k in range(1, num_trials + 1):
            pass_k_count = 0
            for rewards in by_task.values():
                n = len(rewards)
                s = sum(1 for r in rewards if r >= 1.0)
                if s >= 1 and k <= n:
                    if n - s < k:
                        prob = 1.0
                    else:
                        prob = 1.0 - math.comb(n - s, k) / math.comb(n, k)
                    pass_k_count += prob
            metrics[f"pass^{k}"] = pass_k_count / num_tasks

    return metrics


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(
    output_path: Path,
    records: list[dict],
    metrics: dict,
    args: argparse.Namespace,
) -> Path:
    """Write summary.txt next to the output file. Returns summary path."""
    total = len(records)
    errors = sum(1 for r in records if r.get("error"))
    avg_seconds = sum(r.get("seconds", 0) for r in records) / total if total else 0

    by_task: dict[int, list[float]] = {}
    for rec in records:
        by_task.setdefault(rec["task_id"], []).append(rec.get("reward", 0.0))

    thinking_status = "disabled" if args.no_thinking else "enabled"
    lines = [
        "=" * 80,
        f"TAU2-BENCH EVALUATION SUMMARY — {args.domain.upper()}",
        "=" * 80,
        f"Domain: {args.domain}",
        f"Task set: {args.task_set or args.domain}",
        f"Thinking: {thinking_status}",
        f"Tasks: {metrics['num_tasks']}, Trials per task: {args.num_trials}",
        f"Total records: {total}, Errors: {errors}",
        f"Average time per record: {avg_seconds:.1f}s",
        "",
        "--- Metrics ---",
        f"  Pass rate: {metrics['pass_rate']:.4f}",
    ]
    if args.num_trials > 1:
        for k in range(1, args.num_trials + 1):
            key = f"pass^{k}"
            if key in metrics:
                lines.append(f"  {key}: {metrics[key]:.4f}")

    lines += [
        "",
        "--- Arguments ---",
    ]
    for k, v in vars(args).items():
        if "api_key" in k:
            v = "***" if v else None
        if isinstance(v, list):
            v = " ".join(str(x) for x in v)
        lines.append(f"  --{k.replace('_', '-')} {v}")

    lines += [
        "",
        "--- Per-task Breakdown ---",
        f"{'Task ID':<10} {'Best Reward':<15} {'Trials':}",
        "-" * 50,
    ]
    for tid in sorted(by_task.keys()):
        rewards = by_task[tid]
        best = max(rewards)
        detail = ", ".join(f"{r:.0f}" for r in rewards)
        lines.append(f"{tid:<10} {best:<15.0f} {detail}")

    lines += ["", f"Output: {output_path}", "=" * 80]

    summary_path = output_path.parent / f"summary_{args.domain}.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tau2-bench evaluation")

    # Domain
    parser.add_argument("--domain", default="airline",
                        choices=["airline", "retail", "telecom", "telecom-workflow", "all"])
    parser.add_argument("--task-set", default=None,
                        help="Task set name (default: same as domain). "
                             "For telecom: telecom, telecom_full, telecom_small")

    # Agent model
    parser.add_argument("--model-provider", default="fireworks")
    parser.add_argument(
        "--model-type",
        default="accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
    )
    parser.add_argument("--model-url", default=None,
                        help="API base URL for local/compatible providers")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=30)

    # User simulator model
    parser.add_argument("--user-model-provider", default="fireworks")
    parser.add_argument(
        "--user-model-type",
        default="accounts/fireworks/models/kimi-k2p5",
    )
    parser.add_argument("--user-model-url", default=None,
                        help="API base URL for local/compatible user simulator provider")
    parser.add_argument(
        "--user-model-api-key",
        default=None,
        help="API key for user simulator provider; prefer --user-model-api-key-env.",
    )
    parser.add_argument(
        "--user-model-api-key-env",
        default="USER_MODEL_API_KEY",
        help="Environment variable that stores the user simulator API key.",
    )

    # Contextual drop (agent-side only)
    parser.add_argument(
        "--drop-messages",
        default=None,
        help="JSON dict mapping trigger message id to dropped message ids, "
             "e.g. '{\"4\":[2,3]}'",
    )

    # HF-specific
    parser.add_argument("--use-cache-opt", action="store_true",
                        help="Use CacheOptimizeModel (hf provider only)")
    parser.add_argument("--opt-checkpoint", default=None,
                        help="Path to pretrained CacheOptimizeModel checkpoint")
    parser.add_argument("--lora-checkpoint", default=None,
                        help="Path to PEFT LoRA adapter directory (hf provider only)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Pass enable_thinking=False to HF chat template")

    # Data / output
    parser.add_argument("--limit", type=int, nargs="+", default=[0])
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-trials", type=int, default=1)

    # Workers
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--item-timeout-seconds",
        type=int,
        default=600,
        help="Per-item hard timeout in seconds. Exceeding this marks the item as failed.",
    )

    args = parser.parse_args()
    args.drop_messages = _parse_drop_messages(args.drop_messages)

    ALL_DOMAINS = ["airline", "retail"]
    domains = ALL_DOMAINS if args.domain == "all" else [args.domain]

    # Explicit output for a single domain — use as-is
    if args.output is not None and len(domains) == 1:
        run_domain(args)
        return

    # Derive output directory
    if args.output is not None:
        out_dir = Path(args.output).parent
    else:
        if args.opt_checkpoint:
            exp = Path(args.opt_checkpoint).name
        elif args.lora_checkpoint:
            exp = Path(args.lora_checkpoint).name
        else:
            exp = args.model_type.split("/")[-1]
        out_dir = Path(f"local/trajectory/tau2/{exp}")

    for domain in domains:
        args.domain = domain
        args.output = str(out_dir / f"{domain}.jsonl")
        run_domain(args)


def run_domain(args: argparse.Namespace) -> None:
    """Run evaluation for a single domain."""
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load tasks to get count ---
    setup_env()
    tasks = load_tasks(args.domain, task_set=args.task_set)
    num_tasks = len(tasks)

    # --- Determine task indices ---
    limit = args.limit
    if len(limit) == 2:
        start, end = limit
    elif limit[0] > 0:
        start, end = 0, limit[0]
    else:
        start, end = 0, num_tasks

    task_indices = list(range(start, min(end, num_tasks)))

    # --- Build work items: (task_id, trial) pairs ---
    # Use domain-specific process dir to avoid conflicts when running multiple domains
    process_dir = output_path.parent / f"process_{args.domain}"
    if not args.resume and process_dir.exists():
        shutil.rmtree(process_dir)
    process_dir.mkdir(parents=True, exist_ok=True)

    all_work = [(idx, trial) for idx in task_indices for trial in range(args.num_trials)]

    if args.resume:
        # Collect all records per (task_id, trial) across worker files.
        by_key: dict[tuple[int, int], list[dict]] = {}
        for wf in sorted(process_dir.glob("worker_[0-9]*.jsonl")):
            if "_traj" in wf.name:
                continue
            for rec in read_jsonl(wf):
                key = (rec["task_id"], rec.get("trial", 0))
                by_key.setdefault(key, []).append(rec)
        # A key is "done" if any record succeeded or had a non-retryable error.
        # Only retry when ALL records are retryable errors.
        done: set[tuple[int, int]] = set()
        for key, recs in by_key.items():
            all_retryable = all(
                r.get("error") and r["error"].split(":")[0] in RETRYABLE_ERRORS
                for r in recs
            )
            if not all_retryable:
                done.add(key)
        before = len(all_work)
        all_work = [(tid, t) for tid, t in all_work if (tid, t) not in done]
        print(f"Resume: {before - len(all_work)} items done, "
              f"{len(all_work)} remaining")

    if not all_work:
        print("No tasks to evaluate.")
        return

    console = Console()
    console.print(
        f"[bold]Tau2-bench {args.domain}[/bold]: "
        f"{len(all_work)} items ({len(set(t for t, _ in all_work))} tasks), "
        f"{args.num_workers} workers"
    )

    # --- Spawn workers ---
    num_workers = min(args.num_workers, len(all_work))
    chunk_size = (len(all_work) + num_workers - 1) // num_workers
    chunks = [
        all_work[i : i + chunk_size]
        for i in range(0, len(all_work), chunk_size)
    ]

    def _count_lines(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r") as f:
            return sum(1 for line in f if line.strip())

    worker_files = [process_dir / f"worker_{i}.jsonl" for i in range(len(chunks))]
    initial_counts = [_count_lines(wf) for wf in worker_files]

    processes = []
    for wid, chunk in enumerate(chunks):
        p = mp.Process(target=worker, args=(wid, chunk, args, process_dir))
        p.start()
        processes.append(p)

    # --- Progress bar ---
    def _refresh(progress, worker_tasks, overall):
        total_done = 0
        for wid, wf in enumerate(worker_files):
            count = _count_lines(wf) - initial_counts[wid]
            progress.update(worker_tasks[wid], completed=max(0, count))
            total_done += max(0, count)
        progress.update(overall, completed=total_done)

    total_items = len(all_work)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    ) as progress:
        worker_tasks = []
        for wid in range(len(chunks)):
            worker_tasks.append(
                progress.add_task(
                    f"Worker {wid}",
                    total=len(chunks[wid]),
                )
            )
        overall = progress.add_task("[bold]Overall", total=total_items)

        while any(p.is_alive() for p in processes):
            _refresh(progress, worker_tasks, overall)
            time.sleep(0.5)
        _refresh(progress, worker_tasks, overall)

    for p in processes:
        p.join()

    # --- Aggregate (prefer success over error for each (task_id, trial)) ---
    def _is_bad(rec):
        """A record/trajectory is 'bad' if it has an error or empty messages."""
        return bool(rec.get("error")) or ("messages" in rec and not rec["messages"])

    def _dedup(records):
        seen = {}
        for r in records:
            key = (r["task_id"], r.get("trial", 0))
            prev = seen.get(key)
            # Replace if: no previous, previous was bad, or new one is good
            if not prev or _is_bad(prev) or not _is_bad(r):
                seen[key] = r
        return list(seen.values())

    # Read ALL worker files in process dir (not just current run's count)
    all_records: list[dict] = []
    all_trajs: list[dict] = []
    for rec_file in sorted(process_dir.glob("worker_[0-9]*.jsonl")):
        if "_traj" in rec_file.name:
            continue
        all_records.extend(read_jsonl(rec_file))
        traj_file = rec_file.with_name(rec_file.stem + "_traj.jsonl")
        if traj_file.exists():
            all_trajs.extend(read_jsonl(traj_file))
    all_records = _dedup(all_records)
    all_trajs = _dedup(all_trajs)

    write_jsonl(output_path, all_records)
    traj_output = output_path.parent / (output_path.stem + "_trajectories.jsonl")
    write_jsonl(traj_output, all_trajs)

    # --- Write first trajectory as pretty-printed example.json ---
    if all_trajs:
        example_path = output_path.parent / "example.json"
        example_path.write_text(
            json.dumps(all_trajs[0], indent=4, ensure_ascii=False), encoding="utf-8",
        )

    # --- Metrics ---
    metrics = compute_pass_at_k(all_records, args.num_trials)
    errors = sum(1 for r in all_records if r.get("error"))

    # --- Write summary.txt ---
    summary_path = write_summary(output_path, all_records, metrics, args)

    # --- Console output ---
    console.print(f"\n[bold]Results ({args.domain}):[/bold]")
    console.print(f"  Tasks: {metrics['num_tasks']}")
    console.print(f"  Pass rate: {metrics['pass_rate']:.3f}")
    if args.num_trials > 1:
        for k in range(1, args.num_trials + 1):
            key = f"pass^{k}"
            if key in metrics:
                console.print(f"  {key}: {metrics[key]:.3f}")
    if errors:
        console.print(f"  Errors: {errors}")

    console.print(f"\n  Records: {output_path}")
    console.print(f"  Trajectories: {traj_output}")
    if all_trajs:
        console.print(f"  Example: {output_path.parent / 'example.json'}")
    console.print(f"  Summary: {summary_path}")

    # Show per-task breakdown
    by_task: dict[int, list[float]] = {}
    for rec in all_records:
        by_task.setdefault(rec["task_id"], []).append(rec.get("reward", 0.0))

    table = Table(title="Per-task Results", show_header=True, header_style="bold cyan")
    table.add_column("Task ID", justify="right")
    table.add_column("Reward", justify="right")
    if args.num_trials > 1:
        table.add_column("Trials", justify="right")

    for tid in sorted(by_task.keys()):
        rewards = by_task[tid]
        best = max(rewards)
        style = "green" if best >= 1.0 else "red"
        if args.num_trials > 1:
            detail = ", ".join(f"{r:.0f}" for r in rewards)
            table.add_row(str(tid), f"{best:.0f}", detail, style=style)
        else:
            table.add_row(str(tid), f"{best:.0f}", style=style)

    console.print(table)


if __name__ == "__main__":
    main()
