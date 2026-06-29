"""
Simplified singletool evaluation on BrowseComp or HotpotQA.

Runs singletool mode (run_with_tools) with multiprocessing support,
collects full trajectories, and optionally runs LLM judge.

Usage:
    # BrowseComp (default)
    python script/workflow/eval_singletool.py --dataset browsecomp --limit 2 --num-workers 1

    # HotpotQA
    python script/workflow/eval_singletool.py --dataset hotpotqa --limit 100 --num-workers 4

    # Judge existing results
    python script/workflow/eval_singletool.py --judge-only \
        --output local/trajectory/singletool/browsecomp.jsonl

    # Resume interrupted run
    python script/workflow/eval_singletool.py --resume --limit 100
"""

from __future__ import annotations

import argparse
import json
import os
import multiprocessing as mp
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from datasets import load_dataset as load_hf_dataset

from camel.toolkits import FunctionTool

from rosetta.workflow.browse_searcher import configure_search, get_document, search
from rosetta.workflow.camel_utils import create_model, read_jsonl, setup_env, write_jsonl
from rosetta.workflow.retriever import search_engine
from rosetta.workflow.evaluation import (
    LLMJudge,
    exact_match,
    extract_answer,
    write_summary,
)
from rosetta.workflow.basic_utils import msg_system, msg_user
from rosetta.workflow.singletool import make_generate_fn, run_with_tools
from rosetta.workflow.track import InteractionTracker


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

BROWSECOMP_DATA = "local/data/BrowseCompPlus/data/browsecomp_qa_pairs.jsonl"


def load_examples(dataset: str, limit: list[int]) -> list[dict]:
    """Load QA examples. Returns list of dicts with keys: id, question, answer, _idx.

    Args:
        dataset: "browsecomp" or "hotpotqa".
        limit: [N] to load first N items, or [start, end] to slice.
    """
    if len(limit) == 2:
        start, end = limit
    else:
        start, end = 0, limit[0]

    if dataset == "hotpotqa":
        split = "validation"
        if end > 0:
            split = f"validation[{start}:{end}]"
        elif start > 0:
            split = f"validation[{start}:]"
        ds = load_hf_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
        return [dict(ex, _idx=start + i) for i, ex in enumerate(ds)]

    # browsecomp
    data_path = Path(BROWSECOMP_DATA)
    if not data_path.exists():
        raise FileNotFoundError(f"BrowseComp data not found: {data_path}")
    examples = []
    with data_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if end > 0 and idx >= end:
                break
            line = line.strip()
            if not line:
                continue
            if idx < start:
                continue
            ex = json.loads(line)
            ex["question"] = ex.pop("query")
            ex["_idx"] = idx
            examples.append(ex)
    return examples


# ---------------------------------------------------------------------------
# Single example runner
# ---------------------------------------------------------------------------

def run_one(
    ex: dict,
    model,
    tools,
    max_rounds: int,
    system_prompt: str = "You are a helpful assistant.",
) -> tuple[dict, dict]:
    """Run singletool on one example. Returns (record, trajectory)."""
    idx = ex["_idx"]
    example_id = str(ex["id"])
    question = ex["question"]
    gold = ex["answer"]

    tracker = InteractionTracker()
    tracker.register_tools(llm_id=0, tools=tools)
    t0 = time.time()
    err: Optional[str] = None
    pred_raw = ""
    pred = ""
    messages = None

    try:
        messages = [msg_system(system_prompt), msg_user(question)]
        pred_raw, messages = run_with_tools(
            messages,
            make_generate_fn(model, tracker=tracker),
            tools,
            max_iterations=max_rounds,
        )
        extracted = extract_answer(pred_raw)
        pred = extracted if extracted is not None else pred_raw.strip()
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    # Extract tracker state
    rounds = usage = logprobs_data = usage_per_interaction = tools_schemas = None
    try:
        rounds = tracker.rounds
        usage = tracker.usage
        logprobs_data = tracker.get_all_logprobs()
        usage_per_interaction = list(tracker._usage_per_interaction)
        tools_schemas = tracker.get_tools(llm_id=0)
    except Exception:
        pass

    seconds = time.time() - t0
    correct_em = exact_match(pred, gold) if err is None else False

    record = {
        "idx": idx,
        "example_id": example_id,
        "question": question,
        "gold_answer": gold,
        "pred_answer": pred,
        "correct_em": correct_em,
        "seconds": round(seconds, 2),
        "rounds": rounds,
        "usage": usage,
        "error": err,
    }
    # model_identity: needed by apply_chat_template to render correct system section
    model_identity = None
    if messages and messages[0].get("role") == "system":
        model_identity = messages[0]["content"]

    trajectory = {
        "idx": idx,
        "example_id": example_id,
        "model_identity": model_identity,
        "messages": messages,
        "tools": tools_schemas,
        "logprobs": logprobs_data,
        "usage_per_interaction": usage_per_interaction,
    }
    return record, trajectory


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def worker(
    worker_id: int,
    examples: list[dict],
    args: argparse.Namespace,
    process_dir: Path,
) -> None:
    """Worker process: create model/tools, loop run_one, write files."""
    setup_env()

    if args.model_provider == "hf":
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Assign one GPU per worker when multiple GPUs are available
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible:
            gpu_ids = visible.split(",")
            assigned = gpu_ids[worker_id % len(gpu_ids)]
            os.environ["CUDA_VISIBLE_DEVICES"] = assigned

        hf_tokenizer = AutoTokenizer.from_pretrained(args.model_type)
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_type, dtype=torch.bfloat16, device_map="auto",
        )

        # LoRA adapter
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

        model = create_model(
            provider="hf",
            model=hf_model,
            tokenizer=hf_tokenizer,
            opt_model=opt_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            enable_thinking=not args.no_thinking,
        )
    else:
        opt_model = None
        hf_tokenizer = None
        extra_kwargs = {}
        if args.model_url:
            extra_kwargs["model_url"] = args.model_url
        if args.opt_tools:
            extra_kwargs["opt_tools"] = args.opt_tools
        if args.no_thinking and args.model_provider == "local":
            extra_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        model = create_model(
            provider=args.model_provider,
            model_type=args.model_type,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stream=args.stream if args.stream else None,
            **extra_kwargs,
        )

    if args.dataset == "browsecomp":
        configure_search(
            index_path=args.index_path,
            sglang_url=args.sglang_url,
        )
        tools = [FunctionTool(search), FunctionTool(get_document)]
    else:
        tools = [FunctionTool(search_engine)]

    # Register tools on CacheOptimizeModel (after tools are created)
    # Skip if already loaded from checkpoint (register_tools was done at training time)
    if opt_model is not None and not args.opt_checkpoint:
        tool_schemas = [t.get_openai_tool_schema() for t in tools]
        system_msg = {"role": "system", "content": args.system_prompt}
        opt_model.register_tools(hf_tokenizer, tool_schemas, system_msg)

    out_file = process_dir / f"worker_{worker_id}.jsonl"
    traj_file = process_dir / f"worker_{worker_id}_traj.jsonl"

    file_mode = "a" if args.resume else "w"
    with out_file.open(file_mode, encoding="utf-8") as fout, \
         traj_file.open(file_mode, encoding="utf-8") as ftraj:
        for ex in examples:
            rec, traj = run_one(ex, model, tools, args.max_rounds,
                                system_prompt=args.system_prompt)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            ftraj.write(json.dumps(traj, ensure_ascii=False) + "\n")
            ftraj.flush()


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def run_judge(output_path: Path, args: argparse.Namespace) -> None:
    """Load records + trajectories, run LLMJudge, write back."""
    setup_env()

    records = read_jsonl(output_path)
    traj_path = output_path.parent / (output_path.stem + "_trajectories.jsonl")

    # Build example_id -> messages mapping from trajectories
    traj_map: dict[str, list] = {}
    if traj_path.exists():
        for item in read_jsonl(traj_path):
            eid = str(item.get("example_id", ""))
            if eid and item.get("messages"):
                traj_map[eid] = item["messages"]

    # Inject llm0_messages for categorization
    for rec in records:
        eid = str(rec.get("example_id", ""))
        if eid in traj_map:
            rec["llm0_messages"] = traj_map[eid]

    # Create judge model
    judge_api_key = args.judge_api_key or os.getenv("JUDGE_API_KEY")
    judge_model = create_model(
        provider=args.judge_provider,
        model_type=args.judge_model_type,
        temperature=0.0,
        max_tokens=args.max_tokens,
        model_url=args.judge_model_url,
        api_key=judge_api_key,
    )
    judge = LLMJudge(judge_model, max_workers=args.num_workers)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Judging answers", total=1)
        records, total_llm, correct_llm = judge.judge_batch(
            records,
            progress_callback=lambda c, t: progress.update(task, completed=c, total=t),
        )

        task = progress.add_task("Categorizing errors", total=1)
        records, category_counts = judge.categorize_batch(
            records,
            progress_callback=lambda c, t: progress.update(task, completed=c, total=t),
        )

    # Strip llm0_messages before writing back
    for rec in records:
        rec.pop("llm0_messages", None)
    write_jsonl(output_path, records)

    # Write summary and print results
    summary_path = write_summary(
        output_path, records, total_llm, correct_llm, category_counts,
        args_dict=vars(args),
    )

    total = len(records)
    correct_em = sum(1 for r in records if r.get("correct_em", False))
    console = Console()
    console.print(f"\n[bold]Results:[/bold] total={total}")
    console.print(f"  EM:  {correct_em}/{total} = {correct_em/total:.3f}" if total else "  EM: 0")
    console.print(f"  LLM: {correct_llm}/{total_llm} = {correct_llm/total_llm:.3f}" if total_llm else "  LLM: 0")
    if category_counts:
        total_incorrect = sum(len(v) for v in category_counts.values())
        table = Table(title="Error Distribution", show_header=True, header_style="bold cyan")
        table.add_column("Category", style="magenta")
        table.add_column("Count", justify="right", style="green")
        table.add_column("%", justify="right", style="yellow")
        for cat, exs in sorted(category_counts.items(), key=lambda x: -len(x[1])):
            pct = len(exs) / total_incorrect * 100 if total_incorrect else 0
            table.add_row(cat, f"{len(exs)}/{total_incorrect}", f"{pct:.1f}%")
        console.print(table)
    console.print(f"\nSummary: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Singletool evaluation")

    # Data
    parser.add_argument("--dataset", default="browsecomp", choices=["browsecomp", "hotpotqa"])
    parser.add_argument("--output", default=None, help="Output path (default: local/trajectory/singletool/<dataset>.jsonl)")
    parser.add_argument("--limit", type=int, nargs="+", default=[4])
    parser.add_argument("--resume", action="store_true")

    # Model
    parser.add_argument("--model-provider", default="fireworks")
    parser.add_argument("--model-type", default="accounts/fireworks/models/gpt-oss-120b")
    parser.add_argument("--model-url", default=None,
                        help="API URL for local/compatible models (e.g., http://localhost:30002/v1)")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-rounds", type=int, default=30)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--use-cache-opt", action="store_true",
                        help="Use CacheOptimizeModel for learnable KV cache optimization (hf provider only)")
    parser.add_argument("--opt-checkpoint", default=None,
                        help="Path to pretrained CacheOptimizeModel checkpoint (implies --use-cache-opt)")
    parser.add_argument("--lora-checkpoint", default=None,
                        help="Path to PEFT LoRA adapter directory (hf provider only)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Pass enable_thinking=False to HF chat template")
    parser.add_argument("--opt-tools", nargs="+", default=None,
                        help="Tool names to optimize via mini-sglang opt-cache (e.g., search_engine)")
    parser.add_argument("--stream", action="store_true",
                        help="Use streaming mode (required for mini-sglang)")

    # BrowseComp search (only used when --dataset browsecomp)
    parser.add_argument("--index-path", default="local/data/BrowseCompPlus/indexes/qwen3-embedding-8b/corpus.*.pkl")
    parser.add_argument("--sglang-url", default="http://localhost:30001")

    # Workers
    parser.add_argument("--num-workers", type=int, default=4)

    # Judge
    parser.add_argument("--no-judge", action="store_true", help="Skip judge after eval")
    parser.add_argument("--judge-only", action="store_true", help="Only run judge")
    parser.add_argument("--judge-provider", default="fireworks")
    parser.add_argument("--judge-model-type", default="accounts/fireworks/models/qwen3-235b-a22b-instruct-2507")
    parser.add_argument("--judge-model-url", default=None, help="API base URL for judge model (OpenAI-compatible endpoint, e.g. https://cloud.infini-ai.com/maas/v1)")
    parser.add_argument("--judge-api-key", default=None, help="API key for judge model (falls back to JUDGE_API_KEY env var)")

    args = parser.parse_args()

    if args.output is None:
        args.output = f"local/trajectory/singletool/{args.dataset}.jsonl"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------- Judge-only mode -------
    if args.judge_only:
        if not output_path.exists():
            print(f"Error: {output_path} not found")
            return
        run_judge(output_path, args)
        return

    # ------- Load data -------
    examples = load_examples(args.dataset, args.limit)

    # ------- Resume: filter done -------
    process_dir = output_path.parent / "process"
    process_dir.mkdir(exist_ok=True)

    if args.resume and output_path.exists():
        done_ids: set[str] = set()
        for rec in read_jsonl(output_path):
            eid = rec.get("example_id")
            if not eid:
                continue
            eid = str(eid)
            if rec.get("error"):
                done_ids.discard(eid)
            else:
                done_ids.add(eid)
        before = len(examples)
        examples = [ex for ex in examples if str(ex["id"]) not in done_ids]
        print(f"Resume: {before - len(examples)} already done, {len(examples)} remaining")

    if not examples:
        print("No examples to evaluate.")
        return

    # ------- Spawn workers -------
    num_workers = min(args.num_workers, len(examples))
    chunk_size = (len(examples) + num_workers - 1) // num_workers
    chunks = [examples[i:i + chunk_size] for i in range(0, len(examples), chunk_size)]

    processes = []
    for wid, chunk in enumerate(chunks):
        p = mp.Process(target=worker, args=(wid, chunk, args, process_dir))
        p.start()
        processes.append(p)

    # ------- Progress bar -------
    worker_files = [process_dir / f"worker_{i}.jsonl" for i in range(len(chunks))]

    def _count_lines(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r") as f:
            return sum(1 for line in f if line.strip())

    def _refresh(progress, tasks, overall):
        total_done = 0
        for wid, wf in enumerate(worker_files):
            count = _count_lines(wf)
            progress.update(tasks[wid], completed=count)
            total_done += count
        progress.update(overall, completed=total_done)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    ) as progress:
        tasks = []
        for wid in range(len(chunks)):
            tasks.append(progress.add_task(f"Worker {wid}", total=len(chunks[wid])))
        overall = progress.add_task("[bold]Overall", total=len(examples))

        while any(p.is_alive() for p in processes):
            _refresh(progress, tasks, overall)
            time.sleep(0.5)
        _refresh(progress, tasks, overall)

    for p in processes:
        p.join()

    # ------- Aggregate (prefer non-empty entries per example_id) -------
    def _is_bad(rec):
        """A record/trajectory is 'bad' if it has an error or empty messages."""
        return bool(rec.get("error")) or ("messages" in rec and not rec["messages"])

    def _dedup(records, key="example_id"):
        seen = {}
        for r in records:
            k = str(r.get(key, ""))
            prev = seen.get(k)
            if not prev or _is_bad(prev) or not _is_bad(r):
                seen[k] = r
        return list(seen.values())

    all_records, all_trajs = [], []
    for wid in range(len(chunks)):
        rec_file = process_dir / f"worker_{wid}.jsonl"
        traj_file = process_dir / f"worker_{wid}_traj.jsonl"
        if rec_file.exists():
            all_records.extend(read_jsonl(rec_file))
        if traj_file.exists():
            all_trajs.extend(read_jsonl(traj_file))
    all_records = _dedup(all_records)
    all_trajs = _dedup(all_trajs)

    write_jsonl(output_path, all_records)
    traj_output = output_path.parent / (output_path.stem + "_trajectories.jsonl")
    write_jsonl(traj_output, all_trajs)

    # ------- Summary -------
    total = len(all_records)
    correct_em = sum(1 for r in all_records if r.get("correct_em", False))
    errors = sum(1 for r in all_records if r.get("error"))
    print(f"\nDone. total={total}, correct_em={correct_em}/{total}, errors={errors}")
    print(f"  Records: {output_path}")
    print(f"  Trajectories: {traj_output}")

    # ------- Optional judge -------
    if not args.no_judge:
        print("\nRunning LLM judge...")
        run_judge(output_path, args)


if __name__ == "__main__":
    main()
