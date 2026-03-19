from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("launch")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the Contextual System API server",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the pre-trained model (local folder or HuggingFace repo ID).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to the tokenizer. Defaults to --model-path if not specified.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the server (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to listen on (default: 8000).",
    )
    parser.add_argument(
        "--tp-size",
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallelism size (default: 1).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights (default: bfloat16).",
    )
    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to allocate for KV cache (default: 0.9).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )

    return parser.parse_args(argv)


def launch_server(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("Initializing ContextualSystem ...")
    logger.info("  model_path   = %s", args.model_path)
    logger.info("  tokenizer    = %s", args.tokenizer_path or args.model_path)
    logger.info("  tp_size      = %d", args.tp_size)
    logger.info("  dtype        = %s", args.dtype)
    logger.info("  memory_ratio = %.2f", args.memory_ratio)

    from contextual_system import ContextualSystem

    system = ContextualSystem(
        model_path=args.model_path,
        dtype=args.dtype,
        tp_size=args.tp_size,
        memory_ratio=args.memory_ratio,
    )

    from api_server import app, mount_system

    mount_system(system)

    logger.info("Starting API server on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    launch_server(sys.argv[1:])
