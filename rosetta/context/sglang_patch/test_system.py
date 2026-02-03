import asyncio
import uuid
import multiprocessing as mp
from contextual_system import ContextualSystem
import logging
import argparse
import yaml
import os
os.environ["LOG_LEVEL"] = "WARNING"

logging.basicConfig(level=logging.INFO)

def load_examples(yaml_path: str) -> list:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f).get("examples", [])

async def test_contextual_system():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", 
        type=str, 
        # default="checkpoints/sft_contextual/final",
        default="/share/public/public_models/Qwen3-1.7B",
        help="HuggingFace model name"
    )
    
    parser.add_argument("--examples_yaml", type=str, default="script/context/examples.yaml")
    parser.add_argument("--example_name", type=str, default=None)
    args = parser.parse_args()

    examples = load_examples(args.examples_yaml)
    if not examples:
        print(f"No examples found in {args.examples_yaml}")
        return

    if args.example_name:
        examples = [e for e in examples if e.get("name") == args.example_name]

    """
    example = {
        "name": "math_followup",
        "system_prompt": "You are a helpful assistant.",
        "user_messages": ["What is 15 + 27?", "Now multiply that result by 3."],
        "drop_messages": {3: [1,2]}  # Drop the first user message starting from the second user message.
    }
    """
    
    print("Initializing ContextualSystem...")
    system = ContextualSystem(
        model_path=args.model_name,
        dtype="bfloat16",
        tp_size=1,
        memory_ratio=0.9
    )

    for example in examples:
        print(f"Testing example: {example['name']}")
        result = await system.generate_full_conversation(
            example=example,
            max_new_tokens=50,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            enable_thinking=False,
            tools=None,
        )
        print(f"Generation result for example {example['name']}:")

        round = 0

        for idx, (msg) in enumerate(result):
            if msg["role"] == "system":
                continue
            if msg["role"] == "user":
                round += 1
                print(f"[Round {round}] User (ID={idx}): {msg['content']}")
                if idx in example["drop_messages"]:
                    print(f"Dropped ids {example['drop_messages'][idx]}")
            
            if msg["role"] == "assistant":
                print(f"[Round {round}] Assistant (ID={idx}): {msg['content']}")

async def run_test():
    print("Initializing ContextualSystem...")
    system = ContextualSystem(
        model_path="/share/public/public_models/Qwen3-1.7B",
        dtype="bfloat16",
        tp_size=1,
        memory_ratio=0.9
    )
    
    uid = uuid.uuid4().hex
    prompt = "Hello, please introduce yourself."
    print(f"[{uid}] Sending request: {prompt}")
    
    result = await system.generate_one_requests(
        uid=uid,
        messages=prompt,
        max_new_tokens=128,
        temperature=0.0,
    )
    
    print(f"[{uid}] Generation result:\n{result}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    asyncio.run(test_contextual_system())
