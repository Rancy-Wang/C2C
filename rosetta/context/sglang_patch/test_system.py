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
    
async def test_chat_completion():
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
        
        system_prompt = example.get("system_prompt", "You are a helpful assistant.")
        user_messages = example.get("user_messages", [])
        drop_messages_config = example.get("drop_messages", {})
        
        # Build list of all messages with their IDs
        # ID 0 = system, ID 1 = first user, ID 2 = first assistant (generated), etc.
        all_messages = [{"role": "system", "content": system_prompt}]
        message_id = 1

        for user_round, user_content in enumerate(user_messages, start=1):
            all_messages.append({"role": "user", "content": user_content})
            user_id = message_id
            assistant_id = message_id + 1
            print(f"\n[Round {user_round}] User (ID={user_id}): {user_content}")
            result = await system.generate_one_round(
                messages=all_messages,
                max_new_tokens=50,
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                enable_thinking=False,
                tools=None,
                drop_messages=drop_messages_config
            )
            all_messages.append({"role": "assistant", "content": result})
            print(f"Assistant (ID={assistant_id}): {result}")

async def test_full_conversation():
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
            max_new_tokens=512,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            enable_thinking=True,
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
    
    prompt = "Hello, please introduce yourself."
    print(f"Sending request: {prompt}")
    
    result = await system.generate_one_requests(
        messages=prompt,
        max_new_tokens=128,
        temperature=0.0,
    )
    
    print(f"Generation result:\n{result}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    asyncio.run(test_chat_completion())
