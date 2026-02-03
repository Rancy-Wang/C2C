from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import uuid
import asyncio

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from rosetta.context.utils import top_k_top_p_filtering

from minisgl.server.args import ServerArgs, parse_args
from minisgl.distributed.info import DistributedInfo
from minisgl.utils import ZmqPullQueue, ZmqPushQueue
from minisgl.core import SamplingParams
from minisgl.message import (
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)

import multiprocessing as mp
import logging
import threading
import time

def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]

class ContextualSystem:
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        tp_size: int = 1,
        memory_ratio: float = 0.9,
    ):
        """
        args:
            - model_path (str): Path to the pre-trained model.
            - dtype (str): Data type for model weights (e.g., 'float16', 'bfloat16').
            - tp_size (int): Tensor parallelism size.
            - memory_ratio (float): The fraction of GPU memory to use for KV cache.

        """
        self.model_path = model_path
        self.dtype = dtype
        self.tp_size = tp_size
        self.memory_ratio = memory_ratio
        args = [
            "--model-path", self.model_path,
            "--tp-size", str(self.tp_size),
            "--dtype", self.dtype,
            "--memory-ratio", str(self.memory_ratio),
            "--cuda-graph-max-bs", "0",
            "--cache-type", "naive",  # Use naive cache manager for contextual conversations
        ]
        server_args, run_shell = parse_args(args, False)
        self.server_args = server_args
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # mp.set_start_method("spawn", force=True)
        world_size = server_args.tp_info.size

        self.model_procs = []
        self.ready_queue = mp.Queue()

        for i in range(world_size):
            new_args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),
            )
            proc = mp.Process(
                target=self.run_scheduler,
                args=(new_args, self.ready_queue),
                daemon=False,
                name=f"minisgl-TP{i}-scheduler",
            )
            proc.start()
            self.model_procs.append(proc)

        self.send_backend = ZmqPushQueue(server_args.zmq_backend_addr, create=False, encoder=BaseBackendMsg.encoder)
        self.recv_listener = ZmqPullQueue(server_args.zmq_detokenizer_addr, create=server_args.tokenizer_create_addr, decoder=BatchTokenizerMsg.decoder)

        self.event_lock = {}
        self.event_result = {}
        self.token_cache = {}
        self.last_table_idx = {}
        self.skip_next_token = {}  # Track if we should skip the next token (prefill token)

        # Position tracking per conversation (Phase 2)
        self._next_position = {}  # uid -> next absolute position
        self._cached_len = {}  # uid -> current KV cache length (after drops)
        self._table_idx = {}  # uid -> allocated table index

        # Wait for scheduler to be ready
        print("Waiting for scheduler to be ready...")
        ready_msg = self.ready_queue.get()
        print(f"System initialized: {ready_msg}")

        threading.Thread(target=self._result_listener, daemon=True).start()

    @staticmethod
    def run_scheduler(args: ServerArgs, ready_queue: mp.Queue):
        from minisgl.scheduler import Scheduler

        with torch.inference_mode():
            scheduler = Scheduler(args)
            scheduler.sync_all_ranks()

            if args.tp_info.is_primary():
                ready_queue.put("ready")
            with scheduler.engine_stream_ctx:
                scheduler.engine.stream.wait_stream(scheduler.stream)
                while True:
                    blocking = not (scheduler.prefill_manager.runnable or scheduler.decode_manager.runnable)
                    for msg in scheduler.receive_msg(blocking=blocking):
                        scheduler._process_one_msg(msg)

                    forward_input = scheduler._schedule_next_batch()
                    ongoing_data = None
                    if forward_input is not None:
                        ongoing_data = (forward_input, scheduler._forward(forward_input))

                    scheduler._process_last_data(ongoing_data, None)
    
    @staticmethod
    def hit_cache():
        pass
    
    def _result_listener(self):
        while True:
            try:
                raw_msg = self.recv_listener.get()
                msgs = _unwrap_msg(raw_msg)
                for msg in msgs:
                    if isinstance(msg, DetokenizeMsg):
                        uid = msg.uid

                        # Skip prefill token if flagged
                        if uid in self.skip_next_token and self.skip_next_token[uid]:
                            self.skip_next_token[uid] = False
                            # Still update table_idx and mark as finished for prefill
                            if msg.finished:
                                self.event_result[uid] = ""  # Empty result for prefill
                                self.event_lock[uid] = True
                                self.last_table_idx[uid] = msg.table_idx
                            continue

                        if uid not in self.token_cache:
                            self.token_cache[uid] = []

                        self.token_cache[uid].append(msg.next_token)

                        if msg.finished:
                            text = self.tokenizer.decode(
                                self.token_cache[uid],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True
                            )
                            self.event_result[uid] = text
                            self.event_lock[uid] = True
                            self.last_table_idx[uid] = msg.table_idx
                            if uid in self.token_cache:
                                del self.token_cache[uid]
            except Exception as e:
                logging.error(f"Error in result listener: {e}")
                time.sleep(0.1)
    
    def _apply_chat_template_ids(
        self,
        tokenizer,
        messages: List[dict],
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        template_kwargs: Dict[str, Any] = {"enable_thinking": enable_thinking}
        if tools:
            template_kwargs["tools"] = tools

        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )
    
    def tokenize_conversation_round_by_round(
        self,
        messages: List[dict],
        *,
        enable_thinking: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt_last: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int, str, int]]]:
        """
        Tokenize conversation round-by-round for incremental prefill.

        Pattern:
        - Non-assistant messages are tokenized via chat-template diff slicing.
        - If the next message is an assistant, generation-prompt tokens are also
        included and attributed to the input message ID.
        - Assistant messages are tokenized as content-only (or via chat-template
        diff when `tool_calls` are present), excluding the final <|im_end|>.
        """
        
        tokenizer = self.tokenizer

        if not messages:
            return torch.empty((1, 0), dtype=torch.long), []

        all_input_ids: List[torch.Tensor] = []
        boundaries: List[Tuple[int, int, str, int]] = []
        current_messages: List[dict] = []
        seq_len = 0

        eos_id = getattr(tokenizer, "eos_token_id", None)

        for msg_id, msg in enumerate(messages):
            role = msg.get("role")
            start = seq_len

            current_messages.append(dict(msg))

            full_no_gen = self._apply_chat_template_ids(
                tokenizer,
                current_messages,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                tools=tools,
            )
            new_ids = full_no_gen[:, seq_len:]

            if role == "assistant":
                if msg.get("tool_calls"):
                    if eos_id is not None and new_ids.numel() > 0:
                        flat = new_ids[0]
                        eos_pos = (flat == int(eos_id)).nonzero(as_tuple=False)
                        if eos_pos.numel() > 0:
                            new_ids = new_ids[:, : int(eos_pos[0].item())]
                else:
                    content = msg.get("content") or ""
                    new_ids = tokenizer(
                        content, return_tensors="pt", add_special_tokens=False
                    ).input_ids

            if new_ids.numel() > 0:
                all_input_ids.append(new_ids)
                seq_len += int(new_ids.shape[1])

            should_add_gen_prompt = False
            if role != "assistant":
                if msg_id < len(messages) - 1:
                    should_add_gen_prompt = (
                        messages[msg_id + 1].get("role") == "assistant"
                    )
                else:
                    should_add_gen_prompt = bool(add_generation_prompt_last)

            if should_add_gen_prompt:
                full_with_gen = self._apply_chat_template_ids(
                    tokenizer,
                    current_messages,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                    tools=tools,
                )
                gen_prompt_ids = full_with_gen[:, seq_len:]
                if gen_prompt_ids.numel() > 0:
                    all_input_ids.append(gen_prompt_ids)
                    seq_len += int(gen_prompt_ids.shape[1])

            boundaries.append((start, seq_len, str(role), msg_id))

        if not all_input_ids:
            return torch.empty((1, 0), dtype=torch.long), boundaries
        return torch.cat(all_input_ids, dim=1), boundaries
    
    def get_drop_ids(
        self,
        target_id: int,
        drop_messages: Optional[Dict[int, List[int]]] = None,
        new: bool = False,
    ):
        """
        Get message IDs that should be dropped.

        Args:
            target_id: The current message ID being processed
            drop_messages: Dict mapping trigger_id -> list of msg_ids to drop
            new: If False, return all historically dropped IDs without new ones (trigger < target_id - 1)
                 If True, return only IDs dropped by the previous message (trigger == target_id - 1)

        Returns:
            List of message IDs to drop
        """
        if drop_messages is None:
            return []
        drop_ids = []
        if not new:
            # Accumulate all drops triggered before target_id - 1
            for trigger_id, ids_to_drop in drop_messages.items():
                if trigger_id < target_id - 1:
                    drop_ids.extend(ids_to_drop)
        else:
            # Only get drops triggered by the previous message
            if (target_id - 1) in drop_messages:
                drop_ids.extend(drop_messages[target_id - 1])
        return drop_ids

    def _calculate_retained_len(
        self,
        boundaries: List[Tuple[int, int, str, int]],
        current_msg_id: int,
        new_drop_ids: List[int],
    ) -> int:
        """
        Calculate how many tokens are retained in KV cache after drops.
        This is the physical cache size (excludes dropped messages).
        """
        retained = 0
        for start, end, _, msg_id in boundaries:
            if msg_id >= current_msg_id:
                break
            if msg_id not in (new_drop_ids or []):
                retained += end - start
        return retained
    
    async def generate_one_round(
        self,
        uid: str,
        messages: Union[str, List[Dict[str, str]]],
        drop_messages: Optional[Dict[int, List[int]]] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        enable_thinking: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        is_last_round: bool = False,
    ):
        if isinstance(messages, str):
            prompt = messages
            result = await self.generate_one_requests(
                uid=uid,
                messages=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                enable_thinking=enable_thinking,
            )
            return result

        # Initialize tracking for new conversations
        if uid not in self._next_position:
            self._next_position[uid] = 0
            self._cached_len[uid] = 0
            self._table_idx[uid] = None

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        not_gen_ids, boundaries = self.tokenize_conversation_round_by_round(
            messages,
            enable_thinking=enable_thinking,
            tools=tools,
            add_generation_prompt_last=False,
        )

        # Calculate true_seq_len (absolute position including dropped messages)
        self._next_position[uid] = boundaries[-1][1] if boundaries else self._next_position[uid]  # Default to self._next_position[uid] if no messages
        true_seq_len = self._next_position[uid]

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        prefill_sampling_params = SamplingParams(
            max_tokens=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        user_id = len(boundaries) - 1
        assistant_id = user_id + 1

        user_drop_ids = self.get_drop_ids(user_id, drop_messages, new=False)
        assistant_drop_ids = self.get_drop_ids(assistant_id, drop_messages, new=False)
        user_new_drop_ids = self.get_drop_ids(user_id, drop_messages, new=True)
        assistant_new_drop_ids = self.get_drop_ids(assistant_id, drop_messages, new=True)

        # Prefill user message
        user_req = UserMsg(
            uid=uid,
            input_ids=not_gen_ids.view(-1).to(torch.int32),
            sampling_params=prefill_sampling_params,
            table_idx=self._table_idx[uid],  # Reuse table from previous round
            message_id=user_id,
            drop_ids=user_drop_ids,
            new_drop_ids=user_new_drop_ids,
            boundaries=boundaries,
            true_seq_len=true_seq_len,  # Absolute position
            is_table_reuse=True,
        )

        # Mark to skip the prefill token
        self.skip_next_token[uid] = True

        batch_msg = BatchBackendMsg(data=[user_req])
        self.send_backend.put(batch_msg)

        self.event_lock[uid] = False

        while True:
            if self.event_lock[uid] == True:
                result = self.event_result.pop(uid)
                del self.event_lock[uid]
                break
            await asyncio.sleep(0.001)

        # Update table_idx if this is first round
        if self._table_idx[uid] is None:
            self._table_idx[uid] = self.last_table_idx[uid]

        # Update position tracking
        # Calculate how many tokens are retained after drops
        retained_len = self._calculate_retained_len(boundaries, user_id, user_new_drop_ids)
        self._cached_len[uid] = retained_len

        # Update next_position: add all tokens from current round (including those that will be dropped)
        current_round_len = sum(end - start for start, end, _, msg_id in boundaries if msg_id >= user_id)
        self._next_position[uid] = true_seq_len + current_round_len

        gen_len = len(input_ids.view(-1).to(torch.int32)) - len(not_gen_ids.view(-1).to(torch.int32))
        last_end = boundaries[-1][1] if boundaries else 0
        boundaries.append((last_end, last_end + gen_len, "assistant", assistant_id))

        # Generate assistant response
        assistant_req = UserMsg(
            uid=uid,
            input_ids=input_ids.view(-1).to(torch.int32),
            sampling_params=sampling_params,
            table_idx=self._table_idx[uid],  # Reuse same table
            message_id=assistant_id,
            drop_ids=assistant_drop_ids,
            new_drop_ids=assistant_new_drop_ids,
            boundaries=boundaries,
            true_seq_len=self._next_position[uid],  # Continue from current position
            is_table_reuse=False if is_last_round else True,  # Only reuse table for non-last rounds to allow cache cleanup after generation
        )

        batch_msg = BatchBackendMsg(data=[assistant_req])
        self.send_backend.put(batch_msg)

        self.event_lock[uid] = False

        while True:
            if self.event_lock[uid] == True:
                result = self.event_result.pop(uid)
                del self.event_lock[uid]
                return result
            await asyncio.sleep(0.001)
    
    async def reset_cache(self):
        pass
        
    async def generate_full_conversation(
        self,
        example: dict,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        enable_thinking: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        """"""
        uid = str(uuid.uuid4())

        name = example.get("name", "unnamed")
        system_prompt = example.get("system_prompt", "You are a helpful assistant.")
        user_messages = example.get("user_messages", [])
        drop_messages_config = example.get("drop_messages", {})

        self.reset_cache()

        # Initialize position tracking for this conversation
        self._next_position[uid] = 0
        self._cached_len[uid] = 0
        self._table_idx[uid] = None

        all_messages = [{"role": "system", "content": system_prompt}]
        sys_prompt = self.tokenizer.apply_chat_template(all_messages, tokenize=False,add_generation_prompt=False, enable_thinking=False)

        sys_ids = self.tokenizer.encode(sys_prompt, return_tensors="pt")

        prefill_sampling_params = SamplingParams(
            max_tokens=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        user_msg = UserMsg(
            uid=uid,
            input_ids=sys_ids.view(-1).to(torch.int32),
            sampling_params=prefill_sampling_params,
            table_idx=None,
            message_id=0,
            drop_ids=self.get_drop_ids(0, drop_messages_config, new=False),
            new_drop_ids=self.get_drop_ids(0, drop_messages_config, new=True),
            boundaries=[],
            true_seq_len=0,
            is_table_reuse=True if user_messages else False,  # If there are user messages, we will reuse the table for the first user message; otherwise, we can mark it as non-reuse to allow immediate cleanup
        )

        # Mark to skip the prefill token
        self.skip_next_token[uid] = True

        batch_msg = BatchBackendMsg(data=[user_msg])
        self.send_backend.put(batch_msg)

        self.event_lock[uid] = False

        while True:
            if self.event_lock[uid] == True:
                result = self.event_result.pop(uid)
                del self.event_lock[uid]
                break
            await asyncio.sleep(0.001)

        # Update tracking state after system prompt prefill
        self._table_idx[uid] = self.last_table_idx[uid]
        sys_len = sys_ids.shape[1]
        self._next_position[uid] = sys_len
        self._cached_len[uid] = sys_len
        
        message_id = 1

        for user_round, user_content in enumerate(user_messages, start=1):
            user_id = message_id
            assistant_id = message_id + 1
            all_messages.append({"role": "user", "content": user_content})
            
            # print(f"\n[Round {user_round}] User (ID={user_id}): {user_content}")
            
            # Append user message tokens with user_id (no generation prompt)
            response = await self.generate_one_round(
                uid=uid,
                messages=all_messages,
                drop_messages=drop_messages_config,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                enable_thinking=enable_thinking,
                tools=tools,
                is_last_round=(user_round == len(user_messages)),  # Only mark last round for non-reuse to allow cache cleanup after generation
            )
            
            all_messages.append({"role": "assistant", "content": response})
            
            # print(f"Assistant (ID={assistant_id}): {response}")
            
            message_id += 1  # Ready for next user message
        
        return all_messages
        

    async def generate_one_requests(
        self,
        uid: str,
        messages: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        enable_thinking: bool = False,
    ):
        
        if isinstance(messages, str):
            prompt = messages
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        
        user_msg = UserMsg(
            uid=uid,
            input_ids=input_ids.view(-1).to(torch.int32),
            sampling_params=sampling_params,
        )
        
        batch_msg = BatchBackendMsg(data=[user_msg])
        self.send_backend.put(batch_msg)
        
        self.event_lock[uid] = False

        while True:
            if self.event_lock[uid] == True:
                result = self.event_result.pop(uid)
                del self.event_lock[uid]
                return result
            await asyncio.sleep(0.001)

    def shutdown(self):
        self.send_backend.stop()
        self.recv_listener.stop()
        for p in self.model_procs:
            try:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        try:
                            p.kill()
                        except Exception:
                            pass
            except Exception:
                pass

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


"""
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    generator = ContextualSystem(
        model_path="/share/public/public_models/Qwen3-1.7B",
        dtype="bfloat16",
        tp_size=1,
        memory_ratio=0.9,
    )
"""