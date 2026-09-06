#!/usr/bin/env python3
"""HF ALFWorld agent with activation steering hooks (archive wrapper)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from ssopd_agent.representation.steering import (  # noqa: E402
    SteeringSpec,
    attach_steering_hook,
    remove_hook,
)

InjectStyle = Literal[
    "none",
    "generation_wide",
    "decision_point",
    "prefill_decode",
    "decode_only",
    "action_boundary",
]


@dataclass
class GenerateResult:
    text: str
    prompt_token_count: int
    completion_token_count: int
    full_text: str
    prompt_text: str = ""


class SteerableAlfworldAgent:
    """HF causal LM agent matching ssopd_logits multistep interface + steering."""

    def __init__(
        self,
        model_path: str,
        *,
        system_prompt: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        local_files_only: bool = True,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        use_cache: bool = True,
        adapter_path: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from models.tokenizer_utils import chat_template_hash, tokenizer_hash  # noqa: E402

        self.model_path = model_path
        self.system_prompt = system_prompt
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.do_sample = bool(do_sample)
        self.use_cache = bool(use_cache)
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.inject_style: InjectStyle = "none"
        self.steer_layer = 14
        self.steer_layers: list[int] = [14]
        self.steer_alpha = 0.0
        self.steer_vector: torch.Tensor | None = None
        self.steer_vectors: dict[int, torch.Tensor] = {}
        self.steer_mode = "unit_additive"
        self._steer_hook = None
        self._steer_hooks: list[Any] = []
        self._active_spec: SteeringSpec | None = None
        self._gen_prompt_lens: list[int] | None = None
        self._gen_boundary_idx: list[int] | None = None
        self._steer_gate_fn: Any = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=local_files_only
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype_val = dtype_map.get(torch_dtype, torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype_val,
            device_map=device,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                is_trainable=False,
                local_files_only=local_files_only,
            )
        self.model.eval()

        self.meta = type(
            "Meta",
            (),
            {
                "model_name": model_path.rstrip("/").split("/")[-1],
                "model_path": model_path,
                "model_revision": "local",
                "tokenizer_hash": tokenizer_hash(self.tokenizer),
                "chat_template_hash": chat_template_hash(self.tokenizer),
            },
        )()

    def _resolve_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _prompt_text(self, observation: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": observation},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def set_steer_gate(self, gate_fn: Any | None) -> None:
        """If gate_fn() is False, skip injection (CAST-lite)."""
        self._steer_gate_fn = gate_fn

    def clear_steering(self) -> None:
        remove_hook(self._steer_hook)
        self._steer_hook = None
        for h in self._steer_hooks:
            remove_hook(h)
        self._steer_hooks = []
        self._active_spec = None
        self.inject_style = "none"
        self.steer_alpha = 0.0
        self.steer_vectors = {}
        self._steer_gate_fn = None

    def set_steering(
        self,
        *,
        layer: int,
        vector: torch.Tensor | Any,
        alpha: float,
        mode: str = "unit_additive",
        inject_style: InjectStyle = "generation_wide",
    ) -> None:
        self.clear_steering()
        if abs(float(alpha)) < 1e-12:
            self.inject_style = "none"
            return
        if not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector, dtype=torch.float32)
        self.steer_layer = int(layer)
        self.steer_alpha = float(alpha)
        self.steer_vector = vector.detach()
        self.steer_mode = mode
        self.inject_style = inject_style
        self._install_hook()

    def update_steer_vector(self, vector: torch.Tensor | Any, alpha: float | None = None) -> None:
        """Hot-swap steering direction without removing the forward hook."""
        if not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector, dtype=torch.float32)
        self.steer_vector = vector.detach()
        if self.steer_layers:
            self.steer_vectors[int(self.steer_layers[0])] = self.steer_vector
        if alpha is not None:
            self.steer_alpha = float(alpha)

    def set_steering_layers(
        self,
        *,
        vectors: dict[int, Any],
        alpha: float,
        mode: str = "unit_additive",
        inject_style: InjectStyle = "prefill_decode",
    ) -> None:
        """Hook several layers; each has its own v_l (NPM multi-layer)."""
        self.clear_steering()
        if abs(float(alpha)) < 1e-12 or not vectors:
            self.inject_style = "none"
            return
        self.steer_vectors = {}
        for li, vec in vectors.items():
            if not isinstance(vec, torch.Tensor):
                vec = torch.tensor(vec, dtype=torch.float32)
            self.steer_vectors[int(li)] = vec.detach()
        self.steer_layers = sorted(self.steer_vectors.keys())
        self.steer_layer = self.steer_layers[0]
        self.steer_vector = self.steer_vectors[self.steer_layer]
        self.steer_alpha = float(alpha)
        self.steer_mode = mode
        self.inject_style = inject_style
        self._install_hooks_multi()

    def update_steer_vectors(
        self, vectors: dict[int, Any], alpha: float | None = None
    ) -> None:
        for li, vec in vectors.items():
            if not isinstance(vec, torch.Tensor):
                vec = torch.tensor(vec, dtype=torch.float32)
            self.steer_vectors[int(li)] = vec.detach()
        if alpha is not None:
            self.steer_alpha = float(alpha)

    def _base_model(self):
        """Unwrap PeftModel / wrappers to reach the HF causal LM with .model.layers."""
        model = self.model
        # PeftModel: .get_base_model() -> CausalLM; or .base_model.model
        for _ in range(4):
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                return model
            if hasattr(model, "get_base_model"):
                try:
                    cand = model.get_base_model()
                    if cand is not model:
                        model = cand
                        continue
                except Exception:
                    pass
            if hasattr(model, "base_model"):
                cand = model.base_model
                if hasattr(cand, "model"):
                    model = cand.model
                    continue
                model = cand
                continue
            break
        return model

    def _action_boundary_offset(self, prompt_text: str) -> int:
        """Token offset from prompt start to last token of final ACTION: marker."""
        marker_ids = self.tokenizer.encode("ACTION:", add_special_tokens=False)
        ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if not ids:
            return 0
        for i in range(len(ids) - len(marker_ids), -1, -1):
            if ids[i : i + len(marker_ids)] == marker_ids:
                return i + len(marker_ids) - 1
        return len(ids) - 1

    def _boundary_indices_in_batch(
        self, prompts: list[str], prompt_lens: list[int], tlen: int
    ) -> list[int]:
        out: list[int] = []
        for prompt, plen in zip(prompts, prompt_lens):
            plen_i = int(plen)
            off = self._action_boundary_offset(prompt)
            bi = tlen - plen_i + off
            out.append(max(0, min(bi, tlen - 1)))
        return out

    def _injection_mask(self, hidden: torch.Tensor) -> torch.Tensor | None:
        """Build [B,T] bool mask for current forward pass (prefill or decode)."""
        if self._steer_gate_fn is not None and not bool(self._steer_gate_fn()):
            return None
        style = self.inject_style
        if style == "none":
            return None
        bsz, tlen, _ = hidden.shape
        mask = torch.zeros(bsz, tlen, dtype=torch.bool, device=hidden.device)
        is_decode = tlen == 1

        if style == "generation_wide":
            mask[:] = True
            return mask

        if style == "decision_point":
            if is_decode:
                return None
            for i in range(bsz):
                mask[i, tlen - 1] = True
            return mask

        if style == "prefill_decode":
            if is_decode:
                mask[:, 0] = True
            else:
                for i in range(bsz):
                    mask[i, tlen - 1] = True
            return mask

        if style == "decode_only":
            if is_decode:
                mask[:, 0] = True
                return mask
            return None

        if style == "action_boundary":
            if is_decode:
                mask[:, 0] = True
                return mask
            if self._gen_boundary_idx is not None:
                for i, bi in enumerate(self._gen_boundary_idx):
                    if i < bsz:
                        mask[i, int(bi)] = True
            else:
                for i in range(bsz):
                    mask[i, tlen - 1] = True
            return mask

        return None

    def _apply_mask_steering(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor,
        alpha: float,
        mode: str,
    ) -> torch.Tensor:
        if not bool(mask.any()):
            return hidden
        steered = hidden.clone()
        v = vector.to(device=hidden.device, dtype=hidden.dtype)
        if mode == "unit_additive":
            v_hat = (v.float() / v.float().norm().clamp_min(1e-8)).to(dtype=hidden.dtype)
            delta = float(alpha) * v_hat
            steered[mask] = steered[mask] + delta
            return steered
        h_sel = steered[mask]
        h_norm = h_sel.float().norm(dim=-1, keepdim=True)
        v_norm = v.float().norm().clamp_min(1e-8)
        scale = (float(alpha) * h_norm / v_norm).to(dtype=hidden.dtype)
        steered[mask] = h_sel + scale * v.to(dtype=hidden.dtype)
        return steered

    def _install_hook(self) -> None:
        if self.steer_vector is None or abs(self.steer_alpha) < 1e-12:
            return
        vector = self.steer_vector
        alpha = self.steer_alpha
        mode = self.steer_mode
        agent = self

        from ssopd_agent.representation.hooks import decoder_layers

        layers = decoder_layers(self._base_model())
        layer_mod = layers[self.steer_layer]

        def hook(_mod, _inp, output):
            if isinstance(output, tuple):
                hidden, rest = output[0], output[1:]
            else:
                hidden, rest = output, None
            inj_mask = agent._injection_mask(hidden)
            if inj_mask is None:
                out = hidden if rest is None else (hidden, *rest)
                return out
            # Read live vector/alpha so per-step v(q) updates without reinstalling.
            live_v = agent.steer_vector
            live_a = agent.steer_alpha
            live_m = agent.steer_mode
            if live_v is None or abs(float(live_a)) < 1e-12:
                out = hidden if rest is None else (hidden, *rest)
                return out
            steered = agent._apply_mask_steering(
                hidden, inj_mask, live_v, live_a, live_m
            )
            if rest is None:
                return steered
            return (steered, *rest)

        self._steer_hook = layer_mod.register_forward_hook(hook)
        self._steer_hooks = [self._steer_hook]
        self._active_spec = SteeringSpec(
            layer=self.steer_layer,
            alpha=self.steer_alpha,
            vector=self.steer_vector,
            mode=self.steer_mode,  # type: ignore[arg-type]
        )

    def _install_hooks_multi(self) -> None:
        if not self.steer_vectors or abs(self.steer_alpha) < 1e-12:
            return
        from ssopd_agent.representation.hooks import decoder_layers

        layers = decoder_layers(self._base_model())
        agent = self
        self._steer_hooks = []

        def make_hook(layer_i: int):
            def hook(_mod, _inp, output):
                if isinstance(output, tuple):
                    hidden, rest = output[0], output[1:]
                else:
                    hidden, rest = output, None
                inj_mask = agent._injection_mask(hidden)
                if inj_mask is None:
                    return hidden if rest is None else (hidden, *rest)
                live_v = agent.steer_vectors.get(layer_i)
                live_a = agent.steer_alpha
                if live_v is None or abs(float(live_a)) < 1e-12:
                    return hidden if rest is None else (hidden, *rest)
                steered = agent._apply_mask_steering(
                    hidden, inj_mask, live_v, live_a, agent.steer_mode
                )
                return steered if rest is None else (steered, *rest)

            return hook

        for li in self.steer_layers:
            if li < 0 or li >= len(layers):
                continue
            self._steer_hooks.append(layers[li].register_forward_hook(make_hook(li)))
        if self._steer_hooks:
            self._steer_hook = self._steer_hooks[0]
            self._active_spec = SteeringSpec(
                layer=self.steer_layer,
                alpha=self.steer_alpha,
                vector=self.steer_vector,
                mode=self.steer_mode,  # type: ignore[arg-type]
            )

    def _generate_from_encoded(
        self,
        encoded: dict[str, torch.Tensor],
        prompt_lens: list[int],
        prompts: list[str],
        seed: int | None,
    ) -> list[GenerateResult]:
        device = self._resolve_device()
        encoded = {k: v.to(device) for k, v in encoded.items()}
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": self.use_cache,
        }
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        tlen = int(encoded["input_ids"].shape[1])
        self._gen_prompt_lens = [int(x) for x in prompt_lens]
        self._gen_boundary_idx = None
        if self.inject_style == "action_boundary":
            self._gen_boundary_idx = self._boundary_indices_in_batch(
                prompts, prompt_lens, tlen
            )
        try:
            out = self.model.generate(**encoded, **gen_kwargs)
        finally:
            self._gen_prompt_lens = None
            self._gen_boundary_idx = None
        pad_id = self.tokenizer.pad_token_id
        results: list[GenerateResult] = []
        for i, plen in enumerate(prompt_lens):
            plen_i = int(plen)
            completion_ids = out[i, plen_i:]
            if pad_id is not None:
                ids = completion_ids.tolist()
                while ids and ids[-1] == pad_id:
                    ids.pop()
                completion_ids = torch.tensor(ids, device=completion_ids.device)
            text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            results.append(
                GenerateResult(
                    text=text,
                    prompt_token_count=plen_i,
                    completion_token_count=int(completion_ids.shape[0])
                    if completion_ids.numel() > 0
                    else 0,
                    full_text=prompts[i] + text,
                    prompt_text=prompts[i],
                )
            )
        return results

    @torch.no_grad()
    def generate(self, observation: str, seed: int | None = None) -> GenerateResult:
        prompt = self._prompt_text(observation)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        plen = int(encoded["input_ids"].shape[1])
        return self._generate_from_encoded(encoded, [plen], [prompt], seed)[0]

    @torch.no_grad()
    def generate_batch(
        self,
        observations: list[str],
        seed: int | None = None,
        seeds: list[int] | None = None,
    ) -> list[GenerateResult]:
        if not observations:
            return []
        # HF generate has no per-row seed; unique seeds would serialize and
        # kill ALFWorld throughput. Use batched decode with the first seed.
        seed_i = None
        if seeds:
            seed_i = int(seeds[0])
        elif seed is not None:
            seed_i = int(seed)
        if len(observations) == 1:
            return [self.generate(observations[0], seed=seed_i)]
        return self._generate_batch_same_seed(observations, seed_i)

    @torch.no_grad()
    def _generate_batch_same_seed(
        self, observations: list[str], seed: int | None
    ) -> list[GenerateResult]:
        prompts = [self._prompt_text(o) for o in observations]
        encoded = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        )
        prompt_lens = encoded["attention_mask"].sum(dim=1).tolist()
        return self._generate_from_encoded(encoded, prompt_lens, prompts, seed)

    @torch.no_grad()
    def forward_prompt_index_hidden(
        self, prompt_text: str, layer: int, token_index: int
    ) -> torch.Tensor:
        """Return hidden state at token_index (0-based, unpadded prompt)."""
        from ssopd_agent.representation.hooks import attach_block_output_hooks

        device = self._resolve_device()
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        pos = max(0, min(int(token_index), int(encoded["input_ids"].shape[1]) - 1))
        cache, handles = attach_block_output_hooks(self._base_model(), [layer])
        try:
            with torch.no_grad():
                self.model(**encoded, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        if layer not in cache:
            raise RuntimeError(f"layer {layer} not in cache")
        hidden = cache[layer]
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return hidden[0, pos].float().cpu()

    @torch.no_grad()
    def forward_prompt_last_hidden(
        self, prompt_text: str, layer: int
    ) -> torch.Tensor:
        """Return hidden state at last prompt token for direction fitting."""
        from ssopd_agent.representation.hooks import attach_block_output_hooks

        device = self._resolve_device()
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        pos = int(encoded["input_ids"].shape[1]) - 1
        cache, handles = attach_block_output_hooks(self._base_model(), [layer])
        try:
            with torch.no_grad():
                self.model(**encoded, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        if layer not in cache:
            raise RuntimeError(f"layer {layer} not in cache")
        hidden = cache[layer]
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return hidden[0, pos].float().cpu()

    @torch.no_grad()
    def forward_prompt_last_hidden_layers(
        self, prompt_text: str, layers: list[int]
    ) -> dict[int, torch.Tensor]:
        from ssopd_agent.representation.hooks import attach_block_output_hooks

        device = self._resolve_device()
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        pos = int(encoded["input_ids"].shape[1]) - 1
        cache, handles = attach_block_output_hooks(self._base_model(), list(layers))
        try:
            self.model(**encoded, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        out: dict[int, torch.Tensor] = {}
        for layer in layers:
            if layer not in cache:
                raise RuntimeError(f"layer {layer} not in cache")
            hidden = cache[layer]
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            out[int(layer)] = hidden[0, pos].float().cpu()
        return out

    @torch.no_grad()
    def last_prompt_logits(self, prompt_text: str) -> torch.Tensor:
        device = self._resolve_device()
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = self.model(**encoded, use_cache=False).logits
        return logits[0, -1].float()

    @torch.no_grad()
    def choose_alpha_kl(
        self,
        prompt_text: str,
        candidates: list[float],
        *,
        eps: float = 0.2,
    ) -> tuple[float, float]:
        """Largest α in candidates with KL(P || P_α) ≤ eps at the next-token dist."""
        saved = self.steer_alpha
        self.steer_alpha = 0.0
        p0 = torch.softmax(self.last_prompt_logits(prompt_text), dim=-1)
        best = float(min(candidates)) if candidates else 0.0
        last_kl = 0.0
        for a in sorted(float(x) for x in candidates):
            self.steer_alpha = a
            p1 = torch.softmax(self.last_prompt_logits(prompt_text), dim=-1)
            kl = float((p0 * (p0.clamp_min(1e-8).log() - p1.clamp_min(1e-8).log())).sum())
            last_kl = kl
            if kl <= eps:
                best = a
            else:
                break
        self.steer_alpha = saved
        return best, last_kl

    def close(self) -> None:
        self.clear_steering()
        try:
            del self.model
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_agent(cfg: dict[str, Any]) -> SteerableAlfworldAgent:
    from environments.alfworld_adapter import ALFWORLD_FEWSHOT_PROMPT, ALFWORLD_SYSTEM_PROMPT  # noqa: E402

    model = cfg.get("model", {})
    rollout = cfg.get("rollout", {})
    prompt_cfg = cfg.get("prompt", {})
    system = ALFWORLD_FEWSHOT_PROMPT if prompt_cfg.get("few_shot", True) else ALFWORLD_SYSTEM_PROMPT
    chat_kwargs: dict[str, Any] = {}
    if model.get("enable_thinking") is False:
        chat_kwargs["enable_thinking"] = False
    return SteerableAlfworldAgent(
        model.get("path", "/scratch/ktang115/models/Qwen3-1.7B"),
        system_prompt=system,
        device=model.get("device", "cuda"),
        torch_dtype=model.get("torch_dtype", "bfloat16"),
        local_files_only=bool(model.get("local_files_only", True)),
        max_new_tokens=int(rollout.get("max_new_tokens", 32)),
        temperature=float(rollout.get("temperature", 1.0)),
        top_p=float(rollout.get("top_p", 1.0)),
        do_sample=bool(rollout.get("do_sample", True)),
        use_cache=bool(model.get("use_cache", True)),
        adapter_path=model.get("adapter_path"),
        chat_template_kwargs=chat_kwargs or None,
    )


def build_vllm_agent(cfg: dict[str, Any]):
    """Fast rollout agent (no activation steering)."""
    from environments.alfworld_adapter import ALFWORLD_FEWSHOT_PROMPT, ALFWORLD_SYSTEM_PROMPT  # noqa: E402
    from models.vllm_agent import VLLMAgent  # noqa: E402

    model = cfg.get("model", {})
    rollout = cfg.get("rollout", {})
    prompt_cfg = cfg.get("prompt", {})
    vllm_cfg = cfg.get("vllm", {})
    system = ALFWORLD_FEWSHOT_PROMPT if prompt_cfg.get("few_shot", True) else ALFWORLD_SYSTEM_PROMPT
    return VLLMAgent(
        model.get("path", "/scratch/ktang115/models/Qwen3-1.7B"),
        max_new_tokens=int(rollout.get("max_new_tokens", 32)),
        temperature=float(rollout.get("temperature", 1.0)),
        top_p=float(rollout.get("top_p", 1.0)),
        gpu_memory_utilization=float(vllm_cfg.get("gpu_memory_utilization", 0.45)),
        max_model_len=int(vllm_cfg.get("max_model_len", 4096)),
        system_prompt=system,
        seed=int(rollout.get("seed", 1010)),
        lora_path=model.get("adapter_path"),
    )


def build_rollout_agent(cfg: dict[str, Any]):
    backend = str(cfg.get("rollout", {}).get("backend", "vllm")).lower()
    if backend == "hf":
        return build_agent(cfg)
    return build_vllm_agent(cfg)
