"""HuggingFace / mock math generation models for SSOPD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ssopd_math.nxt.positions import DEFAULT_SYSTEM_PROMPT, build_prompt_text


@dataclass
class GenerateResult:
    text: str
    prompt_text: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    full_token_ids: list[int]
    prompt_token_count: int
    completion_token_count: int


class MockMathModel:
    """Deterministic stand-in for unit tests."""

    def __init__(self, mode: str = "boxed_correct", **kwargs: Any) -> None:
        self.mode = mode

    def generate(self, prompt_user: str, gold_answer: str, seed: int | None = None, **kwargs: Any) -> GenerateResult:
        if self.mode == "boxed_correct":
            text = f"Reasoning step.\n\\boxed{{{gold_answer}}}"
        elif self.mode == "boxed_wrong":
            text = "\\boxed{0}"
        elif self.mode == "no_box":
            text = "I think the answer is 0."
        else:
            text = "\\boxed{1}"
        prompt = f"User: {prompt_user}\nAssistant:"
        prompt_ids = list(range(8))
        comp_ids = list(range(8, 8 + max(len(text.split()), 1)))
        full = prompt_ids + comp_ids
        return GenerateResult(
            text=text,
            prompt_text=prompt,
            prompt_token_ids=prompt_ids,
            completion_token_ids=comp_ids,
            full_token_ids=full,
            prompt_token_count=len(prompt_ids),
            completion_token_count=len(comp_ids),
        )

    def generate_batch(self, items: list[tuple[str, str]], **kwargs: Any) -> list[GenerateResult]:
        return [self.generate(p, g, **kwargs) for p, g in items]


class HFMathModel:
    """Causal LM wrapper with optional residual-stream steering during generate."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        torch_dtype: str | torch.dtype = "bfloat16",
        local_files_only: bool = True,
        use_cache: bool = False,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        do_sample: bool = True,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        self.device = device
        self.use_cache = bool(use_cache)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.do_sample = bool(do_sample)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        if isinstance(torch_dtype, str):
            dtype = getattr(torch, torch_dtype)
        else:
            dtype = torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=local_files_only, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding is required for correct batched decoder-only generate.
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        self.model.to(device)
        self.model.eval()

    @property
    def num_hidden_layers(self) -> int:
        cfg = getattr(self.model, "config", None)
        n = getattr(cfg, "num_hidden_layers", None) if cfg is not None else None
        if n is None:
            from ssopd_math.nxt.extract import decoder_layers

            n = len(decoder_layers(self.model))
        return int(n)

    def close(self) -> None:
        try:
            del self.model
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def build_prompt(self, prompt_user: str) -> str:
        return build_prompt_text(self.tokenizer, prompt_user, system_prompt=self.system_prompt)

    def _gen_kwargs(self) -> dict[str, Any]:
        gen_kw: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": self.use_cache,
        }
        if self.do_sample:
            gen_kw["temperature"] = self.temperature
            gen_kw["top_p"] = self.top_p
        return gen_kw

    def _attach_steer_hook(
        self,
        *,
        steer_layer: int,
        steer_direction: torch.Tensor,
        steer_alpha: float,
        steer_mode: str = "unit_additive",
    ):
        """Steer newly generated tokens.

        With KV cache, decode steps often see T==1 (new token only) — steer those.
        Prefill has no completion tokens yet, so the mask stays empty.
        """
        from ssopd_math.steering.residual_hook import SteeringSpec, remove_hook as _remove_hook
        from ssopd_math.nxt.extract import decoder_layers

        spec = SteeringSpec(
            layer=int(steer_layer),
            alpha=float(steer_alpha),
            vector=steer_direction.detach().float(),
            token_indices=None,
            mode=str(steer_mode),
        )

        def _steer_hook(_module, _inputs, output):
            from ssopd_math.steering.residual_hook import (
                _rewrap_output,
                _unwrap_output,
                steer_response_mask,
            )

            hidden, rest = _unwrap_output(output)
            bsz, tlen = hidden.shape[0], hidden.shape[1]
            mask = torch.zeros(bsz, tlen, device=hidden.device, dtype=torch.bool)
            if tlen == 1:
                mask[:, :] = True
            steered, n_writes, h_norms, rhos = steer_response_mask(
                hidden,
                spec.vector,
                spec.alpha,
                mask,
                mode=getattr(spec, "mode", "unit_additive"),
            )
            spec.injection_counts.append(n_writes)
            spec.h_norms.extend(h_norms)
            spec.rho_values.extend(rhos)
            return _rewrap_output(steered, rest)

        layers = decoder_layers(self.model)
        handle = layers[int(steer_layer)].register_forward_hook(_steer_hook)
        return handle, _remove_hook, spec

    def generate(
        self,
        prompt_user: str,
        gold_answer: str | None = None,
        seed: int | None = None,
        steer_layer: int | None = None,
        steer_direction: torch.Tensor | None = None,
        steer_alpha: float = 0.0,
        steer_mode: str = "unit_additive",
    ) -> GenerateResult:
        return self.generate_batch(
            [prompt_user],
            [gold_answer or ""],
            seed=seed,
            steer_layer=steer_layer,
            steer_direction=steer_direction,
            steer_alpha=steer_alpha,
            steer_mode=steer_mode,
        )[0]

    def generate_batch(
        self,
        prompts: list[str],
        gold_answers: list[str] | None = None,
        *,
        seed: int | None = None,
        steer_layer: int | None = None,
        steer_direction: torch.Tensor | None = None,
        steer_alpha: float = 0.0,
        steer_mode: str = "unit_additive",
        **kwargs: Any,
    ) -> list[GenerateResult]:
        """True batched generate with optional residual steering (saturates GPU)."""
        if not prompts:
            return []
        gold_answers = gold_answers or [""] * len(prompts)
        if len(gold_answers) != len(prompts):
            raise ValueError("prompts/gold_answers length mismatch")

        prompt_texts = [self.build_prompt(p) for p in prompts]
        enc = self.tokenizer(
            prompt_texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        prompt_lens = attention_mask.sum(dim=1).tolist()

        gen_kw = self._gen_kwargs()
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        hook_handle = None
        remove_hook = None
        do_steer = (
            steer_layer is not None
            and steer_direction is not None
            and abs(float(steer_alpha)) > 1e-12
        )
        if do_steer:
            # Prefer cache for throughput; steering hook handles T==1 decode steps.
            gen_kw["use_cache"] = True
            hook_handle, remove_hook, _spec = self._attach_steer_hook(
                steer_layer=int(steer_layer),
                steer_direction=steer_direction,
                steer_alpha=float(steer_alpha),
                steer_mode=str(steer_mode),
            )

        try:
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kw,
                )
        finally:
            if hook_handle is not None and remove_hook is not None:
                remove_hook(hook_handle)

        results: list[GenerateResult] = []
        pad_id = self.tokenizer.pad_token_id
        for i in range(len(prompts)):
            full_ids = out[i].tolist()
            # Drop left pads from the returned sequence if present.
            if pad_id is not None:
                while full_ids and full_ids[0] == pad_id:
                    full_ids.pop(0)
            plen = int(prompt_lens[i])
            # After stripping left pads, prompt should be the first plen tokens.
            prompt_ids = full_ids[:plen]
            comp_ids = full_ids[plen:]
            # Stop at first EOS in completion if present.
            eos = self.tokenizer.eos_token_id
            if eos is not None and eos in comp_ids:
                cut = comp_ids.index(eos)
                comp_ids = comp_ids[: cut + 1]
            comp_text = self.tokenizer.decode(comp_ids, skip_special_tokens=True)
            results.append(
                GenerateResult(
                    text=comp_text,
                    prompt_text=prompt_texts[i],
                    prompt_token_ids=list(prompt_ids),
                    completion_token_ids=list(comp_ids),
                    full_token_ids=list(prompt_ids) + list(comp_ids),
                    prompt_token_count=len(prompt_ids),
                    completion_token_count=len(comp_ids),
                )
            )
        return results
