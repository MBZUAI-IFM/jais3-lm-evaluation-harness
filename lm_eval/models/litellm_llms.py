"""LiteLLM AI Gateway model backend for lm-evaluation-harness.

Provides a unified interface to 100+ LLM providers through the LiteLLM SDK.

Usage::

    lm_eval --model litellm \
        --model_args model=anthropic/claude-3-sonnet \
        --tasks hellaswag \
        --num_fewshot 0 \
        --apply_chat_template

See https://docs.litellm.ai/docs/providers for supported providers.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import TYPE_CHECKING

from lm_eval.api.model import GenerationWithMetadata
from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalChatCompletion


if TYPE_CHECKING:
    import asyncio


eval_logger = logging.getLogger(__name__)


@register_model("litellm", "litellm-chat", "litellm-chat-completions")
class LiteLLMChatCompletion(LocalChatCompletion):
    """LiteLLM chat completion model backend.

    Uses ``litellm.completion()`` to route requests to 100+ LLM providers.
    LiteLLM reads provider API keys from environment variables automatically
    (OPENAI_API_KEY, ANTHROPIC_API_KEY, COHERE_API_KEY, etc.).

    Model identifiers use provider prefixes::

        gpt-4o                              # OpenAI
        anthropic/claude-3-sonnet           # Anthropic
        gemini/gemini-pro                   # Google
        bedrock/anthropic.claude-3-sonnet   # AWS Bedrock
        azure/my-deployment                 # Azure OpenAI
    """

    @staticmethod
    def _apply_anthropic_thinking_budget_percentage(
        payload: dict, model: str, percentage: object
    ) -> dict:
        """Convert a relative Claude thinking budget into LiteLLM's nested form.

        ``lm_eval --gen_kwargs`` only supports flat ``key=value`` arguments, so
        callers cannot express Anthropic's nested ``thinking`` object there.
        ``budget_tokens_percentage`` is therefore a harness-only convenience
        argument. It is consumed here and never forwarded to LiteLLM.
        """
        if not model.startswith("anthropic/"):
            raise ValueError(
                "`budget_tokens_percentage` is only supported for models using "
                "the `anthropic/` provider prefix."
            )
        if "thinking" in payload:
            raise ValueError(
                "Pass either `thinking` or `budget_tokens_percentage`, not both."
            )

        try:
            percentage = float(percentage)
        except (TypeError, ValueError):
            raise ValueError(
                "`budget_tokens_percentage` must be a number strictly between 0 and 1."
            ) from None
        if not math.isfinite(percentage) or not 0 < percentage < 1:
            raise ValueError(
                "`budget_tokens_percentage` must be strictly between 0 and 1."
            )

        max_tokens_value = payload.get(
            "max_completion_tokens", payload.get("max_tokens")
        )
        if max_tokens_value is None:
            raise ValueError(
                "`budget_tokens_percentage` requires `max_gen_toks`, "
                "`max_tokens`, or `max_completion_tokens`."
            )
        max_tokens = int(max_tokens_value)
        if max_tokens <= 1024:
            raise ValueError(
                "Anthropic manual thinking requires `max_gen_toks` greater "
                "than the minimum 1024-token thinking budget."
            )
        budget_tokens = max(1024, int(max_tokens * percentage))
        budget_tokens = min(budget_tokens, max_tokens - 1)

        # Do not let LiteLLM remap reasoning_effort to its own fixed budget.
        payload.pop("reasoning_effort", None)
        payload["thinking"] = {
            "type": "enabled",
            "budget_tokens": budget_tokens,
        }
        eval_logger.info(
            "Using Anthropic manual thinking budget %d/%d tokens (%.2f%%).",
            budget_tokens,
            max_tokens,
            percentage * 100,
        )
        return payload

    def _create_payload(
        self,
        messages,
        generate=False,
        gen_kwargs=None,
        seed=1234,
        eos="<|endoftext|>",
        **kwargs,
    ) -> dict:
        gen_kwargs = copy.deepcopy(gen_kwargs) or {}
        budget_percentage = gen_kwargs.pop("budget_tokens_percentage", None)
        payload = super()._create_payload(
            messages,
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=seed,
            eos=eos,
            **kwargs,
        )
        if budget_percentage is not None:
            payload = self._apply_anthropic_thinking_budget_percentage(
                payload, self.model, budget_percentage
            )
        return payload

    def __init__(
        self,
        base_url=None,
        tokenizer_backend=None,
        tokenized_requests=None,
        **kwargs,
    ):
        try:
            import litellm
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "attempted to use 'litellm' LM type, but package `litellm` "
                "is not installed. Please install via "
                "`pip install 'lm-eval[litellm]'` or `pip install litellm`",
            ) from None

        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )
        self._litellm = litellm
        # Drop provider-unsupported params (e.g. seed on Azure AI) instead of erroring
        self._litellm.drop_params = True

    def model_call(
        self,
        messages,
        *,
        generate=True,
        gen_kwargs=None,
        **kwargs,
    ) -> dict | None:
        try:
            payload = self._create_payload(
                self.create_message(messages),
                generate=generate,
                gen_kwargs=copy.deepcopy(gen_kwargs),
                seed=self._seed,
                eos=self.eos_string,
                **kwargs,
            )
            payload.setdefault("timeout", self.timeout)
            response = self._litellm.completion(**payload)
            return response.model_dump()
        except Exception as e:
            eval_logger.error(
                "LiteLLM completion failed for model %s: %s", self.model, e
            )
            raise

    @staticmethod
    def parse_generation_metadata(outputs, **kwargs) -> list[dict]:
        """Keep the complete serialized LiteLLM response for each generation."""
        if not isinstance(outputs, list):
            outputs = [outputs]

        metadata = []
        for output in outputs:
            # Stay aligned with LocalChatCompletion.parse_generations(), including
            # its one-result fallback for blocked or malformed responses.
            response_count = len(LocalChatCompletion.parse_generations(output))
            metadata.extend([output] * response_count)
        return metadata

    async def amodel_call(
        self,
        session,
        sem: asyncio.Semaphore,
        messages,
        *,
        generate=True,
        cache_keys=None,
        ctxlens: list[int] | None = None,
        gen_kwargs: dict | None = None,
        **kwargs,
    ) -> (
        list[str]
        | list[GenerationWithMetadata]
        | list[tuple[float, bool]]
        | None
    ):
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=copy.deepcopy(gen_kwargs),
            seed=self._seed,
            **kwargs,
        )
        payload.setdefault("timeout", self.timeout)
        cache_method = "generate_until" if generate else "loglikelihood"
        acquired = await sem.acquire()
        try:
            response = await self._litellm.acompletion(**payload)
            outputs = response.model_dump()
            answers = (
                self.parse_generations(outputs=outputs)
                if generate
                else self.parse_logprobs(
                    outputs=outputs, tokens=messages, ctxlens=ctxlens
                )
            )
            if generate:
                generation_metadata = self.parse_generation_metadata(outputs=outputs)
                if len(generation_metadata) != len(answers):
                    raise ValueError(
                        "Parsed generation metadata must have the same length as "
                        "the parsed generations."
                    )
                answers = [
                    GenerationWithMetadata(answer, metadata)
                    for answer, metadata in zip(
                        answers, generation_metadata, strict=True
                    )
                ]
            if cache_keys:
                for res, cache in zip(answers, cache_keys, strict=True):
                    if res is not None:
                        self.cache_hook.add_partial(cache_method, cache, res)
            return answers
        except BaseException as e:
            eval_logger.error("LiteLLM async completion failed: %r, retrying.", e)
            raise
        finally:
            if acquired:
                sem.release()

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood is not supported for LiteLLM chat completions. "
            "Most providers do not return logprobs via the chat API. "
            "Use generate_until tasks (e.g. --tasks gsm8k) instead."
        )

    def loglikelihood_rolling(self, requests, **kwargs):
        raise NotImplementedError(
            "Rolling loglikelihood is not supported for LiteLLM chat "
            "completions. Use generate_until tasks instead."
        )
