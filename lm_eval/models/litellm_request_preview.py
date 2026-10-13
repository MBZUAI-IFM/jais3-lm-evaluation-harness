"""Safe, one-request preview backend for LiteLLM chat completions.

This backend is intentionally separate from the normal LiteLLM backend. It
prepares the first evaluation request with an invalid credential, intercepts it
at the HTTP transport boundary, prints its exact final URL and the provider
request emitted by LiteLLM's DEBUG logger, and supplies local dummy completions
for the rest of the evaluation. The intercepted request is never transmitted.

Usage::

    lm_eval --model litellm-request-preview \
        --model_args model=openai/gpt-5.6-luna,base_url=https://api.openai-next.com/v1 \
        --gen_kwargs max_gen_toks=25,reasoning_effort=none,temperature=1 \
        --tasks mmlu_gen --limit 1 --apply_chat_template

Do not use ``--use_cache`` with this model: a response-cache hit can bypass the
model before it has an opportunity to print a request.
"""

from __future__ import annotations

import copy
import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO

from lm_eval.api.registry import register_model
from lm_eval.models.litellm_llms import LiteLLMChatCompletion


_DUMMY_API_KEY = "lm-eval-intentionally-invalid-preview-key"
_PREVIEW_TIMEOUT_SECONDS = 10
_REQUEST_LOG_MARKER = "POST Request Sent from LiteLLM:"
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PROVIDER_API_KEY_ENV_VARS = {
    "openai": ("OPENAI_API_KEY", "VECTRUST_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY", "VECTRUST_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY", "VECTRUST_API_KEY"),
}


class _LiteLLMRequestLogCapture(logging.Handler):
    """Collect request blocks from LiteLLM without retaining other DEBUG logs."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.requests: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if _REQUEST_LOG_MARKER not in message:
            return
        request = message.split(_REQUEST_LOG_MARKER, 1)[1].strip()
        request = _ANSI_ESCAPE_RE.sub("", request).strip()
        if request:
            self.requests.append(request)


@contextmanager
def _temporary_environment(overrides: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _preserve_litellm_logger() -> Iterator[logging.Logger]:
    logger = logging.getLogger("LiteLLM")
    previous_handlers = logger.handlers[:]
    previous_level = logger.level
    previous_propagate = logger.propagate
    previous_disabled = logger.disabled
    try:
        yield logger
    finally:
        logger.handlers = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
        logger.disabled = previous_disabled


@contextmanager
def _capture_litellm_request_logs() -> Iterator[_LiteLLMRequestLogCapture]:
    with _preserve_litellm_logger() as logger:
        capture = _LiteLLMRequestLogCapture()
        logger.handlers = [capture]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.disabled = False
        yield capture


@contextmanager
def _drop_unsupported_litellm_params(litellm) -> Iterator[None]:
    previous = litellm.drop_params
    try:
        # Task-level generation settings can include fields unsupported by the
        # selected provider (for example, MMLU stop sequences on GPT-5). Let
        # LiteLLM perform its normal provider-aware transformation so the log
        # contains the request that can actually be sent.
        litellm.drop_params = True
        yield
    finally:
        litellm.drop_params = previous


@contextmanager
def _capture_final_http_urls() -> Iterator[list[str]]:
    """Capture prepared sync HTTP requests and stop them before transmission."""
    import httpx

    urls: list[str] = []
    previous_send = httpx.Client.send

    def capture_send(client, request, *args, **kwargs):
        del client, args, kwargs
        urls.append(str(request.url))
        raise httpx.ConnectError(
            "LiteLLM request preview intercepted the request before transmission.",
            request=request,
        )

    try:
        httpx.Client.send = capture_send
        yield urls
    finally:
        httpx.Client.send = previous_send


@register_model("litellm-request-preview")
class LiteLLMRequestPreview(LiteLLMChatCompletion):
    """Print the first real LiteLLM request and return dummy completions."""

    def __init__(
        self,
        *,
        api_key=None,
        num_concurrent=1,
        max_retries=3,
        timeout=300,
        **kwargs,
    ) -> None:
        del api_key, num_concurrent, max_retries, timeout

        model = kwargs.get("model") or kwargs.get("pretrained")
        if not isinstance(model, str) or not model:
            raise ValueError(
                "LiteLLM request preview requires "
                "`model=<provider>/<model>` in --model_args."
            )
        self._preview_provider = model.split("/", 1)[0].lower()
        if self._preview_provider not in _PROVIDER_API_KEY_ENV_VARS:
            supported = ", ".join(sorted(_PROVIDER_API_KEY_ENV_VARS))
            raise ValueError(
                "LiteLLM request preview supports these provider prefixes: "
                f"{supported}. Received model={model!r}."
            )

        self._request_printed = False
        environment = self._preview_environment()
        # LiteLLM reads LITELLM_LOG while it is imported/configured by the parent.
        # Preserve both process environment and logger state around that import.
        with (
            _temporary_environment(environment),
            _preserve_litellm_logger(),
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()),
        ):
            super().__init__(
                num_concurrent=1,
                max_retries=1,
                timeout=_PREVIEW_TIMEOUT_SECONDS,
                **kwargs,
            )

    def _preview_environment(self) -> dict[str, str]:
        environment = {
            "LITELLM_LOG": "DEBUG",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        }
        environment.update(
            {
                variable: _DUMMY_API_KEY
                for variable in _PROVIDER_API_KEY_ENV_VARS[
                    self._preview_provider
                ]
            }
        )
        return environment

    def _dummy_response(self) -> dict:
        return {
            "id": "chatcmpl-litellm-request-preview",
            "object": "chat.completion",
            "created": 0,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "[LiteLLM request preview dummy response]",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    def model_call(
        self,
        messages,
        *,
        generate=True,
        gen_kwargs=None,
        **kwargs,
    ) -> dict:
        if self._request_printed:
            return self._dummy_response()

        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=copy.deepcopy(gen_kwargs),
            seed=self._seed,
            # The preview backend has no tokenizer, so resolving ``eos_string``
            # would only emit a warning. An explicitly supplied EOS value is
            # still honored.
            eos=self._eos_string,
            **kwargs,
        )
        # Empty stop lists have no effect, but LiteLLM still validates the field
        # and rejects it for models that do not support stop sequences.
        if not payload.get("stop"):
            payload.pop("stop", None)
        if self.base_url:
            payload["api_base"] = self.base_url
        payload["api_key"] = _DUMMY_API_KEY
        payload["num_retries"] = 0
        payload["timeout"] = _PREVIEW_TIMEOUT_SECONDS

        error = None
        succeeded = False
        with (
            _temporary_environment(self._preview_environment()),
            _drop_unsupported_litellm_params(self._litellm),
            _capture_litellm_request_logs() as capture,
            _capture_final_http_urls() as final_http_urls,
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()),
        ):
            try:
                self._litellm.completion(**payload)
                succeeded = True
            except Exception as exc:
                error = exc

        if not capture.requests:
            error_detail = (
                f" LiteLLM failed before request logging with "
                f"{type(error).__name__}: {error}"
                if error is not None
                else ""
            )
            raise RuntimeError(
                "LiteLLM DEBUG logging did not emit the expected "
                f"{_REQUEST_LOG_MARKER!r} request block.{error_detail}"
            ) from error

        if succeeded:
            raise RuntimeError(
                "The LiteLLM request preview unexpectedly bypassed its HTTP "
                "interceptor. Stopping immediately to avoid additional potentially "
                "billable requests."
            )

        if not final_http_urls:
            raise RuntimeError(
                "LiteLLM emitted a request log, but the preview could not capture "
                "the final HTTP URL at the transport boundary."
            ) from error

        print("LiteLLM provider request (first evaluation request):", flush=True)
        print(f"Final HTTP URL: {final_http_urls[0]}", flush=True)
        print(capture.requests[-1], flush=True)
        self._request_printed = True

        return self._dummy_response()
