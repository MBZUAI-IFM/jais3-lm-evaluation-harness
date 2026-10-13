import asyncio

import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM, CachingLM, GenerationWithMetadata, hash_args
from lm_eval.models.litellm_llms import LiteLLMChatCompletion


def _request(prompt="prompt"):
    return Instance(
        request_type="generate_until",
        doc={},
        arguments=(prompt, {"do_sample": False}),
        idx=0,
        metadata=("task", 0, 1),
    )


class _MetadataLM(LM):
    def __init__(self, text="answer", metadata=None):
        super().__init__()
        self.text = text
        self.metadata = metadata or {"usage": {"total_tokens": 3}}
        self.calls = 0

    def generate_until(self, requests):
        self.calls += 1
        for request in requests:
            request.generation_metadata.append(self.metadata)
        return [self.text for _ in requests]

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError


class _PartialFailureLM(_MetadataLM):
    def generate_until(self, requests):
        self.calls += 1
        for request in requests:
            self.cache_hook.add_partial(
                "generate_until",
                request.args,
                GenerationWithMetadata(self.text, self.metadata),
            )
        raise RuntimeError("simulated interrupted run")


class _CompletionResponse:
    def __init__(self, output):
        self.output = output

    def model_dump(self):
        return self.output


class _AsyncLiteLLM:
    def __init__(self, output):
        self.output = output

    async def acompletion(self, **kwargs):
        return _CompletionResponse(self.output)


class _RecordingCacheHook:
    def __init__(self):
        self.entries = []

    def add_partial(self, attr, request, result):
        self.entries.append((attr, request, result))


def test_litellm_partial_cache_keeps_complete_metadata():
    output = {
        "id": "response-id",
        "choices": [
            {
                "index": 0,
                "message": {"content": "answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }
    model = object.__new__(LiteLLMChatCompletion)
    model._seed = 1234
    model._litellm = _AsyncLiteLLM(output)
    model._create_payload = lambda *args, **kwargs: {}
    model.create_message = lambda messages: messages
    model.cache_hook = _RecordingCacheHook()
    cache_key = ("prompt", {"do_sample": False})

    answers = asyncio.run(
        model.amodel_call(
            session=None,
            sem=asyncio.Semaphore(1),
            messages=[{"role": "user", "content": "prompt"}],
            cache_keys=[cache_key],
        )
    )

    expected = GenerationWithMetadata("answer", output)
    assert answers == [expected]
    assert model.cache_hook.entries == [("generate_until", cache_key, expected)]


def test_completed_generation_caches_and_restores_metadata(tmp_path):
    cache_path = str(tmp_path / "responses.db")
    metadata = {"usage": {"prompt_tokens": 1, "completion_tokens": 2}}
    producer = _MetadataLM(metadata=metadata)
    cached_producer = CachingLM(producer, cache_path)

    assert cached_producer.generate_until([_request()]) == ["answer"]
    cached_value = cached_producer.dbdict[
        hash_args("generate_until", _request().args)
    ]
    assert cached_value == GenerationWithMetadata("answer", metadata)
    cached_producer.dbdict.close()

    consumer = _MetadataLM(text="must not be called")
    cached_consumer = CachingLM(consumer, cache_path)
    request = _request()

    assert cached_consumer.generate_until([request]) == ["answer"]
    assert request.generation_metadata == [metadata]
    assert consumer.calls == 0
    cached_consumer.dbdict.close()


def test_partial_cache_survives_interrupted_run_with_metadata(tmp_path):
    cache_path = str(tmp_path / "responses.db")
    metadata = {"usage": {"total_tokens": 3}}
    producer = _PartialFailureLM(metadata=metadata)
    cached_producer = CachingLM(producer, cache_path)

    with pytest.raises(RuntimeError, match="interrupted run"):
        cached_producer.generate_until([_request()])
    cached_producer.dbdict.close()

    consumer = _MetadataLM(text="must not be called")
    cached_consumer = CachingLM(consumer, cache_path)
    request = _request()

    assert cached_consumer.generate_until([request]) == ["answer"]
    assert request.generation_metadata == [metadata]
    assert consumer.calls == 0
    cached_consumer.dbdict.close()


def test_legacy_string_cache_remains_compatible(tmp_path):
    cache_path = str(tmp_path / "responses.db")
    consumer = _MetadataLM(text="must not be called")
    cached_consumer = CachingLM(consumer, cache_path)
    request = _request()
    cached_consumer.dbdict[hash_args("generate_until", request.args)] = "legacy"

    assert cached_consumer.generate_until([request]) == ["legacy"]
    assert request.generation_metadata == []
    assert consumer.calls == 0
    cached_consumer.dbdict.close()
