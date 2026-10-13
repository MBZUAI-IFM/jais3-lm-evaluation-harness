import pytest

from lm_eval.models.litellm_llms import LiteLLMChatCompletion


def _model(model="anthropic/claude-haiku-4-5-20251001"):
    instance = object.__new__(LiteLLMChatCompletion)
    instance.model = model
    instance._max_gen_toks = 256
    return instance


@pytest.mark.parametrize(
    "max_gen_toks, expected_budget",
    [
        (8192, 2048),
        (64000, 16000),
    ],
)
def test_anthropic_thinking_budget_percentage(max_gen_toks, expected_budget):
    gen_kwargs = {
        "max_gen_toks": max_gen_toks,
        "reasoning_effort": "low",
        "temperature": 1,
        "budget_tokens_percentage": 0.25,
    }

    payload = _model()._create_payload(
        [{"role": "user", "content": "Solve this problem."}],
        generate=True,
        gen_kwargs=gen_kwargs,
    )

    assert payload["thinking"] == {
        "type": "enabled",
        "budget_tokens": expected_budget,
    }
    assert "reasoning_effort" not in payload
    assert "budget_tokens_percentage" not in payload
    assert gen_kwargs["budget_tokens_percentage"] == 0.25


@pytest.mark.parametrize("percentage", [0, 1, -0.25, 1.25, "invalid", float("nan")])
def test_anthropic_thinking_budget_percentage_rejects_invalid_values(percentage):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        _model()._create_payload(
            [{"role": "user", "content": "Hello"}],
            gen_kwargs={
                "max_gen_toks": 8192,
                "budget_tokens_percentage": percentage,
            },
        )


def test_thinking_budget_percentage_rejects_non_anthropic_models():
    with pytest.raises(ValueError, match="anthropic/"):
        _model("openai/gpt-5.6-luna")._create_payload(
            [{"role": "user", "content": "Hello"}],
            gen_kwargs={
                "max_gen_toks": 8192,
                "budget_tokens_percentage": 0.25,
            },
        )
