# EDRAC

An `lm-evaluation-harness` implementation of
[EDRAC](https://huggingface.co/datasets/MBZUAI/EDRAC), a generative reading
comprehension benchmark covering five Arabic dialects.

## Tasks

The `edrac` group selects five independently reported dialect tasks:

| Task | Dialect | Hub code | QA pairs |
| --- | --- | --- | ---: |
| `edrac_egy` | Egyptian | `egy` | 999 |
| `edrac_mor` | Moroccan | `mor` | 997 |
| `edrac_ksa` | Saudi | `sau` | 999 |
| `edrac_syr` | Syrian | `syr` | 984 |
| `edrac_uae` | Emirati | `ema` | 998 |

The Hub stores one row per passage with up to ten question/answer pairs. This
task expands those 499 passage rows into 4,977 independent evaluation samples.
For every sample, the model receives the full unchanged passage and one
question; questions from the same passage are not evaluated as a conversation.

The group intentionally has no combined metric. This follows the official
EDRAC evaluation, which reports each dialect separately.

## Prompt and generation

With `--apply_chat_template`, the prompt is rendered as two messages:

```text
system: Given the following passage, answer the accompanying questions.

user:   Passage: {passage}

        Question: {question}

        Answer:
```

Generation is zero-shot and greedy with a 256-token limit, temperature 0,
top-p 1, and repetition penalty 1.

## Metric

Each dialect reports mean ROUGE-L F1 using whitespace tokenization, on a
0--100 scale. Blank outputs receive a score of zero. The mDeBERTa and
CAMeLBERT BERTScore metrics from the paper are intentionally not included.

## Dataset access

EDRAC is gated and evaluation-only. Accept its terms on the Hugging Face Hub,
then authenticate using the standard Hugging Face mechanisms:

```bash
hf auth login
# or
export HF_TOKEN=hf_...
```

No token or machine-specific credential path is stored in this task package.

## Running

From an `lm-evaluation-harness` environment:

```bash
lm-eval run \
  --model hf \
  --model_args pretrained=<model> \
  --tasks edrac \
  --include_path /scratch/mostafa.awad/BGM/EDRAC \
  --apply_chat_template \
  --batch_size auto
```

To run one dialect, replace `--tasks edrac` with a subtask such as
`--tasks edrac_egy`.

## References

- Dataset: <https://huggingface.co/datasets/MBZUAI/EDRAC>
- Official evaluation repository: <https://github.com/mbzuai-nlp/EDRAC>

