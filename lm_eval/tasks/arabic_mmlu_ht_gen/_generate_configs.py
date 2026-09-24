"""Generate YAML configs for every human-translated Arabic MMLU subject."""

from __future__ import annotations

from pathlib import Path


SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


TASK_TEMPLATE = """\
include: _default_template_yaml
task: arabic_mmlu_ht_{subject}
task_alias: "Arabic MMLU-HT - {alias}"
dataset_name: {subject}
"""

GROUP_HEADER = """\
group: arabic_mmlu_ht
group_alias: Arabic MMLU-HT
task:
"""

GROUP_FOOTER = """\
aggregate_metric_list:
  - metric: acc_norm
    aggregation: mean
    weight_by_size: true
metadata:
  version: 1.0
"""


def main() -> None:
    task_dir = Path(__file__).resolve().parent
    for subject in SUBJECTS:
        alias = subject.replace("_", " ").title()
        path = task_dir / f"arabic_mmlu_ht_{subject}.yaml"
        path.write_text(
            TASK_TEMPLATE.format(subject=subject, alias=alias), encoding="utf-8"
        )

    task_lines = "".join(
        f"  - arabic_mmlu_ht_{subject}\n" for subject in SUBJECTS
    )
    (task_dir / "_arabic_mmlu_ht.yaml").write_text(
        GROUP_HEADER + task_lines + GROUP_FOOTER,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
