from __future__ import annotations

from datasets import load_dataset
from huggingface_hub import snapshot_download

from .config import MATH_DATASETS
from .lcb_data import load_lcb_v6
from .training_data import load_math_cot_20k


MODELS = (
    ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
)
MATH_REVISIONS = {
    "aime24": "ea5b061c3e8039dc9858defaafc407d04b995e9f+29d5d31e9b46e215fc24d9b2a3047506823dd101",
    "aime25": "c94da77eb22bbd6439e62a323bec18493a421302",
    "aime26": "d2de22f3c656b4f56cf8981212186377d1e23bc3",
    "hmmt25": "6fdc4277120810ff75aa22d2d5489b91f7a262a1",
}


def main() -> None:
    for model, revision in MODELS:
        path = snapshot_download(model, revision=revision)
        print(f"cached model {model}@{revision}: {path}", flush=True)
    for alias, revision in MATH_REVISIONS.items():
        spec = MATH_DATASETS[alias]
        components = spec.get("components")
        if components:
            datasets = [
                load_dataset(path, split=spec["split"], revision=component_revision,
                             trust_remote_code=True)
                for (_, path), component_revision in zip(
                    components, revision.split("+"), strict=True
                )
            ]
            count = sum(len(dataset) for dataset in datasets)
        else:
            count = len(load_dataset(
                spec["path"], split=spec["split"], revision=revision,
                trust_remote_code=True,
            ))
        if count != spec["expected_count"]:
            raise RuntimeError(f"{alias}: expected {spec['expected_count']}, found {count}")
        print(f"cached {alias}: {count} rows", flush=True)
    lcb = load_lcb_v6("0fe84c3912ea0c4d4a78037083943e8f0c4dd505")
    if len(lcb) != 175:
        raise RuntimeError(f"LCB v6: expected 175, found {len(lcb)}")
    training = load_math_cot_20k()["train"]
    print(f"cached Math-CoT-20k: {len(training)} rows", flush=True)


if __name__ == "__main__":
    main()
