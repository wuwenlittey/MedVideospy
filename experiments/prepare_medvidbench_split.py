import argparse
import json
import random
from collections import Counter
from pathlib import Path


DEFAULT_METADATA_PATH = Path(
    "data/MedVidBench/MedVidU_ECCV2026_TrainVal/"
    "medvidu_eccv2026_trainval.json"
)
DEFAULT_OUTPUT_DIR = Path("experiments/data_splits/main/medvidbench")
DEFAULT_TEST_SAMPLES = 1124
DEFAULT_SEED = 42
DEFAULT_TRIALS = 5000
SEED_STEP = 104729


def medvid_group(item: dict, index: int) -> str:
    metadata = item.get("metadata") or {}
    video_id = metadata.get("video_id")
    if not video_id:
        video = item.get("video") or []
        video_id = Path(video[0]).parent.name if video else item.get("id", index)
    source = item.get("data_source") or item.get("dataset_name")
    return f"{source}:{video_id}"


def _group_samples(items: list[dict]) -> list[tuple[str, list[int]]]:
    grouped: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        grouped.setdefault(medvid_group(item, index), []).append(index)
    return list(grouped.items())


def _select_exact_samples(
    groups: list[tuple[str, list[int]]], target_samples: int, seed: int
) -> list[int] | None:
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    previous: list[tuple[int, int] | None] = [None] * (target_samples + 1)
    previous[0] = (-1, -1)
    for group_index in order:
        size = len(groups[group_index][1])
        for sample_count in range(target_samples - size, -1, -1):
            candidate = sample_count + size
            if previous[sample_count] is not None and previous[candidate] is None:
                previous[candidate] = (sample_count, group_index)
    if previous[target_samples] is None:
        return None

    selected = []
    sample_count = target_samples
    while sample_count:
        prior_count, group_index = previous[sample_count]
        selected.append(group_index)
        sample_count = prior_count
    return selected


def _distribution_score(
    items: list[dict],
    groups: list[tuple[str, list[int]]],
    selected_groups: list[int],
    target_samples: int,
) -> float:
    selected = [
        index for group_index in selected_groups for index in groups[group_index][1]
    ]
    source_total = Counter(item["dataset_name"] for item in items)
    task_total = Counter(item["qa_type"] for item in items)
    source_selected = Counter(items[index]["dataset_name"] for index in selected)
    task_selected = Counter(items[index]["qa_type"] for index in selected)
    fraction = target_samples / len(items)

    def mean_relative_error(total: Counter, observed: Counter) -> float:
        return sum(
            abs(observed[key] - count * fraction) / (count * fraction)
            for key, count in total.items()
        ) / len(total)

    missing = sum(source_selected[key] == 0 for key in source_total)
    missing += sum(task_selected[key] == 0 for key in task_total)
    expected_groups = round(len(groups) * fraction)
    group_error = abs(len(selected_groups) - expected_groups) / expected_groups
    return (
        missing * 100
        + 2 * mean_relative_error(source_total, source_selected)
        + mean_relative_error(task_total, task_selected)
        + 0.25 * group_error
    )


def create_split(
    items: list[dict],
    test_samples: int = DEFAULT_TEST_SAMPLES,
    seed: int = DEFAULT_SEED,
    trials: int = DEFAULT_TRIALS,
) -> tuple[list[int], list[int]]:
    groups = _group_samples(items)
    best: tuple[float, list[int]] | None = None
    for trial in range(trials):
        selected_groups = _select_exact_samples(
            groups, test_samples, seed + trial * SEED_STEP
        )
        if selected_groups is None:
            continue
        score = _distribution_score(items, groups, selected_groups, test_samples)
        if best is None or score < best[0]:
            best = (score, selected_groups)
    if best is None:
        raise ValueError(f"Cannot select exactly {test_samples} grouped samples.")

    test_groups = {groups[index][0] for index in best[1]}
    test = [
        index
        for group, indices in groups
        if group in test_groups
        for index in indices
    ]
    train = [
        index
        for group, indices in groups
        if group not in test_groups
        for index in indices
    ]
    return train, test


def audit_split(items: list[dict], train: list[int], test: list[int]) -> dict:
    train_groups = {medvid_group(items[index], index) for index in train}
    test_groups = {medvid_group(items[index], index) for index in test}

    def counts(indices: list[int], field: str) -> dict[str, int]:
        return dict(sorted(Counter(items[index][field] for index in indices).items()))

    return {
        "samples": {"train": len(train), "test": len(test)},
        "groups": {
            "train": len(train_groups),
            "test": len(test_groups),
            "overlap": len(train_groups & test_groups),
        },
        "sample_overlap": len(set(train) & set(test)),
        "complete": set(train) | set(test) == set(range(len(items))),
        "sources": {
            "train": counts(train, "dataset_name"),
            "test": counts(test, "dataset_name"),
        },
        "tasks": {
            "train": counts(train, "qa_type"),
            "test": counts(test, "qa_type"),
        },
    }


def _write(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the fixed grouped and stratified MedVidBench split."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    items = json.loads(args.metadata.read_text(encoding="utf-8"))
    train, test = create_split(items, args.test_samples, args.seed, args.trials)
    audit = audit_split(items, train, test)
    if audit["groups"]["overlap"] or audit["sample_overlap"] or not audit["complete"]:
        raise AssertionError(
            "Generated MedVidBench split is not disjoint and complete."
        )
    if set(audit["sources"]["test"]) != set(audit["sources"]["train"]):
        raise AssertionError(
            "Generated test split does not cover every source dataset."
        )
    if set(audit["tasks"]["test"]) != set(audit["tasks"]["train"]):
        raise AssertionError("Generated test split does not cover every task variant.")
    _write(args.output_dir / "train.json", train)
    _write(args.output_dir / "test.json", test)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
