def normalize_task_type(value: object) -> str:
    task_type = str(value or "").strip()
    for prefix in ("dense_captioning", "region_caption", "video_summary"):
        if task_type.startswith(prefix):
            return prefix
    return task_type or "general"


def record_task_types(record: dict) -> list[str]:
    raw = record.get("question_types")
    if isinstance(raw, list):
        values = raw
    elif raw:
        values = str(raw).split("|")
    else:
        values = [record.get("qa_type") or record.get("category") or "general"]
    result = []
    for item in values:
        task_type = normalize_task_type(item)
        if task_type not in result:
            result.append(task_type)
    return result or ["general"]


__all__ = ["normalize_task_type", "record_task_types"]
