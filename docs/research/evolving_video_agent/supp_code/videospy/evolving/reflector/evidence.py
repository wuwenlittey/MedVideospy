"""Readable trajectories and original observations used by the Reflector."""
import re
from videospy.core.evidence import frame_timestamps


def _readable_content(value):
    if isinstance(value, str):
        return re.sub(r"data:image/[^\s\"']+", "<image omitted>", value)
    if isinstance(value, list):
        return [_readable_content(item) for item in value
                if not isinstance(item, dict) or item.get("type") not in {"image_url", "image", "input_image"}]
    if isinstance(value, dict):
        return {key: _readable_content(item) for key, item in value.items()
                if key not in {"model_call", "provider_specific_fields", "image_url"}}
    return value


def trajectory_for_reflection(value: dict) -> dict:
    """Preserve the entire readable conversation, without duplicated model requests."""
    if not isinstance(value.get("messages"), list):
        raise ValueError("Trajectory must contain a messages list")
    fields = {"role", "content", "reasoning_content", "reasoning", "tool_calls", "tool_call_id", "name"}
    messages = []
    for message in value["messages"]:
        if not isinstance(message, dict):
            raise ValueError("Trajectory messages must be objects")
        messages.append({key: _readable_content(item) for key, item in message.items() if key in fields})
    return {"question": value.get("question", ""), "messages": messages,
            "total_steps": value.get("total_steps"), "finish_reason": value.get("finish_reason"),
            "final_answer": value.get("final_answer", "")}


def repeated_frame_observations(trajectory):
    """Record frame reuse from tool-generated headers, without judging its utility."""
    seen, repeated = {}, []
    for index, message in enumerate(trajectory.get('messages', [])):
        frames = tuple(frame_timestamps(message))
        if not frames:
            continue
        if frames in seen:
            repeated.append({'earlier_message_index': seen[frames], 'message_index': index,
                             'actual_timestamps': list(frames)})
        seen[frames] = index
    return repeated


MAX_REFERENCES = 1
MAX_QUOTE_CHARS = 800
MEMORY_SPAN_CHARS = 400


def build_evidence(trajectory, source_sha256):
    """Keep original tool text for extraction and local quote validation."""
    messages = []
    for index, message in enumerate(trajectory.get("messages") or []):
        role = message.get("role")
        if role != "tool":
            continue
        content = message.get("content")
        content = content if isinstance(content, str) else ""
        # Image payloads are not readable observations.
        content = re.sub(r"data:image/[^\s\"']+", "<image omitted>", content)
        value = {"message_index": index, "role": role, "content": content}
        value.update(name=str(message.get("name") or ""),
                     tool_call_id=str(message.get("tool_call_id") or ""),
                     failed="execution failed" in content.casefold()
                     or "invalid function name" in content.casefold()
                     or content.startswith("Duplicate `"))
        messages.append(value)
    return {"source_sha256": source_sha256, "messages": messages}


def validate_reference_shape(refs):
    if not isinstance(refs, list) or not 1 <= len(refs) <= MAX_REFERENCES:
        raise ValueError("Memory evidence_refs requires exactly one original tool reference.")
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {"tool_call_id", "quote"}:
            raise ValueError("Each Memory reference requires tool_call_id and quote only.")
        if not isinstance(ref["tool_call_id"], str) or not ref["tool_call_id"].strip():
            raise ValueError("Memory reference requires a non-empty tool_call_id.")
        quote = ref["quote"]
        if not isinstance(quote, str) or not quote.strip() or len(quote) > MAX_QUOTE_CHARS:
            raise ValueError("Memory quote must be non-empty and at most 800 characters; narrow the fact.")


def tool_observations(source):
    """Expose complete successful observations without silently truncating text."""
    result = []
    for message in source["messages"]:
        if message["role"] != "tool" or message["failed"] or not message["content"].strip():
            continue
        result.append({"tool_call_id": message["tool_call_id"], "tool": message["name"],
                       "content": message["content"]})
    return result


def observation_spans(source):
    """Index every character of successful observations without rewriting it."""
    spans = []
    for index, observation in enumerate(tool_observations(source)):
        text, start = observation['content'], 0
        # Mark explicit answer-analysis sections, preserving their text for context.
        # This is a structural exclusion, not a semantic verification of other text.
        headings = list(re.finditer(r'(?m)^[ \t]*(?:#{1,6}[ \t]+[^\n]+|\*\*[^\n]+\*\*:?|[A-Za-z][A-Za-z /-]*:)[ \t]*$', text))
        excluded = []
        for n, heading in enumerate(headings):
            title = heading.group().strip(' \t*#:\r').casefold()
            if title in {'conclusion', 'final answer', 'answer', 'categorization analysis',
                         'option analysis', 'option comparison', 'answer selection'}:
                excluded.append((heading.start(), headings[n + 1].start() if n + 1 < len(headings) else len(text)))
        while start < len(text):
            section_end = next((h.start() for h in headings if h.start() > start), len(text))
            end = min(start + MEMORY_SPAN_CHARS, section_end)
            if end < len(text):
                boundary = max(text.rfind('\n', start, end), text.rfind('. ', start, end))
                if boundary > start:
                    end = boundary + 1
            complete = (start == 0 or text[start - 1] in '\n.!?。！？') and (end == len(text) or text[end - 1] in '\n.!?。！？')
            spans.append({'id': f'o{index}:{start}', 'tool': observation['tool'],
                          'tool_call_id': observation['tool_call_id'], 'start': start,
                          'end': end, 'text': text[start:end],
                          'memory_eligible': complete and not any(start < b and end > a for a, b in excluded)})
            start = end
    return spans


def resolve_evidence_ids(source, ids):
    spans = {span['id']: span for span in observation_spans(source)}
    if not isinstance(ids, list) or not 1 <= len(ids) <= MAX_REFERENCES:
        raise ValueError('Memory requires exactly one evidence_ids entry')
    if any(not isinstance(i, str) or i not in spans for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('evidence_ids must be distinct IDs from the supplied observations')
    if any(not spans[i]['memory_eligible'] for i in ids):
        raise ValueError('Select only memory_eligible observations; answer-analysis sections and cut sentences cannot become Memory')
    refs = [{'tool_call_id': span['tool_call_id'], 'quote': span['text']} for i, span in spans.items() if i in ids]
    local_memory_evidence(source, refs)
    return refs


def local_memory_evidence(source, refs):
    """Validate quotes against successful observations in this sample."""
    validate_reference_shape(refs)
    observations = {m["tool_call_id"]: m for m in source["messages"] if m["role"] == "tool"}
    for ref in refs:
        message = observations.get(ref["tool_call_id"])
        if message is None or message["failed"]:
            raise ValueError(f"Memory reference {ref['tool_call_id']} is unknown or a failed tool call.")
        start = message["content"].find(ref["quote"])
        if start < 0:
            raise ValueError(f"Memory quote is not a contiguous original observation in {ref['tool_call_id']}.")
