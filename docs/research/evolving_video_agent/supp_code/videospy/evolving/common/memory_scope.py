"""Source identity and availability checks shared by training and inference."""
from bisect import bisect_right
from pathlib import PurePosixPath
import math
import re

from ..storage.artifacts import hash_json


def case_question(question):
    """Index transferable problems without answer options or runtime boilerplate."""
    return re.split(r'\n\s*(?:\(?A\)[ \t]|A\.[ \t]|Current query time:|Output requirement:)',
                    question, maxsplit=1)[0].strip()


def egolife_seconds(query_time):
    if not isinstance(query_time, dict):
        return None
    day = re.fullmatch(r"DAY([1-9]\d*)", str(query_time.get("date", "")).upper())
    clock = str(query_time.get("time", ""))
    if not day or not re.fullmatch(r"\d{8}", clock):
        return None
    hour, minute, second, fraction = (int(clock[i:i + 2]) for i in range(0, 8, 2))
    if hour >= 24 or minute >= 60 or second >= 60:
        return None
    return (int(day[1]) - 1) * 86400 + hour * 3600 + minute * 60 + second + fraction / 100


def egolife_source_id(video):
    # The current EgoLife adapter supports the A1_JAKE timeline only. Paths may
    # differ between training and evaluation hosts; arbitrary roots are unknown.
    parts = str(video or "").replace("\\", "/").upper().split("/")
    return "egolife:A1_JAKE" if "A1_JAKE" in parts else None


def source_context(benchmark, video, query_time=None):
    result = {"benchmark": benchmark, "video": str(video or ""), "query_time": query_time}
    if benchmark == "egolife":
        result.update(source_id=egolife_source_id(video), available_at=egolife_seconds(query_time))
    else:
        result["source_id"] = f"{benchmark}:{video}" if video else None
    return result


def medvidbench_source_context(sample):
    """MedVidBench timestamps address the supplied frames, not the full video."""
    metadata = sample["metadata"]
    frames, fps = sample["video"], float(metadata["fps"])
    if not isinstance(frames, list) or not frames or not math.isfinite(fps) or fps <= 0:
        raise ValueError("MedVidBench source requires frames and a positive finite FPS")
    video = metadata.get("video_id", "")
    dataset = sample.get("data_source", sample.get("dataset_name", ""))
    frame_ids = []
    for frame in frames:
        parts = PurePosixPath(str(frame).replace('\\', '/')).parts
        # Keep the path within the video when available, including nested clips.
        frame_ids.append('/'.join(parts[parts.index(str(video)):]) if str(video) in parts else parts[-1])
    return {"benchmark": "medvidbench", "video": str(video), "data_source": dataset,
            "source_id": "medvidbench:" + hash_json([dataset, video, frame_ids, fps])[:24],
            "clip": {"first_frame": frames[0], "last_frame": frames[-1],
                     "frame_count": len(frames), "fps": fps},
            "time_basis": "seconds from the start of this frame sequence"}


def memory_exclusion(metadata, context):
    source = metadata.get('source') or {}
    context = context or {}
    if not source.get('source_id') or not context.get('source_id'):
        return 'unknown_source'
    is_case = metadata.get('kind') == 'reasoning_case'
    if source.get('benchmark') != context.get('benchmark') or (not is_case and source['source_id'] != context['source_id']):
        return 'different_source'
    if is_case and source.get('sample_id') == context.get('sample_id') and context.get('sample_id') is not None:
        return 'same_sample'
    if source.get('benchmark') == 'egolife':
        before, now = source.get('available_at'), context.get('available_at')
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                   for v in (before, now)):
            return 'unknown_availability'
        if before > now:
            return 'future_source'
        # Numeric coordinates are unsafe across different/missing timeline maps.
        time = metadata.get('time') or {}
        if not is_case and (metadata.get('evidence_frames') or any(time.get(k) is not None for k in ('start_seconds', 'end_seconds'))):
            if not source.get('timeline_id') or source['timeline_id'] != context.get('timeline_id'):
                return 'incomparable_timeline'
    return None


def memory_available(metadata, context):
    return memory_exclusion(metadata, context) is None


def timeline_wall_seconds(mapping, seconds):
    """Inverse of the reader's concatenated timeline, using the same boundary rule."""
    segments = (mapping or {}).get('segments', [])
    if not segments or not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds):
        return None
    index = bisect_right([s['start'] for s in segments], seconds) - 1
    if index < 0:
        return None
    segment = segments[index]
    offset = seconds - segment['start']
    if not 0 <= offset <= segment['duration']:
        return None
    return segment['wall_start'] + offset


def clock_label(seconds):
    ticks = round(seconds * 100)
    day, ticks = divmod(ticks, 8640000)
    hour, ticks = divmod(ticks, 360000)
    minute, ticks = divmod(ticks, 6000)
    second, fraction = divmod(ticks, 100)
    return f'DAY{day + 1} {hour:02d}{minute:02d}{second:02d}{fraction:02d}'


def bind_event_time(time, source, mapping):
    """Keep observed coordinates; never turn availability into an event timestamp."""
    time = dict(time)
    time['label'] = ''
    if source.get('benchmark') != 'egolife':
        return time
    start, end = time.get('start_seconds'), time.get('end_seconds')
    wall_start = timeline_wall_seconds(mapping, start)
    wall_end = timeline_wall_seconds(mapping, end)
    if mapping and any(position is not None and wall is None for position, wall in ((start, wall_start), (end, wall_end))):
        raise ValueError('Memory event position is outside the source timeline')
    if any(wall is not None and wall > source['available_at'] for wall in (wall_start, wall_end)):
        raise ValueError('Observed Memory event is later than its source query_time')
    if wall_start is not None and wall_end is not None:
        time['calendar'] = {'start_seconds': wall_start, 'end_seconds': wall_end,
                            'continuous': math.isclose(wall_end - wall_start, end - start, abs_tol=0.01)}
        time['label'] = f'{clock_label(wall_start)} - {clock_label(wall_end)}'
    return time
