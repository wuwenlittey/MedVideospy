"""Shared context and coverage for visual inspections."""


def inspection_prompt(parameters):
    return (
        f"Original task (read-only context):\n{parameters.get('original_question') or parameters['query']}\n\n"
        f"Current inspection goal:\n{parameters['query']}\n\n"
        "Preserve the original task's person, action, time, temporal order and "
        "viewpoint. If the inspection goal changes these constraints, report the "
        "mismatch rather than treating the changed target as the answer. "
        "The task and its options are not evidence; any final-answer formatting "
        "requirement in the original task applies to the agent, not this inspection. "
        "Return concise sections: Facts (directly visible details or quoted subtitles "
        "with timestamps), Interpretation (only judgments supported by those facts, "
        "preserving uncertainty), and Unknowns (answer-critical gaps and plausible "
        "alternatives, or none). Do not repeat the same observation for unchanged "
        "frames. Non-detection in sampled frames does not establish absence throughout "
        "the interval or video. Sample positions are not event numbers; first, last "
        "and total-count claims require evidence of the relevant order or coverage.\n\n"
    )


def sampling_summary(timestamps):
    times = sorted({float(timestamp) for timestamp in timestamps})
    max_gap = max((b - a for a, b in zip(times, times[1:])), default=0.0)
    return (
        f"Actual sampled frames: {len(times)}; timestamp range: "
        f"{times[0]:.3f}s - {times[-1]:.3f}s; "
        f"maximum adjacent sample gap: {max_gap:.3f}s.\n"
        "Coverage: individual sampled frames only, not continuous observation. "
        "Unsampled moments were not inspected.\n"
    )
