"""Shared evidence checks for reviewing methods and historical reasoning cases."""

EVIDENCE_REVIEW_CHECKS = """For every proposed keep, report checks with three independent verdicts:
{"action":true,"scope":true,"conclusion":true}. Use false for a mismatch and null for unknown.
Judge the candidate's stated claim and the assigned evidence role, not whether the entire
trajectory was successful. For supports and successful cases, action checks actual calls and
arguments. For failure_example, action checks whether the observed mistake motivates this
specific remedy; the remedy need not have been executed. Do not relabel failure as success.
Judge the trigger BEFORE the corrective action and the result AFTER it: an initial wrong
MM:SS conversion followed by a corrected query is positive evidence for time correction.
Success after correction does not mean the initial error or condition never existed.
Do not credit an imagined verification step or a tool named only in prose.
scope: the evidence covers the candidate's claimed people, times, and relation. Convert MM:SS correctly.
A state is not a change; a later sampled frame is not necessarily the first/next event.
conclusion: concrete details establish the claim, not just a tool's summary or assertion.
Explicitly transcribed subtitles, name captions, and signs ARE observations of what the video
states. Reading a caption identifying someone does not require witnessing the reported event
or every named person interacting. Do not turn a video's statement into independently verified truth.
For a case, a useful local subproblem may be resolved even if the full multiple-choice answer
remains uncertain. Reading a route or name is useful despite a spelling mismatch in options,
provided the case reports the observed text without claiming an unsupported option match.
Do not reject that observation because an option differs; do not claim they are equivalent either.
Objects present together do not prove an interaction; smiles do not establish life attitudes;
missing sparse observations do not prove absence or a total count. Check ALL stop alternatives.
For change, require relevant before/after states; for order or absence, require coverage that
establishes that relation. A correct answer does not supply missing evidence.
For failure_example, conclusion checks that the recorded error supports the prevention lesson,
not that the proposed remedy already succeeded. Generic advice can still fail the usefulness check.
In reason, name the decisive observation or the specific missing evidence. A claim repeated
in the tool's conclusion is not an additional observation. Do not silently narrow or repair
the claim while keeping its original wording. Keep requires all three checks to be true.
These are text support checks, not independent verification of the video.
"""


def validate_review_checks(checks):
    if not isinstance(checks, dict) or set(checks) != {'action', 'scope', 'conclusion'}:
        raise ValueError('Kept evidence requires checks: action, scope, conclusion')
    if any(value is not None and not isinstance(value, bool) for value in checks.values()):
        raise ValueError('Review checks must be true, false or null')
    return all(value is True for value in checks.values())
