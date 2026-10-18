"""Final-answer instructions after evidence gathering."""


def final_answer_message(finish_reason: str) -> dict:
    prefix = (
        "The visual tool-call limit has been reached. "
        if finish_reason == "reach_max_steps"
        else "The evidence-gathering phase is complete. "
    )
    return {
        "role": "user",
        "content": (
            prefix
            + "Review the complete conversation, including the question, "
            "applicable prior memories, subtitles, and visual tool observations. Provide the final "
            "answer based only on that evidence. Verify any preceding "
            "assistant summary against the observations; it is not independent "
            "evidence. For multiple-choice questions, first match the factual "
            "evidence to the meaning of an option, then output that option's "
            "letter. Check that its distinguishing details and the question's "
            "constraints are supported; facts shared by competing options do not "
            "resolve their difference. Preserve uncertainty in tool interpretations "
            "rather than treating them as verified facts. Do not copy a previously "
            "proposed letter. Follow the final-answer "
            "format in the system prompt and any explicit output requirement "
            "in the question exactly. Return only the requested answer, "
            "without reasoning, headings, code fences, or extra text. "
            "For multiple-choice questions requiring a single letter, return "
            "exactly one valid option letter even when evidence remains uncertain. "
            "Uncertainty does not change the required output format. Do not "
            "repeat earlier explanations or debate the answer format."
        ),
    }
