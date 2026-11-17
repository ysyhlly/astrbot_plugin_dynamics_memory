"""Recognize terminal refusals without depending on another plugin's classes."""


def terminal_provider_failure(error):
    reason = getattr(error, "block_reason", None)
    text = str(error).lower()
    if (
        isinstance(reason, str)
        and reason
        or any(
            marker in text
            for marker in (
                "gemini prompt blocked:",
                "prohibited_content",
                "prompt was blocked",
                "content_filter",
            )
        )
    ):
        return "provider_content_blocked"
    if "candidates is empty" in text:
        return "provider_empty_or_blocked"
    return ""
