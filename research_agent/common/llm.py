"""Shared LLM helper used across the pipeline."""

from __future__ import annotations

from anthropic import AnthropicBedrock


def complete(
    client: AnthropicBedrock,
    *,
    model: str,
    max_tokens: int,
    system: str,
    user: str,
) -> str:
    """Single Claude completion.

    Non-streaming: Bedrock's streaming decoder requires ``botocore``, which
    isn't needed for bearer-token (``AWS_BEARER_TOKEN_BEDROCK``) auth. Our
    completions stay well under the ~16K-token threshold where non-streaming
    risks an HTTP timeout, so a plain request is both sufficient and avoids the
    extra dependency.
    """
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in message.content if b.type == "text")
