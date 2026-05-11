from profine.profiler.orchestrator import ProfileOrchestrator, ProfileResult


def profile(
    script_path,
    *,
    hardware="1x_a100",
    steps=60,
    warmup_steps=30,
    provider="openai",
    api_key=None,
    model=None,
    script_args=None,
    **kwargs,
) -> ProfileResult:
    """Profile a training script on Modal.

    Convenience function wrapping ProfileOrchestrator.

    Args:
        script_path: Path to the training script.
        hardware: Hardware preset name (e.g. "1x_a100", "1x_h100").
        steps: Total optimizer steps to run.
        warmup_steps: Steps to discard as warmup.
        provider: LLM provider ("anthropic" or "openai").
        api_key: Optional API key override.
        model: Optional model name override.
        script_args: Optional CLI arguments for the script.

    Returns:
        ProfileResult with .record (machine), .markdown (human), .save().
    """
    orchestrator = ProfileOrchestrator(
        provider=provider,
        api_key=api_key,
        model=model,
        **kwargs,
    )
    return orchestrator.profile(
        script_path,
        hardware=hardware,
        steps=steps,
        warmup_steps=warmup_steps,
        script_args=script_args,
    )
