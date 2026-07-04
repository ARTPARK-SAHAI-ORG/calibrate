"""General-purpose (non-conversational) task evaluation.

Score arbitrary single-shot LLM tasks — summarization, extraction,
classification, rewriting, code generation, etc. — by passing a list of
``(input, output)`` pairs and a list of evaluators. See
:func:`calibrate_agent.general.metrics.get_general_judge_score` for the core
scoring function and :func:`calibrate_agent.general.eval.run_general_eval` for the
file-based runner used by the ``calibrate-agent general`` CLI subcommand.
"""
