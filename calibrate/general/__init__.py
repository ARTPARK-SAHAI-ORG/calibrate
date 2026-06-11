"""General-purpose (non-conversational) task evaluation.

Score arbitrary single-shot LLM tasks — summarization, extraction,
classification, rewriting, code generation, etc. — by passing a list of
``(input, output)`` pairs and a list of evaluators. See
:func:`calibrate.general.metrics.get_general_judge_score` for the core
scoring function and :func:`calibrate.general.eval.run_general_eval` for the
file-based runner used by the ``calibrate general`` CLI subcommand.
"""
