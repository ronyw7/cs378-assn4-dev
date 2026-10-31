"""Engines.

vidur_adapter, vidur_profile   the evaluator behind score() and viz.record (Vidur's vLLM V1 scheduler)
kv, scheduler, seq, sim, timing   a small block-level teaching model used by agentkv.harness.CacheHarness
                                  and the cache-invariant tests; it never produces scores
"""
