# Trace format (`agentkv-trace/1`)

A trace fixes all of the token work: every prompt, every output, and every tool
delay. A policy only changes when calls run and which prefixes stay cached.
Every workload family uses this one format. You (and your agents) may read the
public traces directly; `agentkv.trace.load(path)` gives you parsed jobs with
token counts and block hashes, and `trace.stats()` a summary.

One JSON file per trace:

```json
{
  "schema": "agentkv-trace/1",
  "trace_id": "dev-D-0",
  "family": "D",
  "split": "public_dev",
  "description": "free text for humans",
  "tokenizer": "qwen3-0.6b@c1899de2",
  "memory": {"num_blocks": 192, "block_size": 256},
  "engine": {"max_num_seqs": 16, "max_num_batched_tokens": 8192},
  "generation": {"generator": "tracegen.dfam", "seed": 0, "sources": ["data/corpus (English Wikipedia, CC BY-SA 4.0)"]},
  "segments": {
    "s3f2a...": {"text": "<|im_start|>system\n...<|im_end|>\n", "n_tokens": 812}
  },
  "jobs": [
    {"id": "J00", "arrival": 0.0, "meta": {},
     "nodes": [
       {"id": "n0", "type": "llm", "stage": "plan", "deps": [],
        "prompt": ["s3f2a...", "s91c0..."], "output": "s77d1..."},
       {"id": "n1", "type": "tool", "stage": "search", "deps": ["n0"], "delay": 0.62}
     ]}
  ]
}
```

## Rules

Enforced by `agentkv.trace.validate`:

- `segments` maps a segment id to its text and token count. The token
  sequence of a prompt is the concatenation of each segment tokenized
  **separately** with the frozen tokenizer, in order. Shared prefixes are
  expressed by sharing segment ids, so put shared material first and split
  long prompts at natural boundaries (system prompt, instructions, each
  document, the question, the generation header).
- Segment ids should be content hashes: `"s" + sha1(text)[:16]`. Use
  `agentkv.trace.segment_id(text)`.
- `n_tokens` must equal the tokenizer's count; load fails otherwise.
- `type: "llm"` nodes have `prompt` (non-empty list of segment ids) and
  `output` (one segment id; its tokens are the frozen output, at least 1 token).
- `type: "tool"` nodes have `delay` (seconds, >= 0). Think time is a tool node
  with `stage: "think"`.
- `deps` name nodes in the same job; the graph must be acyclic. A job
  completes when all of its nodes complete. Flow time is completion minus
  `arrival`.
- `stage` is a short label (`plan`, `extract`, `chat`, `answer`, `agent`, ...). The policy
  only ever sees it rekeyed to an opaque id; `family`, `split`, `trace_id`,
  `description`, `generation` and `meta` are never shown to a policy.
- Exactly 16 jobs for graded splits, sorted by `arrival`.
- Every single LLM call must fit in half of memory:
  `ceil((prompt + output) / block_size) <= num_blocks // 2`.

## Chat rendering convention

All workloads render chat with Qwen3 control tokens so that history is
append-only and later prompts share the earlier prompt + output as a prefix:

```text
system   "<|im_start|>system\n{content}<|im_end|>\n"
user     "<|im_start|>user\n{content}<|im_end|>\n"
tool     "<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
header   "<|im_start|>assistant\n"            (the generation prompt; its own segment)
output   "{assistant content}<|im_end|>\n"     (the node's output segment)
```

Turn k+1's prompt is turn k's prompt + turn k's output segment + the new
message segments + header. Helpers: `agentkv.trace.chat_segment(role, text)`,
`agentkv.trace.HEADER`.

W1 is built from the Mooncake conversation trace, which publishes token counts
and the hash of every 512-token prompt block but no text. Its segments are
placeholder words, one segment per source block, chosen so that prompts that
shared a block in the source share it here and every count is exact. W5 comes
from the Qwen Bailian thinking trace, which hashes 16-token blocks; each run of
16 hashes becomes one 256-token placeholder segment, with the same guarantees.

## Vidur replay settings

Source files retain their original token encoding and block-size metadata.
The Vidur evaluator rebuilds chained hashes in 16-token blocks and keeps
`num_blocks * block_size` tokens of KV capacity per replica. It does not change
arrivals, dependencies, tools, prompt lengths, or output lengths. Part 2 uses
one replica; Part 3 uses two independent replicas. Run diagnostics record the
effective settings. See the [README](../README.md#evaluator-configuration) for
the model profile and limitations of this length-based replay.
