# Frozen benchmark inputs

`stage19/` contains the version-1 mixed workload, prompt bank, and manifest.
These are experiment inputs and are intentionally outside the ignored `results/`
directory. Preserve them across performance comparisons.

- `mixed-v1.json`: 100 scheduled requests with exact token IDs.
- `prompt-bank.json`: nine source texts, token IDs, and length-class boundaries.
- `manifest.json`: seeds, model/tokenizer identity, settings, prompt assignments,
  SHA-256 checksums, and the two-source round-robin distribution.

Run `PYTHONPATH=src uv run python src/verify_mixed_workload.py` from the repository
root to verify integrity and CPU correctness. Add `--tokenizer` to retokenize with
the cached pinned tokenizer and reproduce all three files byte for byte.

Generate copies with `src/freeze_workload.py --output <empty-directory>`. Choose
a new versioned directory for future workload changes; do not silently replace
these inputs when tuning performance. The stage 19 manifest records an input
imbalance of about 5.4% in total prompt tokens, not measured worker utilization.
