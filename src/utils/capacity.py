"""Deterministic backlog construction and capacity reporting."""

from copy import deepcopy
import statistics


def make_backlog(payload, num_requests, output_tokens):
    if type(num_requests) is not int or num_requests <= 0:
        raise ValueError("Request count must be a positive integer")
    if type(output_tokens) is not int or output_tokens <= 0:
        raise ValueError("Output length must be a positive integer")
    if not payload["requests"]:
        raise ValueError("Source workload must contain requests")
    result = deepcopy(payload)
    result["requests"] = []
    for index in range(num_requests):
        item = deepcopy(payload["requests"][index % len(payload["requests"])])
        item.update(
            request_id=f"capacity-{index:06d}",
            arrival_offset_s=0.0,
            max_new_tokens=output_tokens,
            eos_token_id=-1,
        )
        result["requests"].append(item)
    return result


def summarize_capacity(report):
    rows = []
    for batch in sorted({run["decode_batch_size"] for run in report["runs"]}):
        runs = [run for run in report["runs"] if run["decode_batch_size"] == batch]
        rows.append(
            dict(
                batch_size=batch,
                passes=len(runs),
                complete_passes=sum(run["complete"] for run in runs),
                correctness_passed=all(run["validation"]["status"] == "passed" for run in runs),
                observed_tokens_per_s=statistics.mean(
                    run["request_metrics"]["observed_output_tokens_per_s"] for run in runs
                ),
            )
        )
    return rows
