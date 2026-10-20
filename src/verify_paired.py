"""CPU Ray check for counterbalanced mode order, actor reuse and mode summaries."""
import os
os.environ['RAY_ENABLE_UV_RUN_RUNTIME_ENV']='0'
import sys
import ray
import torch

from benchmark_stage21 import comparison_order,aggregate_runs,smoke_workload
from distengine.independent_ray_coordinator import IndependentRayCoordinator
from distengine.metric import MetricsRecorder,LatencyLimits
from distengine.ray_workers import PrefillActor,DecodeActor
from distengine.workload import replay_workload
from verify_independent import close


def main():
    torch.set_num_threads(1)
    modes=comparison_order(3)
    assert modes==[False,True,True,False,False,True]
    ray.init(address='local',num_cpus=4,include_dashboard=False,runtime_env={'py_executable':sys.executable})
    sources=[PrefillActor.remote(num_blocks=24,block_size=4) for _ in range(2)]
    targets=[DecodeActor.remote(num_blocks=24,block_size=4) for _ in range(2)]
    workers=sources+targets
    reports=[];expected=None
    try:
        placement=ray.get([w.ready.remote() for w in workers])
        pids=[w['pid'] for w in placement]
        for index,mode in enumerate(modes):
            # Separate warm-up and measured coordinators; same loaded actors.
            warm=IndependentRayCoordinator(sources,targets,decode_batch_size=4,stealing_enabled=mode)
            replay_workload(smoke_workload(),warm)
            c=IndependentRayCoordinator(sources,targets,metrics=MetricsRecorder(),
                                        decode_batch_size=4,stealing_enabled=mode)
            ray.get([w.start_measurement.remote() for w in workers])
            results=replay_workload(smoke_workload(),c)
            final=ray.get([w.telemetry.remote(stop=True) for w in workers])
            assert [w['pid'] for w in final]==pids
            assert all(not w['active_request_ids'] and w['num_free_blocks']==w['num_blocks'] for w in final)
            outputs={r['request_id']:r['generated_token_ids'] for r in results}
            if expected is None:expected=outputs
            assert outputs==expected
            reports.append({'run':index,'stealing_enabled':mode,
                            'request_metrics':c.metrics.report(LatencyLimits(2,.1,10))})
        summaries={name:aggregate_runs([r for r in reports if r['stealing_enabled']==enabled])
                   for name,enabled in [('no-stealing',False),('stealing',True)]}
        for summary in summaries.values():
            assert summary['executed_repeats']==3
            assert len(summary['aggregate']['request_throughput_rps']['values'])==3
        print('PASS: six counterbalanced passes, unchanged actor PIDs, exact outputs, warm-up/measurement isolation and KV cleanup',flush=True)
        print('PASS: three repeats per mode, separate throughput and latency aggregates',flush=True)
    finally:
        close(workers)
        ray.shutdown()


if __name__=='__main__':main()
