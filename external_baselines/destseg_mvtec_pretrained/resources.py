"""Non-sampling resource records: no background thread, subprocess or CUDA fence."""
from contextlib import contextmanager
import time
from external_baselines.patchcore_official_eval.storage import csv_write


class AllocatorResources:
    def __init__(self, directory, devices, category=None, *, cuda=None):
        if cuda is None:
            from torch import cuda
        self.cuda = cuda
        self.directory = directory
        self.devices = [str(d) for d in devices if str(d).startswith('cuda:')]
        self.category = category
        self.rows = []

    @contextmanager
    def measure(self, stage):
        warnings = []
        for device in self.devices:
            try:
                self.cuda.reset_peak_memory_stats(device)
            except Exception as exc:
                warnings.append(type(exc).__name__)
        start = time.perf_counter()
        status = 'failed'
        try:
            yield
            status = 'complete'
        finally:
            row = dict(category=self.category, stage=stage, status=status,
                seconds=time.perf_counter()-start,
                timing='host wall time; no added CUDA synchronization; training loss.item synchronizes steps',
                nvml_sampling='disabled; no whole-card observations; allocator peaks only')
            for device in self.devices:
                try:
                    row[device+'_allocated_peak_bytes'] = self.cuda.max_memory_allocated(device)
                    row[device+'_reserved_peak_bytes'] = self.cuda.max_memory_reserved(device)
                except Exception as exc:
                    warnings.append(type(exc).__name__)
            row['resource_warnings'] = ','.join(warnings)
            self.rows.append(row)
            try:
                csv_write(self.directory/'resources.csv', self.rows)
            except Exception as exc:
                # Auxiliary statistics must never prevent a completed model save.
                print(f'WARNING resource record {stage}: {type(exc).__name__}', flush=True)
