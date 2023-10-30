import gc
import time
import numpy as np
from tqdm import tqdm
from SimPEG.potential_fields import gravity as simpeg_gravity


class SimulationBenchmarker:
    """
    Class to benchmark a gravity simulation
    """

    def __init__(self, n_runs=10, verbose=True, **kwargs):
        self.n_runs = n_runs
        self.verbose = verbose
        self.kwargs = kwargs

    @property
    def simulation(self):
        simulation = simpeg_gravity.simulation.Simulation3DIntegral(**self.kwargs)
        return simulation

    def benchmark(self, model):
        # Run a first round to compile Numba code
        if "engine" in self.kwargs and self.kwargs["engine"] == "choclo":
            self._benchmark_single(model)

        # Build iterator to time the simulation
        iterator = range(self.n_runs)
        if self.verbose:
            iterator = tqdm(iterator, unit="run")

        # Benchmark the simulation
        times = []
        for i in iterator:
            time = self._benchmark_single(model)
            times.append(time)

        # Compute mean and std
        mean = np.mean(times)
        std = np.std(times)

        if self.verbose:
            print(f"Elapsed time: {mean:.2f} +/- {std:.2f} s")

        return (mean, std)

    def _benchmark_single(self, model):
        # Build simulation
        simulation = self.simulation
        # Benchmark the simulation
        start = time.perf_counter()
        simulation.dpred(model)
        end = time.perf_counter()
        # Clean up memory
        del simulation._G
        del simulation
        gc.collect()
        return end - start
