# Benchmarks for comparing gravity simulations in SimPEG

## About

This repository was created to run benchmarks between the gravity forward model
available in [SimPEG][simpeg] [v0.20][simpeg-v0.20] and the new implementation
of it that was merged in
[simpeg@eb07ef133a6c5e1714ff1af181147cb172eba97c](https://github.com/simpeg/simpeg/commit/eb07ef133a6c5e1714ff1af181147cb172eba97c).
This new implementation
makes use of [Numba][numba] and the gravity kernels implemented in
[Choclo][choclo] to achieve faster run times and handle memory in a more
efficient way.

The benchmarks were set up in the Python files located in the `notebooks`
folder. They compare the two implementations in different scenarios, varying
the number of cells in the mesh, the number of receivers, enabling and
disabling parallelization (using both `mutliprocessing` and `dask` for the
SimPEG v0.20 implementation), and varying the number of threads assigned to
the simulation.

The results were saved as netCDF files (that can be opened with
[Xarray][xarray]) in the `raw` folder. Figures of these results can be found in
the `figs` folder.

![Results of the benchmarks comparing the SimPEG v0.20 and the new
implementation of the gravity simulation. The plot shows run times for
different number of threads. The new implementation proves to be between 2 and
13 times faster on building the sensitivity matrix, and 4 to 63 times faster
for computing only the forward model.](figs/benchmark_n-processes.png)

## Get started

In order to run these benchmarks, you need to have a Python distribution like
[Miniforge][miniforge] installed.

Then, clone this repository:

```bash
git clone https://github.com/santisoler/simpeg-benchmarks-gravity
cd simpeg-benchmarks-gravity
```

And create a `conda` environment with all the required dependencies for running
these benchmarks:

```bash
mamba env create -f environment.yml
```

> **Important**
> I recommend using `mamba` over `conda` due to its faster speeds for resolving
> dependencies and installing packages. In case you don't have `mamba` installed,
> feel free to replace `mamba` with `conda`.


## Run the benchmarks

All benchmarks can be run by executing the Python scripts in `notebooks`
folder, and through the `benchmark-memory.sh` script.

Alternatively, we can run all benchmarks by executing the `run.sh` shell
script:

```bash
bash run.sh
```

> **Important**
> Most of the benchmarks were designed to be run on a machine with 125 GB of
> ram and a minimum of 30 threads. If your system doesn't meet these specs, you
> can modify the scripts to adjust them to your needs.
>
> The benchmarks for the "large problem" require more memory: up to ~800 GB.

[miniforge]: https://github.com/conda-forge/miniforge
[simpeg]: https://simpeg.xyz
[simpeg-v0.20]: https://github.com/simpeg/simpeg/releases/tag/v0.20.0
[numba]: https://numba.pydata.org
[choclo]: https://www.fatiando.org/choclo
[xarray]: https://docs.xarray.dev
