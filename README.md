# Benchmarks for comparing gravity simulations in SimPEG

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
> These benchmarks were designed to be run on a machine with 125GB of ram and
> a minimum of 30 threads. If your system don't meet these specs, you can
> modify the scripts to adequate them to your needs.


[miniforge]: https://github.com/conda-forge/miniforge
