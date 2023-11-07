#!/bin/bash

command time -v python notebooks/large-problem.py choclo ram 2> results/memory-choclo-ram
command time -v python notebooks/large-problem.py geoana ram 2> results/memory-geoana-ram
command time -v python notebooks/large-problem.py dask  ram 2> results/memory-dask-ram

command time -v python notebooks/large-problem.py choclo forward_only 2> results/memory-choclo-forward
command time -v python notebooks/large-problem.py geoana forward_only 2> results/memory-geoana-forward
command time -v python notebooks/large-problem.py dask forward_only 2> results/memory-dask-forward
