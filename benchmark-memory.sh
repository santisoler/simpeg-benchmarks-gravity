#!/bin/bash

command time -v python notebooks/large-problem.py choclo 2> results/memory-choclo
command time -v python notebooks/large-problem.py geoana 2> results/memory-geoana
command time -v python notebooks/large-problem.py dask 2> results/memory-dask
