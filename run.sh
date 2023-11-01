#!/bin/bash

# Run python scripts
for file in notebooks/[0-9][0-9]*.py; do
    echo ""
    echo "Running $file"
    python "$file"
done

# Benchmark memory usage on large problem using GNU's time
bash benchmark-memory.sh
