#!/bin/bash

for file in notebooks/*.py; do
    echo ""
    echo "Running $file"
    python "$file"
done
