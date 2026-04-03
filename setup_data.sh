#!/bin/bash

mkdir -p data

if [ ! -f data/chemical_process.parquet ]; then
    echo "Installing dependencies..."
    pip install pandas pyarrow -q

    echo "Downloading dataset..."
    kaggle datasets download -d rohit8527kmr7518/chemical-process-monitoring-time-series-dataset
    unzip -o chemical-process-monitoring-time-series-dataset.zip -d data/

    echo "Converting to Parquet..."

    python3 - << 'PYCODE'
import pandas as pd

df = pd.read_csv("data/chemical_process_timeseries.csv")
df.to_parquet("data/chemical_process.parquet")

print("Conversion complete")
PYCODE

    rm data/chemical_process_timeseries.csv
    echo "Setup complete ✅"

else
    echo "Dataset already exists ✅"
fi
