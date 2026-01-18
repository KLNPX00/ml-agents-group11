#!/bin/bash

# The aim of this code is to run all configs generated in our config folder
# It uses the single training run file as a basis to do that

BASE_DIR="./configs"
DATA_DIR="./training_data"

# loop through all rounds
for ROUND_DIR in $(ls -d "$BASE_DIR"/round* | sort -V); do
    ROUND_NAME=$(basename "$ROUND_DIR")
    OUTPUT_CSV="$DATA_DIR/${ROUND_NAME}.csv"

    echo "=== Starting $ROUND_NAME ==="
    echo "Metrics will be saved to $OUTPUT_CSV"

    # uses the single run training shell script and passes command parameters from here
    ./run_training.sh "$ROUND_NAME" "$OUTPUT_CSV"

    echo "=== Finished $ROUND_NAME ==="
done

echo "All trainings completed. Check $DATA_DIR for round-specific CSV files."