#!/bin/bash

# These are command-line parameters
CONFIG_FOLDER="$1"
TRAINING_FILE="$2"

CONFIG_DIR="./configs/$CONFIG_FOLDER"
ENV_BINARY="./builds/mlagents-exec.x86_64"
OUTPUT_CSV="$TRAINING_FILE"

# Here, we check if the file exists and has the relative header (hyperparameters and metrics)
if [ ! -f "$OUTPUT_CSV" ]; then
    echo "run_id,config_name,total_ram_mb,cpu_cores,batch_size,buffer_size,learning_rate,beta,epsilon,lambd,num_epoch,num_layers,hidden_units,gamma,strength,time_horizon,summary_freq,keep_checkpoints,max_steps,cpu_avg_unity,cpu_peak_unity,ram_avg_unity,ram_peak_unity,cpu_avg_python,cpu_peak_python,ram_avg_python,ram_peak_python,duration_sec" > "$OUTPUT_CSV"
fi

# looping through all configs sequentially
for YAML_FILE in "$CONFIG_DIR"/*.yaml; do

    DONE_FLAG="$YAML_FILE.done"

    if [ -f "$DONE_FLAG" ]; then
        # if the file has the .done pointer, it is skipped as we already used it
        # this also helps to track used and unused configs in case the machine crashes
        echo "SKIP: $(basename "$YAML_FILE") already done"
        continue
    fi

    echo "===================================="
    echo "Starting training for: $YAML_FILE"
    echo "===================================="

    # unique run id
    RUN_ID="$(basename "$YAML_FILE" .yaml)_$(date +%s)"

    # starts the logger in the background
    python3 logger.py \
        --unity-process-name "mlagents-exec.x86_64" \
        --python-process-name "mlagents-learn" \
        --interval 0.5 \
        --yaml "$YAML_FILE" \
        --output "$OUTPUT_CSV" &
    LOGGER_PID=$!

    # delay to ensure the logger has started
    sleep 1

    # starts ml-agents
    mlagents-learn "$YAML_FILE" \
        --env "$ENV_BINARY" \
        --run-id "$RUN_ID" \
        --no-graphics

    # delay to ensure the logger finished sample collection
    sleep 1
    kill $LOGGER_PID 2>/dev/null

    touch "$DONE_FLAG"

    echo "Finished training for: $YAML_FILE"
    echo ""
done

echo "All trainings completed. Metrics appended to $OUTPUT_CSV"