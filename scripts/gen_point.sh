#!/bin/bash

# This script runs the generate.py script for a list of puzzle types.

# Define an array of all the puzzle names to process
PUZZLES=(
    'arc_connect_point_ver'
    "ray_intersection"
    "midpoint"
    "parallelogram"
    "circle_center"
    "incenter"
    "circumcenter"
    "triangle_center"
    "perpendicular"
    "angle_bisector"
    "parallel"
    "perpendicular_bisector"
    "orthocenter"
    "reflection"
    "ray_reflect"
    "right_triangle"
    "square_outlier"
    "fermat_point"
    "isosceles_trapezoid"
    "circle_tangent_point"
    "circle_tangent_line"
)

# Common arguments for the script
NUM_PUZZLES=50
GEN_OPTS="--canvas-width=480 --aspect=0.55"
OUTPUT_DIR="dataset/"
VIDEO=true  # <--- Change this to true or false

# Prepare the video flag based on the setting above
VIDEO_FLAG=""
if [ "$VIDEO" = "true" ]; then
    VIDEO_FLAG="--video"
fi

# Loop through the array and execute the command for each puzzle name
for PUZZLE_NAME in "${PUZZLES[@]}"; do
    echo "----------------------------------------------------"
    echo "Processing puzzle type: $PUZZLE_NAME"
    echo "----------------------------------------------------"
    
    # Construct and run the command
    python -m puzzle.$PUZZLE_NAME.generator "$NUM_PUZZLES" $GEN_OPTS --output-dir "$OUTPUT_DIR/$PUZZLE_NAME" $VIDEO_FLAG
    
    # Check the exit code of the last command
    if [ $? -ne 0 ]; then
        echo "Command failed for puzzle type: $PUZZLE_NAME. Aborting script."
        exit 1
    fi
    
    echo "" # Add a blank line for readability
done

echo "===================================================="
echo "All puzzle types processed successfully."
echo "===================================================="