#!/bin/bash

# Run the command and capture output
output=$(squeue)

# Count the number of jobs for user "arvind"
curr_jobs=$(echo "$output" | grep "arvind" | wc -l)

# Determine priority based on the number of current jobs
if [ $curr_jobs -lt 2 ]; then
    echo "high"
elif [ $curr_jobs -lt 3 ]; then
    echo "medium"
elif [ $curr_jobs -lt 5 ]; then
    echo "default"
else
    echo "scavenger"
fi