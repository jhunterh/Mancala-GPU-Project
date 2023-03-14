#!/bin/bash

module load cuda

echo -ne "\n\nWaiting for job to start...\n\n"

echo -ne "==================\n"
echo -ne "Starting execution\n"
echo -ne "==================\n\n"

# nsys profile build/bin/simulation

# echo -ne "\n\n"

# ncu -k game_simulation -o profile build/bin/simulation

build/bin/simulation 2 2

echo -ne "\n==================\n"
echo -ne "Finished execution\n"
echo -ne "==================\n\n"
echo "Hit Ctrl + C to exit..."
