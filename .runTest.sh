#!/bin/bash

module load cuda

echo -ne "\n\nWaiting for job to start...\n\n"

echo -ne "==================\n"
echo -ne "Starting execution\n"
echo -ne "==================\n\n"

# nsys profile build/bin/test

# echo -ne "\n\n"

# ncu -k game_test -o profile build/bin/test

build/bin/test &> test_output.txt

echo -ne "\n==================\n"
echo -ne "Finished execution\n"
echo -ne "==================\n\n"
echo "Hit Ctrl + C to exit..."
