#!/bin/bash
pwd

block_size="$1"
problem_size="$2"

make run ARGS="benchmarks/example.cu --gpu h200 --verbose --block-size $block_size --problem-size $problem_size"
