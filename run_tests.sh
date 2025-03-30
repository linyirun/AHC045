#!/bin/bash

cd ./tools

echo "Running for $1 iterations per test"


for ((i = 0; i < $1; i++)) ; do
    RESULT=$(cargo run -r --bin tester ../main < in/000${i}.txt 2>&1 1> ../out.txt | tail -n 1)
    echo "${i}:" "$RESULT"
done

echo "Old:\n"
for ((i = 0; i < $1; i++)) ; do
    RESULT=$(cargo run -r --bin tester ../main_old < in/000${i}.txt 2>&1 1> ../out.txt | tail -n 1)
    echo "${i}:" "$RESULT"
done


cd ../