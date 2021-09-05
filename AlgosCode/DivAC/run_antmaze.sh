#!/bin/bash

for ((i=0;i<5;i+=1))
do
  python3 main.py -alg "DivAC-R" -d_g "Stochastic"
done
