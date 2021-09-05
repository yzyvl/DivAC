#!/bin/bash

for ((i=0;i<5;i+=1))
do
  python3 main.py -alg "DivAC-R" -d_g "Stochastic" -env "HalfCheetah-v2"
  python3 main.py -alg "DivAC-R" -d_g "Stochastic" -env "Humanoid-v2"
  python3 main.py -alg "DivAC-R" -d_g "Stochastic" -env "Walker2d-v2"
  python3 main.py -alg "DivAC-R" -d_g "Stochastic" -env "Swimmer-v2"
  python3 main.py -alg "DivAC-R" -d_g "Stochastic" -env "Hopper-v2"
  python3 main.py -alg "DivAC-R" -d_g "Stochastic" -env "Ant-v2"
done
