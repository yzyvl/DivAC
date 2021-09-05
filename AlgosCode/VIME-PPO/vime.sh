#!/bin/bash

for ((i=0;i<5;i+=1))

do
  python3 main.py --env_name "Swimmer-v2"
  python3 main.py --env_name "Hopper-v2"
  python3 main.py --env_name "Walker2d-v2"
  python3 main.py --env_name "Ant-v2"
  python3 main.py --env_name "HalfCheetah-v2"
  python3 main.py --env_name "Humanoid-v2"
done
