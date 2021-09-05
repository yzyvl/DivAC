# A Divergence Approach to Optimal Policy in Reinforcement Learning

PyTorch implementation of Divergence Actor-Critic (DivAC). Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym). 
Networks are trained using [PyTorch 1.3](https://github.com/pytorch/pytorch) and Python 3.6.5. 


### Usage

Experiments on single environments can be run by calling:
```
python3 main.py -env HalfCheetah-v2
```

Hyper-parameters can be modified with different arguments to main.py.


Algorithms which TD3 compares against (TD3, SAC) can be found at [OpenAI spinningup repository](https://github.com/openai/spinningup) and VIME+PPO. 

















