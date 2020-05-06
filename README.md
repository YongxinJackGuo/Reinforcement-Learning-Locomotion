# A Reinforcement Learning Project with Application on Quadruped Locomotion using OpenAI Gym and Mujoco Simulation Toolbox  
-agents: target physical model, Walker2d, Hopper2d and ant-v3 in this case. This algorithm is model invariant.   
-model: value and policy network. Value network outputs predicted value, and policy network outputs mean and standard deviation of actions  
-core:  Trust Region Policy Optimization (TRPO) step  
-utils: utility functions (conjugate-gradient, KL-divergence, line-search, functions of get and set parameters for neworks, Hessian-vector product (hvp), get_advantages using General Advantage Estimate (GAE).  
-main.py: main policy training script that uses all of above functions  
-gym test: some environment pre-experiments and tests  
 
