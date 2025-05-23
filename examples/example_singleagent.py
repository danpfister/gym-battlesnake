from gym_battlesnake.gymbattlesnake import BattlesnakeEnv
from gym_battlesnake.custompolicy import CustomPolicy
from stable_baselines import PPO2

env = BattlesnakeEnv(n_threads=4, n_envs=16)

model = PPO2(CustomPolicy, env, verbose=1, learning_rate=1e-3)
model.learn(total_timesteps=1000000)
model.save('../dd2438-battlesnake/models/ppo2_trainedmodel_2')
print("trained model.")

del model

# model = PPO2.load('ppo2_trainedmodel')

# obs = env.reset()
# for _ in range(10000):
#     action,_ = model.predict(obs)
#     obs,_,_,_ = env.step(action)
#     env.render()