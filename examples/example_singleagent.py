from gym_battlesnake.gymbattlesnake import BattlesnakeEnv
from gym_battlesnake.custompolicy import CustomPolicy
from stable_baselines import PPO2

env = BattlesnakeEnv(n_threads=4, n_envs=16)

model = PPO2(
    CustomPolicy,
    env,
    learning_rate=1e-4,
    n_steps=128,
    cliprange=0.1,
    vf_coef=1.0,
    ent_coef=0.01,  # Can experiment with lowering this later
    verbose=1
)
model.learn(total_timesteps=1_000_000)
model.save('../dd2438-battlesnake/models/ppo2_trainedmodel')

# del model

# model = PPO2.load('ppo2_trainedmodel')

# obs = env.reset()
# for _ in range(10000):
#     action,_ = model.predict(obs)
#     obs,_,_,_ = env.step(action)
#     env.render()