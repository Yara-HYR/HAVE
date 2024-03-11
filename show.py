import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

log_train_filename = "/data/hyr/RL/carla/ckpt/deepmdp2_0.9.6/train.log"
log_eval_filename = "/data/hyr/RL/carla/ckpt/deepmdp2_0.9.6/eval.log"

log_train_filename2 = "/data/hyr/RL/carla/ckpt/deepmdp_0.9.6_mask/train.log"
log_eval_filename2 = "/data/hyr/RL/carla/ckpt/deepmdp_0.9.6_mask/eval.log"

train_steps = []
eval_steps = []

train_episode_rewards = []
eval_episode_rewards = []

br_steps = []
batch_rewards = []



train_steps2 = []
eval_steps2 = []

train_episode_rewards2 = []
eval_episode_rewards2 = []

br_steps2 = []
batch_rewards2 = []


with open(log_train_filename, "r") as file:
    for oneline in file.readlines():
        oneline = oneline[:-1]  # 去掉换行符
        oneline = eval(oneline)

        if "episode_reward" in oneline.keys():
            train_steps.append(oneline["step"])
            train_episode_rewards.append(oneline["episode_reward"])
        
#         if "batch_reward" in oneline.keys():
#             br_steps.append(oneline["step"])
#             batch_rewards.append(oneline["batch_reward"])


with open(log_train_filename2, "r") as file:
    for oneline in file.readlines():

        oneline = oneline[:-1]  # 去掉换行符
        oneline = eval(oneline)

        if "episode_reward" in oneline.keys():
            train_steps2.append(oneline["step"])
            train_episode_rewards2.append(oneline["episode_reward"])
        
#         if "batch_reward" in oneline.keys():
#             br_steps.append(oneline["step"])
#             batch_rewards.append(oneline["batch_reward"])



with open(log_eval_filename, "r") as file:
    for oneline in file.readlines():
        oneline = oneline[:-1]  # 去掉换行符
        oneline = eval(oneline)
        
        if "episode_reward" in oneline.keys():
            eval_steps.append(oneline["step"])
            eval_episode_rewards.append(oneline["episode_reward"])
            
            
with open(log_eval_filename2, "r") as file:
    for oneline in file.readlines():
        oneline = oneline[:-1]  # 去掉换行符
        oneline = eval(oneline)
        
        if "episode_reward" in oneline.keys():
            eval_steps2.append(oneline["step"])
            eval_episode_rewards2.append(oneline["episode_reward"])            



plt.cla()
plt.xlabel("step")
plt.ylabel("reward")

train_episode_rewards_df = pd.DataFrame(train_episode_rewards)
smooth_mean_train_episode_rewards = train_episode_rewards_df[0].rolling(10).mean()

eval_episode_rewards_df = pd.DataFrame(eval_episode_rewards)
smooth_mean_eval_episode_rewards = eval_episode_rewards_df[0].rolling(10).mean()




train_episode_rewards_df2 = pd.DataFrame(train_episode_rewards2)
smooth_mean_train_episode_rewards2 = train_episode_rewards_df2[0].rolling(10).mean()

eval_episode_rewards_df2 = pd.DataFrame(eval_episode_rewards2)
smooth_mean_eval_episode_rewards2 = eval_episode_rewards_df2[0].rolling(10).mean()



# train
# plt.plot(
#     train_steps, train_episode_rewards, 'lightblue'
# )
# plt.plot(
#     train_steps, smooth_mean_train_episode_rewards, 'b', label="train-episode-reward"
# )
# eval
plt.plot(
    eval_steps, eval_episode_rewards, 'lightgreen'
)
plt.plot(
    eval_steps, smooth_mean_eval_episode_rewards, 'g', label="eval-episode-reward"
)






# train
# plt.plot(
#     train_steps2, train_episode_rewards2, 'magenta'
# )
# plt.plot(
#     train_steps2, smooth_mean_train_episode_rewards2, 'r', label="train-episode-reward2"
# )
# eval
plt.plot(
    eval_steps2, eval_episode_rewards2, 'darkorange'
)
plt.plot(
    eval_steps2, smooth_mean_eval_episode_rewards2, 'y', label="eval-episode-reward2"
)









plt.legend()



plt.cla()
plt.xlabel("step")
plt.ylabel("reward")

train_episode_rewards_df = pd.DataFrame(train_episode_rewards)
smooth_mean_train_episode_rewards = train_episode_rewards_df[0].rolling(10).mean()

eval_episode_rewards_df = pd.DataFrame(eval_episode_rewards)
smooth_mean_eval_episode_rewards = eval_episode_rewards_df[0].rolling(10).mean()

# eval
plt.plot(
    eval_steps, eval_episode_rewards, 'lightgreen'
)
plt.plot(
    eval_steps, smooth_mean_eval_episode_rewards, 'g', label="eval-episode-reward"
)
plt.legend()