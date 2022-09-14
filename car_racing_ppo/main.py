import os
import shutil
import gym
import torch as t
from torchvision import transforms
import numpy as np
from libs import ActorCritic, ReplayMemory, Updater, gae_calculator, normalize, visualize
from collections import deque

"""
'train' 'test' and 'visualize' train also visualizes the results, 
but only visualization visualize mode can be used.
"""

MODE = "test"
SAVE_FOLDER_NAME = "models"
TEST_WEIGHT_FILE = "reward_871.pth"
MEMORY_CAPACITY = 512 * 3 * 16
BATCH_SIZE = 32
EPOCHS = 3000
MAX_STEP = 512 * 3

TRAIN_EPISODE_TERMINATION_TOLERANCE = 15
TEST_EPISODE_TERMINATION_TOLERANCE = 50
AVG_EPOCH_MEMORY = 10
MODEL_TRAINING_EPOCHS = 2
TEST_EPOCH_NUM = 50
TEST_TERMINATION_REWARD = 800

FCL_HIDDEN_1 = 128
FCL_HIDDEN_2 = 128
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_PARAM = 0.2
CRITIC_DISCOUNT = 0.5
ENTROPY_BETA = 0.001
LEARNING_RATE = 0.001

env = gym.make("CarRacing-v2", new_step_api=True)
agent = ActorCritic(env.observation_space.shape, env.action_space.shape[0], FCL_HIDDEN_1, FCL_HIDDEN_2)

""" if train is wanted to continue from snapshot, uncomment below. And reorganize folder name. """
# agent.load_state_dict(t.load(f"./models/{TEST_WEIGHT_FILE}"))
memory = ReplayMemory(BATCH_SIZE, MEMORY_CAPACITY)
model_updater = Updater(agent, CLIP_PARAM, CRITIC_DISCOUNT, ENTROPY_BETA, learning_rate=LEARNING_RATE)
transformer = transforms.ToTensor()

best_avg_reward = -300
avg_reward = best_avg_reward
epoch_memory = deque(maxlen=AVG_EPOCH_MEMORY)

if MODE == "test":
    epoch_memory = deque(maxlen=TEST_EPOCH_NUM)

if __name__ == "__main__":
    if MODE == "train":
        if SAVE_FOLDER_NAME == "temp":
            shutil.rmtree('./temp/', ignore_errors=True)
        if not os.path.isdir(f"./{SAVE_FOLDER_NAME}"):
            os.mkdir(f"./{SAVE_FOLDER_NAME}")

        total_reward_file = open(f"./{SAVE_FOLDER_NAME}/reward_history.txt", "a")
        for epoch in range(1281,EPOCHS):
            total_reward = 0
            temp_max_total_reward = 0
            state = env.reset()

            for i in range(MAX_STEP):
                # env.render()   #if you want to watch agent while training, just uncomment.
                tensor = transformer(state)
                dist, value = agent(tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                next_state, reward, done, _, _ = env.step(t.squeeze(action).tolist())

                if total_reward > temp_max_total_reward:
                    temp_max_total_reward = total_reward

                if temp_max_total_reward - total_reward > TRAIN_EPISODE_TERMINATION_TOLERANCE:
                    reward = -10
                    done = True

                total_reward += reward
                memory.add_data(np.array(state), action.detach().numpy(), log_prob.detach().numpy(), reward,
                                value.detach().numpy(), int(done))
                if done:
                    break

                if len(memory) == MEMORY_CAPACITY:
                    break

                state = next_state

            epoch_memory.append(total_reward)
            if len(epoch_memory) == AVG_EPOCH_MEMORY:
                avg_reward = int(np.array(epoch_memory).mean())

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                t.save(agent.state_dict(), f"./{SAVE_FOLDER_NAME}/reward_{int(best_avg_reward)}.pth")
                print("Checkpoint saved successfully...", "AVG_REWARD:", best_avg_reward)

            print(
                f"EPOCH_NO:{epoch}, REWARD:{int(total_reward)}, AVG_REWARD:{avg_reward}, BEST_AVERAGE_REWARD:{best_avg_reward}")
            total_reward_file.write(f"{epoch} {total_reward}\n")
            total_reward_file.flush()

            if len(memory) >= MEMORY_CAPACITY:
                next_state_tensor = transformer(next_state)
                _, next_value = agent(next_state_tensor)
                states, actions, log_probs, rewards, values, dones = memory.get_memory()
                returns = gae_calculator(next_value.detach().numpy(), rewards, dones, values[:, 0].tolist(),
                                         gamma=GAMMA, lam=GAE_LAMBDA)
                batches = memory.generate_batches(states, actions, log_probs, returns, values)
                for _ in range(MODEL_TRAINING_EPOCHS):
                    for batch in batches:
                        batch = np.array(batch, dtype=object)
                        states, actions, log_probs, returns, values = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4]
                        tensor_list = []
                        for state in states:
                            img_tensor = transformer(state)
                            tensor_list.append(img_tensor)
                        states_tensor = t.stack(tensor_list)
                        returns_tensor = t.tensor(np.vstack(returns), dtype=t.float).detach()
                        log_probs_tensor = t.tensor(np.vstack(log_probs), dtype=t.float).detach()
                        values_tensor = t.tensor(np.vstack(values), dtype=t.float).detach()
                        actions_tensor = t.tensor(np.vstack(actions), dtype=t.float)

                        advantage = returns_tensor - values_tensor
                        advantage = normalize(advantage)
                        model_updater.update(states_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantage)
                memory.memory.clear()

        visualize(f"./{SAVE_FOLDER_NAME}/reward_history.txt")

    elif MODE == "test":
        agent.load_state_dict(t.load(f"./{SAVE_FOLDER_NAME}/{TEST_WEIGHT_FILE}"))
        for _ in range(TEST_EPOCH_NUM):
            state = env.reset()
            total_rew = 0
            temp_max_total_rew = 0.0
            for i in range(MAX_STEP):
                env.render()
                tensor = transformer(state)
                dist, value = agent(tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                next_state, reward, done, _, _ = env.step(t.squeeze(action).tolist())
                total_rew += reward
                if total_rew > temp_max_total_rew:
                    temp_max_total_rew = total_rew

                if temp_max_total_rew - total_rew > TEST_EPISODE_TERMINATION_TOLERANCE or total_rew >= TEST_TERMINATION_REWARD:
                    done = True

                state = next_state
                if done:
                    break
            print(f"REWARD:{total_rew}")
            epoch_memory.append(total_rew)
        mem = np.array(epoch_memory)
        print(f"Average score in 10 test epochs: {mem.mean()}")
        visualize(mem)

    elif MODE == "visualize":
        visualize(f"./{SAVE_FOLDER_NAME}/reward_history.txt")
