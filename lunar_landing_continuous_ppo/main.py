import gym
import torch
import numpy as np
from agent import ActorCritic, ReplayMemory, Updater,gae_calculator,  normalize, visualize
from collections import deque

"""    
'train' 'test' and 'visualize' train also visualizes the results, 
but only visualization visualize mode can be used.
"""
MODE = "test"
MODEL_SAVE_FILE = "models"
TEST_WEIGHT_FILE = "reward_243.pth"
MEMORY_CAPACITY = 4096
BATCH_SIZE = 256
EPOCHS = 2500
MAX_STEP = 2048

AVG_EPOCH_MEMORY = 10
MODEL_TRAINING_EPOCHS = 10

HIDDEN_SIZE = 128
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_PARAM = 0.2
CRITIC_DISCOUNT = 0.5
ENTROPY_BETA = 0.001
LEARNING_RATE = 0.001

env = gym.make("LunarLander-v2",continuous=True, new_step_api = True)
agent = ActorCritic(env.observation_space.shape[0],env.action_space.shape[0],HIDDEN_SIZE)
memory = ReplayMemory(BATCH_SIZE,MEMORY_CAPACITY)
model_updater = Updater(agent,CLIP_PARAM,CRITIC_DISCOUNT,ENTROPY_BETA,learning_rate=LEARNING_RATE)

best_avg_reward = -300
epoch_memory = deque(maxlen=AVG_EPOCH_MEMORY)

if __name__ == "__main__":
    if MODE == "train":
        total_reward_file = open(f"./{MODEL_SAVE_FILE}/rewards_data.txt","a")
        for epoch in range(EPOCHS):
            total_reward = 0
            state = env.reset()

            for i in range(MAX_STEP):
                env.render()
                tensor = torch.tensor([state], dtype=torch.float)
                dist, value = agent(tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                next_state, reward, done, _, _ = env.step(torch.squeeze(action).tolist())
                total_reward += reward
                memory.add_data(state,action.detach().numpy(),log_prob.detach().numpy(),reward,value.detach().numpy(),int(done))
                state = next_state
                if len(memory.memory) == MEMORY_CAPACITY:
                    break
                if done:
                    break

            epoch_memory.append(total_reward)
            avg_reward = int(np.array(epoch_memory).mean())

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(agent.state_dict(),f"./{MODEL_SAVE_FILE}/reward_{int(best_avg_reward)}.pth")
                print("Checkpoint saved successfully...", "AVG_REWARD:", best_avg_reward)

            print(f"EPOCH_NO:{epoch}, REWARD:{int(total_reward)}, BEST_REWARD:{avg_reward}, BEST_AVERAGE_REWARD:{best_avg_reward}")
            total_reward_file.write(f"{epoch} {total_reward}\n")
            total_reward_file.flush()

            if len(memory) >= MEMORY_CAPACITY:
                next_state_tensor = torch.tensor([next_state],dtype=torch.float)
                _, next_value = agent(next_state_tensor)
                states, actions, log_probs, rewards, values, dones = memory.get_memory()
                returns = gae_calculator(next_value.detach().numpy(),rewards,dones,values[:,0].tolist(), gamma = GAMMA, lam=GAE_LAMBDA)
                batches = memory.generate_batches(states, actions, log_probs, returns,values)
                for _ in range(MODEL_TRAINING_EPOCHS):
                    for batch in batches:
                        batch = np.array(batch,dtype=object)
                        states, actions, log_probs, returns, values = batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:,4]
                        returns_tensor = torch.tensor(np.vstack(returns),dtype=torch.float).detach()
                        log_probs_tensor = torch.tensor(np.vstack(log_probs), dtype=torch.float).detach()
                        values_tensor = torch.tensor(np.vstack(values), dtype=torch.float).detach()
                        states_tensor = torch.tensor(np.vstack(states), dtype=torch.float)
                        actions_tensor = torch.tensor(np.vstack(actions), dtype=torch.float)

                        advantage = returns_tensor - values_tensor
                        advantage = normalize(advantage)
                        model_updater.update(states_tensor,actions_tensor, log_probs_tensor, returns_tensor, advantage)
                memory.memory.clear()

        visualize(f"./{MODEL_SAVE_FILE}/rewards_data.txt")

    elif MODE == "test":
        agent.load_state_dict(torch.load(f"./{MODEL_SAVE_FILE}/{TEST_WEIGHT_FILE}"))
        for _ in range(10):
            state = env.reset()
            total_rew = 0
            for i in range(MAX_STEP):
                env.render()
                tensor = torch.tensor([state], dtype=torch.float)
                dist, value = agent(tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                next_state, reward, done, _, _ = env.step(torch.squeeze(action).tolist())
                total_rew += reward
                state = next_state
                if done:
                    break
            print(f"REWARD:{total_rew}")

    elif MODE == "visualize":
        visualize(f"./{MODEL_SAVE_FILE}/rewards_data.txt")
