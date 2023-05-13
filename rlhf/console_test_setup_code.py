# fine-tune for one response.
# need to pay attention to device settings
import os.path

import torch
import torch.nn as nn
from llm.models import *
import copy
from rlhf.reward_model import *
from torch.optim import Adam, SGD
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

GENERATE_LIMIT = 200
prompt_batch_size = 16

curr_directory = os.path.abspath("../llm")
# vocab = torch.load(f"{curr_directory}\\vocab_obj")

vocab = torch.load("llm/vocab_obj")
preference_model = AbstractRewardModel(len(vocab), GENERATE_LIMIT)

initial_model = BaseLGM(len(vocab))
# initial_model.to(device)
prompts = torch.randint(0, 300, (12, prompt_batch_size))

current_model = copy.deepcopy(initial_model)
# current_model.to(device)
# actor_optimizer = Adam(current_model.parameters(), lr=1e-4)

critic_model = AbstractRewardModel(len(vocab), GENERATE_LIMIT)
# current_model.to(device)
# critic_optimizer = Adam(critic_model.parameters(), lr=1e-4)

# for epoch in range(10):
#     # below comes the formal update procedure, for one batch of data.
#     total_preference_score = []
#     critic_prediction_score = []
#     for prompt_idx in range(prompt_batch_size):
#         with torch.no_grad():
#             initial_model_response = initial_model.rlhf_generate(prompts[:, prompt_idx].unsqueeze(1), vocab,
#                                                                  generate_limit=GENERATE_LIMIT)
#         curr_model_response = current_model.rlhf_generate(prompts[:, prompt_idx].unsqueeze(1), vocab,
#                                                           generate_limit=GENERATE_LIMIT)
#         preference_score = preference_model.forward(prompts[:, prompt_idx].unsqueeze(1), initial_model_response,
#                                                     curr_model_response)  # this is the reward signal
#         critic_prediction = critic_model.forward(
#             prompts[:, prompt_idx].unsqueeze(1), initial_model_response, curr_model_response)
#         # outputs one reward signal, used for future finetuning
#
#         total_preference_score.append(preference_score)  # GT reward
#         critic_prediction_score.append(critic_prediction)  # critic value
#
#     # now perform parameter update. Realizing the preference score should be maximized.
#     total_preference_score = torch.stack(total_preference_score)
#     critic_prediction_score.append(torch.tensor(0.0))
#     critic_prediction_score = torch.stack(critic_prediction_score)
#
#     # first update critic model. Calculate cumulative reward and use MSE
#     cumulative_reward = reward_to_go_calculation(total_preference_score)
#     critic_loss = nn.MSELoss()(cumulative_reward, critic_prediction_score[:-1])
#
#     critic_optimizer.zero_grad()
#     critic_loss.backward(retain_graph=True)
#     critic_optimizer.step()
#     # now update actor model. calculate advantage by: A(t) = r(t) + V(t+1) - V(t), t is time;
#     advantage = total_preference_score + \
#                 critic_prediction_score[1:] - critic_prediction_score[:-1]
#     policy_loss = torch.sum(advantage * total_preference_score)
#     actor_optimizer.zero_grad()
#     policy_loss.backward()
#     actor_optimizer.step()
#
#     print(policy_loss, critic_loss)