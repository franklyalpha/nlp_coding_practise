import os.path

import torch
import torch.nn as nn
from llm.models import *
import copy
from rlhf.reward_model import *
from torch.optim import Adam, SGD
import torch.nn.functional as F


class RLHFTrainer:
    def __init__(self, vocab, preference_model,
                 agent_model, critic_model,
                 generate_limit=200, prompt_batch_size=16,
                 epochs=10, decay_factor=0.999):
        super(RLHFTrainer, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generate_limit = generate_limit
        self.prompt_batch_size = prompt_batch_size
        self.vocab = vocab
        self.preference_model = preference_model.to(self.device)
        self.actor_model = agent_model.to(self.device)
        self.initial_model = copy.deepcopy(agent_model)
        self.initial_model.to(self.device)
        for parameter in self.initial_model.parameters():
            parameter.requires_grad = False
        self.critic_model = critic_model.to(self.device)
        self.actor_optimizer = Adam(self.actor_model.parameters(), lr=1e-4)
        self.critic_optimizer = Adam(self.critic_model.parameters(), lr=1e-4)
        self.epochs = epochs
        self.decay_factor = decay_factor

    def reward_to_go_calculation(self, gt_reward_stack):
        """
        :param gt_reward_stack: a 1D torch tensor where index "i" indicates the state reward at
            time "i", or at "ith" prompt.
        :return: cumulative reward with same shape as gt_reward_stack
        """
        assert gt_reward_stack.size()[0] == self.prompt_batch_size
        reward_decay_tensor = torch.from_numpy(
            np.array([self.decay_factor ** i for i in range(self.prompt_batch_size)])).to(self.device)
        reward_to_go = torch.zeros_like(gt_reward_stack).to(self.device)
        for i in range(self.prompt_batch_size - 1, -1, -1):
            reward_to_go[i] = torch.sum(reward_decay_tensor[:self.prompt_batch_size - i]
                                        * gt_reward_stack[i:], dim=0)
        return reward_to_go

    def finetune(self):

        prompts = torch.randint(0, 300, (12, self.prompt_batch_size)).to(self.device)
        self._finetune_one_batch(prompts)

    def _finetune_one_batch(self, input_prompt):
        """

        :param input_prompt: a tensor containing integers of size [prompt_length, prompt_batch_size]
        :return:
        """
        total_preference_score = []
        critic_prediction_score = []  # for updating critic model
        response_likelihood = []
        # for calculating the probability of generating the response, simulating policy in RL

        # below perform trajectory evaluation, a standard step for adopting reinforcement learning algorithm
        for prompt_idx in range(self.prompt_batch_size):
            curr_prompt = input_prompt[:, prompt_idx].unsqueeze(1)
            with torch.no_grad():
                initial_model_response = self.initial_model.rlhf_generate(curr_prompt, vocab,
                                                                          generate_limit=self.generate_limit)
            curr_model_response = self.actor_model.rlhf_generate(curr_prompt, vocab,
                                                                 generate_limit=self.generate_limit)
            # [generate_limit, batch, vocab_len]; softmaxed probabilities
            response_likelihood.append(torch.prod(torch.max(curr_model_response, dim=-1)[0].flatten()))
            preference_score = self.preference_model.forward(curr_prompt, initial_model_response,
                                                             curr_model_response)  # this is the reward signal
            critic_prediction = self.critic_model.forward(
                curr_prompt, initial_model_response, curr_model_response)
            # outputs one reward signal, used for future finetuning

            total_preference_score.append(preference_score)  # GT reward
            critic_prediction_score.append(critic_prediction)  # critic value

        # now perform parameter update. Realizing the preference score should be maximized.
        total_preference_score = torch.stack(total_preference_score)  # [self.prompt_batch_size]
        critic_prediction_score.append(torch.tensor(0.0).to(self.device))  # this is for calculating advantage value later: V(t+1)
        critic_prediction_score = torch.stack(critic_prediction_score)  # [self.prompt_batch_size + 1]

        # first update critic model. Calculate cumulative reward and use MSE
        cumulative_reward = self.reward_to_go_calculation(total_preference_score)
        critic_loss = nn.MSELoss()(cumulative_reward, critic_prediction_score[:-1])

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)  # gradients need to be saved for performing policy updates.
        self.critic_optimizer.step()

        # now update actor model. calculate advantage by: A(t) = r(t) + V(t+1) - V(t), t is time;
        advantage = total_preference_score + \
            critic_prediction_score[1:] - critic_prediction_score[:-1]

        # here a loss function different from log-gradient is adopted.
        policy_loss = torch.sum(advantage * torch.stack(response_likelihood))
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()


# for debug purposes only
if __name__ == "__main__":
    curr_directory = os.path.abspath("../llm")
    vocab = torch.load(f"{curr_directory}\\vocab_obj")
    GENERATE_LIMIT = 10
    prompt_batch_size = 8
    preference_model = AbstractRewardModel(len(vocab), GENERATE_LIMIT)
    initial_model = GPT(len(vocab))
    critic_model = AbstractRewardModel(len(vocab), GENERATE_LIMIT)

    rlhf_trainer = RLHFTrainer(vocab, preference_model, initial_model, critic_model, generate_limit=GENERATE_LIMIT)
    rlhf_trainer.finetune()
