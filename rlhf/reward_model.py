"""
Defining reward model:

start with an abstract class defining all the properties the reward model must include, and
need to write specifics for how language models should interact with it. Remember to handle the
devices properly.

Reward model should perform the following procedure:
After receiving the output of language model, it will use the network model to evaluate
the output, and gives a reward signal. The reward signal will then be used to train the
model. May consider having one method for providing the reward signal, and another
general method for fine-tuning the language model.

one problem is how should the fine tuning being conducted. Realizing in typical RL, a
trajectory needs to be sampled before training could go through (delayed reward).
Is this the same case for language model fine-tuning?

"""
import copy

import torch
import torch.nn as nn
import torchtext as ttext
import numpy as np
import llm.models as lgm


class RLHFEnv:
    def __init__(self):
        super(RLHFEnv, self).__init__()

    def step(self):
        pass

    def reset(self):
        pass

    def modify_prompt_dataset(self):
        pass


class AbstractRewardModel(nn.Module):
    def __init__(self, model_dim=512, token_limit=200):
        super(AbstractRewardModel, self).__init__()
        # just create a linear layer for producing a scalar output
        self._init_network(model_dim, token_limit)

    def _init_network(self, model_dim, token_limit):
        self.embedding_mapping = nn.Linear(model_dim, 1)
        self.sequence_mapping = nn.Linear(token_limit, 1)

    def forward(self, input_prompt, response_initial, response_tuned):
        """
        :param input_prompt: the prompt used for language generative model
            to output a result.
        :param response_initial: the response generated by pretrained model
        :param response_tuned: the response generated by current model, fine tuned.

        All above parameters should have shape [sequence_length, batch_size, model_size], where each element
        is a numeric index that corresponds to a token in the Vocab object.
            model_size is the output shape of the language generative model, which is usually 512.

        :return: a scalar score as reward signal, calculated following RLHF's paper,
            and adopting KL divergence of scores between two responses.
        """
        assert input_prompt.shape[1] == response_initial.shape[1] \
               and input_prompt.shape[1] == response_tuned.shape[1]
        self.sequence_mapping.weight.requires_grad = False
        self.embedding_mapping.weight.requires_grad = False
        res = self.embedding_mapping(response_tuned)
        res = self.sequence_mapping(res.permute(1, 2, 0))

        # below implements KL divergence calculation: following: sum(tuned_prob * log(tuned_prob / initial_prob)),
        # applied on "x", or samples selected by tuned model.
        # response_tuned has size [seq_len, batch_size, model_dim]
        tuned_selection_prob, selection_indices = torch.max(response_tuned, dim=-1)
        tuned_selection_prob = tuned_selection_prob.flatten()
        initial_selection_prob = torch.gather(response_initial, -1,
                                              selection_indices.unsqueeze(-1)).squeeze(-1).flatten()
        kl_terms = tuned_selection_prob * torch.log(tuned_selection_prob / (1e-8 + initial_selection_prob))
        kl_loss = torch.sum(kl_terms)

        return res.flatten().sum() + kl_loss
