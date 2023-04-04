import torch
import torch.nn as nn
from prepare_text_data import *


class BaseLGM(nn.Module):
    def __init__(self, num_words, vocab, model_dim=512):
        super(BaseLGM, self).__init__()
        # will need to define embedding layers apart from transformer layers. Also it would be recommended
        # if you have time, that you can consider implementing multi-head transformer mechanisms from scratch
        # for practises.
        self.word_embedding = nn.Embedding(num_words, model_dim)
        self.transformer_model = nn.Transformer(d_model=model_dim)
        self.embedding_decode = nn.Linear(model_dim, num_words)
        self.vocab = vocab

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # first require an embedding layer
        # need to ensure batch size are the same
        assert src.shape[1] == tgt.shape[1]
        src_embed = self.word_embedding(src)
        tgt_embed = self.word_embedding(tgt)
        transformer_res = self.transformer_model(src_embed, tgt_embed, src_mask, tgt_mask)
        decode_res = self.embedding_decode(transformer_res)
        # returned result has shape: [dec_seq_len, batch, num_words]
        return nn.functional.softmax(decode_res, dim=-1)  # this is for loss calculation.

    def generate(self, src, tgt, src_mask=None, tgt_mask=None):
        # need to ensure inputs have shape: [seq_len, 1, 1], representing [sequence length, batch, token_size)
        assert src.shape[1] == 1 and tgt.shape[1] == 1
        soft_res = self.forward(src, tgt, src_mask, tgt_mask)  # [dec_seq_len, 1, num_words]
        return self.vocab.lookup_tokens(torch.argmax(soft_res, dim=-1).flatten().tolist())


if __name__ == "__main__":
    # for debugging purposes only
    curr_directory = os.path.abspath("")
    vocab = torch.load(f"{curr_directory}\\vocab_obj")
    print("finished loading vocab\n")
    model = BaseLGM(len(vocab), vocab)
    src = torch.randint(0, 300, (12, 5))
    tgt = torch.randint(0, 400, (10, 5))
    forward_test = model(src, tgt)
    print("forward test complete")
    print(forward_test.shape)
    print(torch.sum(forward_test[0][0]))
    print("")
    generate_test = model.generate(src[:, 0].unsqueeze(-1), tgt[:, 0].unsqueeze(-1))
    print("generate test complete. ")
    print(tgt[:, 0])
    print(str(generate_test) + "\n")
