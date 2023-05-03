import torch
import torch.nn as nn
from llm.prepare_text_data import *


class BaseLGM(nn.Module):
    def __init__(self, vocab, model_dim=512, n_head=8,
                 dim_feedforward=2048, enc_layer=6, dec_layer=6):
        super(BaseLGM, self).__init__()
        # will need to define embedding layers apart from transformer layers. Also it would be recommended
        # if you have time, that you can consider implementing multi-head transformer mechanisms from scratch
        # for practises.
        num_words = len(vocab)
        self.word_embedding = nn.Embedding(num_words, model_dim)
        self.transformer_model = nn.Transformer(d_model=model_dim, nhead=n_head,
                                                dim_feedforward=dim_feedforward,
                                                num_encoder_layers=enc_layer, num_decoder_layers=dec_layer)
        self.embedding_decode = nn.Linear(model_dim, num_words)

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

    def rlhf_forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # first require an embedding layer
        # need to ensure batch size are the same
        assert src.shape[1] == tgt.shape[1]
        src_embed = self.word_embedding(src)
        tgt_embed = self.word_embedding(tgt)
        transformer_res = self.transformer_model(src_embed, tgt_embed, src_mask, tgt_mask)
        return transformer_res  # [dec_seq_len, batch, model_dim]

    @torch.no_grad()
    def generate(self, src, vocab, generate_limit=200, src_mask=None, tgt_mask=None):
        # need to ensure inputs have shape: [seq_len, 1], representing [sequence length, batch)
        assert src.shape[1] == 1
        tgt = torch.zeros((generate_limit, 1), dtype=torch.int64)
        tgt[0, :] = vocab.lookup_indices([BOS_TOKEN])[0]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(generate_limit, src.device)
        soft_res = self.forward(src, tgt, src_mask, tgt_mask)  # [dec_seq_len, 1, num_words]
        return vocab.lookup_tokens(torch.argmax(soft_res, dim=-1).flatten().tolist())


if __name__ == "__main__":
    # for debugging purposes only
    curr_directory = os.path.abspath("")
    vocab = torch.load(f"{curr_directory}\\vocab_obj")
    print("finished loading vocab\n")
    model = BaseLGM(vocab)
    src = torch.randint(0, 300, (12, 5))
    tgt = torch.randint(0, 400, (10, 5))
    forward_test = model(src, tgt)
    argmax_res = torch.argmax(forward_test, dim=-1)
    print("forward test complete")
    print(forward_test.shape)
    print(torch.sum(forward_test[0][0]))
    print("")
    generate_test = model.generate(src[:, 0].unsqueeze(-1), vocab)
    print("generate test complete. ")
    print(tgt[:, 0])
    print(str(generate_test) + "\n")
