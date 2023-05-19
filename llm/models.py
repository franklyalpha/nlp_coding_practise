import torch
import torch.nn as nn
from llm.prepare_text_data import *


class BaseLGM(nn.Module):
    def __init__(self, vocab_len, model_dim=512, n_head=8,
                 dim_feedforward=2048, enc_layer=6, dec_layer=6):
        super(BaseLGM, self).__init__()
        # will need to define embedding layers apart from transformer layers. Also it would be recommended
        # if you have time, that you can consider implementing multi-head transformer mechanisms from scratch
        # for practises.
        self.num_words = vocab_len
        self.word_embedding = nn.Embedding(self.num_words, model_dim)
        self.transformer_model = nn.Transformer(d_model=model_dim, nhead=n_head,
                                                dim_feedforward=dim_feedforward,
                                                num_encoder_layers=enc_layer, num_decoder_layers=dec_layer)
        self.embedding_decode = nn.Linear(model_dim, self.num_words)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # first require an embedding layer
        # need to ensure batch sizes are the same
        assert src.shape[1] == tgt.shape[1]
        src_embed = self.word_embedding(src)
        tgt_embed = self.word_embedding(tgt)
        transformer_res = self.transformer_model(src_embed, tgt_embed, src_mask, tgt_mask)
        decode_res = self.embedding_decode(transformer_res)
        # returned result has shape: [dec_seq_len, batch, vocab_len]
        return nn.functional.softmax(decode_res, dim=-1)  # this is for loss calculation.

    def rlhf_forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # first require an embedding layer
        # need to ensure batch size are the same
        assert src.shape[1] == tgt.shape[1]
        # only fine-tune transformer, so eliminate gradient signal for embedding.
        for param in self.word_embedding.parameters():
            param.requires_grad = False
        for param in self.embedding_decode.parameters():
            param.requires_grad = False

        decode_res = self.forward(src, tgt, src_mask, tgt_mask)

        # in case there are follow-up trainings, reset requires_grad to True.
        for param in self.word_embedding.parameters():
            param.requires_grad = True
        for param in self.embedding_decode.parameters():
            param.requires_grad = True

        return decode_res  # [dec_seq_len, batch, vocab_len]

    @torch.no_grad()
    def generate(self, src, vocab, generate_limit=200, src_mask=None):
        # need to ensure inputs have shape: [seq_len, 1], representing [sequence length, batch)
        # assert src.shape[1] == 1
        # tgt = torch.zeros((generate_limit, 1), dtype=torch.int64).to(src.device)
        # tgt[0, :] = vocab.lookup_indices([BOS_TOKEN])[0]
        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(generate_limit, src.device)
        soft_res = self.rlhf_generate(src, vocab, generate_limit, src_mask)
        return vocab.lookup_tokens(torch.argmax(soft_res, dim=-1).flatten().tolist())

    def rlhf_generate(self, src, vocab, generate_limit=200, src_mask=None):
        # need to ensure inputs have shape: [seq_len, 1], representing [sequence length, batch)
        # also need gradients for updating model.
        assert src.shape[1] == 1
        tgt = torch.zeros((generate_limit, 1), dtype=torch.int64).to(src.device)
        tgt[0, :] = vocab.lookup_indices([BOS_TOKEN])[0]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(generate_limit, src.device)
        return self.rlhf_forward(src, tgt, src_mask, tgt_mask)  # [dec_seq_len, batch, vocab_len]


class GPT(BaseLGM):
    """
    Compared with basic language generative models which might adopt encoder-decoder architecture,
    GPT only keeps transformer decoder as the main generative architecture.
    When generating results, GPT adopts autoregressive generation process that loops to generate tokens one by one.
    """
    def __init__(self, vocab_len, model_dim=512, n_head=8,
                 dim_feedforward=2048, enc_layer=0, dec_layer=6, refer_prev_len=100):
        """

        :param vocab_len:
        :param model_dim:
        :param n_head:
        :param dim_feedforward:
        :param enc_layer:
        :param dec_layer:
        :param refer_prev_len: when GPT is doing autoregressive generation, the maximum size of
            memory/input/previous context sequence to refer to.
        """
        super(GPT, self).__init__(vocab_len, model_dim, n_head, dim_feedforward, enc_layer, dec_layer)
        transformer_block = nn.TransformerDecoderLayer(model_dim, n_head, dim_feedforward)
        self.transformer_model = nn.TransformerDecoder(transformer_block, dec_layer)
        self.refer_prev_len = refer_prev_len

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        During training, given a training sequence with length n, src = seq[:-1], tgt = seq[1:]
        Given GPT contains only decoder structure, will adopt src as "memory" of decoder input.


        :param src:
        :param tgt:
        :param src_mask:
        :param tgt_mask:
        :return: shape: [dec_seq_len, batch, vocab_len], "softmax"ed probabilities
        """
        memory = self.word_embedding(src)
        tgt_embedded = self.word_embedding(tgt)
        transformer_res = self.transformer_model(memory, tgt_embedded)
        decode_res = self.embedding_decode(transformer_res)
        return nn.functional.softmax(decode_res, dim=-1)

    def rlhf_generate(self, src, vocab, generate_limit=200, src_mask=None):
        """
        Autoregressively generate maximum "generate_limit" many tokens.
        In each iteration, only the last token's softmax scores will be extracted and used for
        :param src: shape: [sequence length, batch]; batch is usually just 1
        :param vocab:
        :param generate_limit:
        :param src_mask:
        :return: shape: [dec_seq_len, batch, vocab_len]
        """
        assert src.shape[1] == 1
        prev_context = src  # [source_seq_len, batch]
        # the source sequence to refer to when doing autoregressive generation
        tgt = torch.zeros((generate_limit, 1), dtype=torch.int64).to(src.device)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(generate_limit, src.device)

        generated_result = []

        for idx in range(generate_limit):
            if len(prev_context) > self.refer_prev_len:
                prev_context = prev_context[1:]
            decode_res = self.rlhf_forward(prev_context, tgt, src_mask, tgt_mask)
            curr_token = decode_res[-1]  # only take the last token as what to predict next
            generated_result.append(curr_token)
            tgt[idx] = torch.argmax(curr_token, dim=-1).flatten().item()
            prev_context = torch.cat([prev_context, tgt[idx][None]], dim=0)

        return torch.stack(generated_result)  # increase dimension by 1


if __name__ == "__main__":
    # for debugging purposes only
    curr_directory = os.path.abspath("")
    vocab = torch.load(f"{curr_directory}\\vocab_obj")
    print("finished loading vocab\n")
    model = GPT(len(vocab))
    src = torch.randint(0, 300, (12, 5))
    tgt = torch.randint(0, 400, (10, 5))
    forward_test = model(src, tgt)
    argmax_res = torch.argmax(forward_test, dim=-1)
    print("forward test complete")
    print(forward_test.shape)
    print(torch.sum(forward_test[0][0]))
    print("")
    generate_test = model.generate(src[:, 0].unsqueeze(-1), vocab, generate_limit=20)
    print("generate test complete. ")
    print(tgt[:, 0])
    print(str(generate_test) + "\n")
