# implements mechanisms for data preprocessing and loading. After performing those operations, the
# output should be a dataloader that could be directly used for training.
# need to realize that when using the language generative model, the same value assigned to the word should
# be used instead. Therefore must ensure the dictionary is being created that could include for as much
# words as possible, and should assign each one with a unique numeric.
# however the key right now would be, how to map those words into an embedding layer.
# preprocessing on all sentences can take a quite long time. Thus what would rather be preferred is to
# process on batched tensors, and save vocabs into a file for direct loading.

import torchtext
from torch.utils.data.dataset import T_co
from torchtext.data import get_tokenizer
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import DATASETS
from torchtext.vocab import build_vocab_from_iterator

import torch
import torch.nn as nn
import os  # for file loading procedures
from pathlib import Path, PureWindowsPath
import re

BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNKNOWN_TOKEN = "<unk>"
MASK_TOKEN = "<mask>"
BATCH_SIZE = 4


def yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        # debug = tokenizer(text)[2:]
        yield tokenizer(text)[2:-1]


def simple_text_preprocessing(lines, bos_eos):
    for idx in range(len(lines)):
        # remove punctuations, set to lower case, remove first indexing token.
        result = re.sub(r'[^\w\s]', '', lines[idx].lower()).split()[1:]
        # can optionally add BOS and EOS tokens
        if bos_eos:
            sequence_res = [BOS_TOKEN]
            sequence_res.extend(result)
            sequence_res.append(EOS_TOKEN)
            lines[idx] = sequence_res
        else:
            lines[idx] = result
        return lines


def load_text_files_by_lines(directory_path, bos_eos=False):
    """

    :param bos_eos: if True, will append "BOS" and "EOS" tokens at beginning and end of a sentence sequential token
    :param directory_path: must be a string of a directory under current folder: "llm"
    :return:
    """
    abs_dir_path = os.path.abspath(directory_path)
    # this line will be abolished, as the result is incompatible with windows 10's file path.
    # abs_dir_path = "llm/datasets"
    all_lines = []
    for path in os.listdir(abs_dir_path):
        if path.endswith(".txt"):
            file_path = f"{abs_dir_path}\\{path}"
            # file_path = f"{abs_dir_path}/{path}"

            # now start read files
            f = open(file_path, "r", encoding="utf-8")
            lines = f.readlines()
            # simple text preprocessing procedure
            # lines = simple_text_preprocessing(lines, bos_eos)
            all_lines.extend(lines)

    return all_lines


def translate_text_to_vocab(split_tokens, vocab):
    """
    realizing this method only converts data into numeric indexing for each token,
    still require embedding modules to convert into vector tensors for each sentence.
    :param all_lines: a list of strings
    :param vocab:
    :return:
    """
    vocab_result = [vocab([BOS_TOKEN])[0]]
    vocab_result.extend(vocab(split_tokens))
    vocab_result.append(vocab([EOS_TOKEN])[0])
    vocab_result = torch.Tensor(vocab_result).long()
    longest_len = vocab_result.size()[0]
    eos_token = vocab.lookup_indices([EOS_TOKEN])[0]
    return create_vector_tensor([vocab_result], eos_token, longest_len)


def create_vector_tensor(all_lines_tensor, padding_token, longest_len, pad_left=False):
    vector_tensor = torch.ones((len(all_lines_tensor), longest_len)) * padding_token
    for j in range(len(all_lines_tensor)):
        if pad_left:
            vector_tensor[j, (longest_len - all_lines_tensor[j].size()[0]):] = all_lines_tensor[j]
        else:
            vector_tensor[j, :all_lines_tensor[j].size()[0]] = all_lines_tensor[j]
    return vector_tensor


# dataset for loading texts for training
class LLMTrainingDataset(Dataset):
    """
    modified so that
    """

    def __init__(self, all_lines_text, vocab, tokenizer):
        super(LLMTrainingDataset, self).__init__()
        self.all_lines_text = all_lines_text
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.all_lines_text)

    def vocab_size(self):
        return len(self.vocab)

    def __getitem__(self, index) -> T_co:
        line = self.all_lines_text[index]
        split_line = self.tokenizer(line)[2:-1]
        tokenized_line = translate_text_to_vocab(split_line, self.vocab)
        return tokenized_line, line


def dataloader_collate_fn(data):
    """
    data is a list containing tuples of dataset's outputs, consistent with output of dataset's
    __getitem__() method.
    """
    # first iterate through all tensors and determine the longest sequence length.
    longest_len = 0
    all_lines_tensor = []
    first_line_token, _ = data[0]
    bos_token = first_line_token[:, 0]
    eos_token = first_line_token[:, -1]
    for tensor, text in data:
        # tensor is a torch tensor, each numeric represents a tokenized word.
        # text is a str corresponding to tensor.
        longest_len = max(longest_len, tensor.shape[-1])
        all_lines_tensor.append(tensor.flatten())
    return create_vector_tensor(all_lines_tensor, bos_token, longest_len, pad_left=True), \
        create_vector_tensor(all_lines_tensor, eos_token, longest_len, pad_left=False)
    # src, tgt respectively.


if __name__ == "__main__":
    all_lines = load_text_files_by_lines("datasets", bos_eos=True)  # a list of str!
    tokenizer = get_tokenizer("spacy")
    tokenize_iter = iter(all_lines)

    curr_directory = os.path.abspath("")
    vocab = torch.load(f"{curr_directory}\\vocab_obj")
    print("vocab loaded!")

    # below is the code for creating vocab object from loaded text datasets.
    # vocab = build_vocab_from_iterator(yield_tokens(tokenize_iter, tokenizer), specials=[UNKNOWN_TOKEN])
    # vocab.set_default_index(vocab[UNKNOWN_TOKEN])
    # vocab.append_token(BOS_TOKEN)
    # vocab.append_token(EOS_TOKEN)
    # vocab.append_token(MASK_TOKEN)
    # torch.save(vocab, f"{curr_directory}\\vocab_obj")
    # print("vocab object saved")
    """
    above line assigns all unknown tokens to "<unk>", a safety measure. Realizing "specials" parameter must
    include default vocab. Besides, this process can take quite a long period of time to process. Beware.
    
    below is a sample usage of vocab, for words that must exist in input data:
    print(vocab(["the", "EOS"]))  
    which returns a list of indexes for the two words. Looks like frequency is adopted for making the order.
    """
    # once vocabs are being acquired, need to "translate" each sentence into vectorized format.
    # dataset = LLMTrainingDataset(all_lines, vocab, tokenizer)
    # torch.save(dataset, f"{curr_directory}\\dataset_instance")
    # print("dataset saved!")

    dataset = torch.load(f"{curr_directory}\\dataset_instance")
    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True, collate_fn=dataloader_collate_fn)
    for index, (src_tensor, tgt_tensor) in enumerate(dataloader):  # this is the standard method for loading data.
        # tensor has shape: [batch_size, sequence_len]; text: Tuple[str] with <batch_size> strings corresponding
        # to each tensor.
        a = src_tensor.T  # [seq_len, batch_size]
        b = tgt_tensor.T

