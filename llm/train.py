import torch
import torch.nn as nn
from torch import optim
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset

from llm.prepare_text_data import *
from llm.models import *

curr_directory = os.path.abspath("")
vocab = torch.load(f"{curr_directory}\\vocab_obj")
print("vocab loaded!")
dataset = torch.load(f"{curr_directory}\\dataset_instance")
dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True, collate_fn=dataloader_collate_fn)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(len(vocab), 64, 4, 128, 3, 3)
model.to(device)
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
EPOCH = 10

print("number of batches: " + str(len(dataloader)))


# below will implement the training process. Realizing that one instance of batched data is required.
for epoch in range(EPOCH):
    total_loss = 0
    for index, (src_tensor, tgt_tensor) in enumerate(dataloader):
        # acquire text-completion training technique: masking out last one/few words in
        # sentence and request prediction for training.
        # handling src_tensor: realizing that EOS token should be eliminated, as well as some words.
        # don't forget the punctuations!!!
        src_tensor = src_tensor.T.long().to(device)
        tgt_tensor = tgt_tensor.T.long().to(device)
        # handle src_tensor
        src_tensor = src_tensor[:-4, :]  # predict last 3 characters, and EOS token
        tgt_mask_shape = tgt_tensor.shape[0]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_mask_shape, device)
        output = model(src_tensor, tgt_tensor, tgt_mask=tgt_mask)

        # prepare for defining loss criterion, using cross entropy loss
        optimizer.zero_grad()
        loss = cross_entropy(output.permute(1, 2, 0), tgt_tensor.permute(1, 0))
        loss.backward()
        optimizer.step()
        total_loss += loss
        if index % 100 == 0:
            print(f"epoch: {epoch}, batch index: {index}, loss: {loss.item()}")
    print("epoch {}".format(epoch) + " loss: " + str(total_loss))

