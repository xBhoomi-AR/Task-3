import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

from transformer import Transformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"

MAX_LEN = 50
BATCH_SIZE = 32
EPOCHS = 10


def build_vocab(file_path):
    counter = Counter()

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            counter.update(line.strip().split())

    vocab = {
        PAD: 0,
        SOS: 1,
        EOS: 2,
        UNK: 3
    }

    for word in counter:
        if word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def encode(sentence, vocab):
    tokens = [SOS]
    tokens.extend(sentence.strip().split())
    tokens.append(EOS)

    ids = [vocab.get(tok, vocab[UNK]) for tok in tokens]

    ids = ids[:MAX_LEN]

    ids += [vocab[PAD]] * (MAX_LEN - len(ids))

    return torch.tensor(ids)


class TranslationDataset(Dataset):

    def __init__(self, src_file, tgt_file, src_vocab, tgt_vocab):

        with open(src_file, encoding="utf-8") as f:
            self.src = f.readlines()

        with open(tgt_file, encoding="utf-8") as f:
            self.tgt = f.readlines()

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):

        src = encode(
            self.src[idx],
            self.src_vocab
        )

        tgt = encode(
            self.tgt[idx],
            self.tgt_vocab
        )

        return src, tgt


src_vocab = build_vocab("train.en")
tgt_vocab = build_vocab("train.de")

train_dataset = TranslationDataset(
    "train.en",
    "train.de",
    src_vocab,
    tgt_vocab
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = Transformer(
    input_vocab_size=len(src_vocab),
    output_vocab_size=len(tgt_vocab)
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    ignore_index=0
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for src, tgt in train_loader:

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        optimizer.zero_grad()

        output = model(
            src,
            tgt[:, :-1]
        )

        loss = criterion(
            output.reshape(-1, output.shape[-1]),
            tgt[:, 1:].reshape(-1)
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}"
    )

torch.save(
    {
        "model": model.state_dict(),
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab
    },
    "transformer.pt"
)

print("Model saved.")

