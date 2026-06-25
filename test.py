import torch
from torch.utils.data import Dataset, DataLoader
from transformer import Transformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"

MAX_LEN = 50


# ---------------- ENCODE ----------------
def encode(sentence, vocab):
    tokens = [SOS] + sentence.strip().split() + [EOS]
    ids = [vocab.get(tok, vocab[UNK]) for tok in tokens]
    ids = ids[:MAX_LEN]
    ids += [vocab[PAD]] * (MAX_LEN - len(ids))
    return torch.tensor(ids)


# ---------------- LOAD MODEL ----------------
checkpoint = torch.load("transformer.pt", map_location=DEVICE)

src_vocab = checkpoint["src_vocab"]
tgt_vocab = checkpoint["tgt_vocab"]
id_to_word = {v: k for k, v in tgt_vocab.items()}

model = Transformer(
    input_vocab_size=len(src_vocab),
    output_vocab_size=len(tgt_vocab)
).to(DEVICE)

model.load_state_dict(checkpoint["model"])
model.eval()


# ---------------- BEAM SEARCH ----------------
def beam_search(model, src, beam_size=5, max_len=50):
    src = src.to(DEVICE)

    sos = tgt_vocab[SOS]
    eos = tgt_vocab[EOS]

    beams = [([sos], 0.0)]

    for _ in range(max_len):

        new_beams = []

        for seq, score in beams:

            if seq[-1] == eos:
                new_beams.append((seq, score))
                continue

            tgt_input = torch.tensor(seq).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(src, tgt_input)

            probs = torch.softmax(output[:, -1, :], dim=-1)

            topk_probs, topk_idx = torch.topk(probs, beam_size)

            for i in range(beam_size):
                token = topk_idx[0, i].item()
                prob = topk_probs[0, i].item()

                new_seq = seq + [token]
                new_score = score + torch.log(torch.tensor(prob + 1e-9)).item()

                new_beams.append((new_seq, new_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        if all(b[0][-1] == eos for b in beams):
            break

    return beams[0][0]


# ---------------- DECODE ----------------
def decode(seq):
    words = []

    for idx in seq:
        word = id_to_word.get(idx, UNK)

        if word == EOS:
            break
        if word not in [PAD, SOS]:
            words.append(word)

    return words


# ---------------- DATASET ----------------
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
        src = encode(self.src[idx], self.src_vocab)
        tgt = encode(self.tgt[idx], self.tgt_vocab)
        return src, tgt


# ---------------- TEST DATA ----------------
test_dataset = TranslationDataset(
    "test.en",
    "test.de",
    src_vocab,
    tgt_vocab
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# ---------------- RUN INFERENCE ----------------
print("\n=== SAMPLE TRANSLATIONS ===\n")

for i, (src, tgt) in enumerate(test_loader):

    src = src.to(DEVICE)

    pred_seq = beam_search(model, src, beam_size=5)
    pred_sentence = decode(pred_seq)

    print(f"Sample {i+1}")
    print("Prediction:", " ".join(pred_sentence))
    print()

    if i == 5:  # show first 6 samples only
        break


# ---------------- OPTIONAL: BLEU ----------------
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    print("\n=== BLEU EVALUATION ===\n")

    smoothie = SmoothingFunction().method4
    total_bleu = 0
    count = 0

    for src, tgt in test_loader:

        src = src.to(DEVICE)

        pred_seq = beam_search(model, src, beam_size=5)
        pred_sentence = decode(pred_seq)

        ref = decode(tgt.squeeze().tolist())

        bleu = sentence_bleu([ref], pred_sentence, smoothing_function=smoothie)
        total_bleu += bleu
        count += 1

    avg_bleu = total_bleu / count
    print(f"Average BLEU (0–1): {avg_bleu:.4f}")
    print(f"Average BLEU (0–100): {avg_bleu * 100:.2f}")

except ImportError:
    print("NLTK not installed. BLEU skipped.")