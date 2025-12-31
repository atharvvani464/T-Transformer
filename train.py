import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset, concatenate_datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
torch.manual_seed(42)

MAX_LEN = 64
BATCH_SIZE = 64
EPOCHS = 120
VOCAB_SIZE = 16000
WARMUP_STEPS = 4000

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"
CA = "<ca>"
DE = "<de>"

BASE_DIR = Path(".")
MODEL_DIR = BASE_DIR / "models"
TOKENIZER_DIR = BASE_DIR / "tokenizers"
CKPT_DIR = BASE_DIR / "checkpoints"

MODEL_DIR.mkdir(exist_ok=True)
TOKENIZER_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

def train_bpe(path, texts):
    tok = Tokenizer(BPE(unk_token=UNK))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[PAD, SOS, EOS, UNK, CA, DE]
    )
    tok.train_from_iterator(texts, trainer)
    tok.save(str(path))
    return tok

def get_tokenizer(path, dataset, lang):
    if path.exists():
        return Tokenizer.from_file(str(path))
    texts = (ex["translation"][lang] for ex in dataset)
    return train_bpe(path, texts)

def encode(tok, text, lang_token):
    ids = tok.encode(text).ids[: MAX_LEN - 3]
    return [
        tok.token_to_id(SOS),
        tok.token_to_id(lang_token),
        *ids,
        tok.token_to_id(EOS)
    ]

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, data, tok_src, tok_tgt):
        self.data = data
        self.tok_src = tok_src
        self.tok_tgt = tok_tgt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]["translation"]
        if "ca" in ex:
            src = encode(self.tok_src, ex["ca"], CA)
        else:
            src = encode(self.tok_src, ex["de"], DE)

        tgt = encode(self.tok_tgt, ex["en"], SOS)
        return torch.tensor(src), torch.tensor(tgt)

def collate(batch):
    src, tgt = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, padding_value=0)
    tgt = nn.utils.rnn.pad_sequence(tgt, padding_value=0)
    return src.to(DEVICE), tgt.to(DEVICE)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[: x.size(0)]

class MultilingualTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, heads=8, layers=6):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=heads,
            num_encoder_layers=layers,
            num_decoder_layers=layers,
            dim_feedforward=2048,
            dropout=0.1
        )

        self.fc = nn.Linear(d_model, tgt_vocab)
        self.fc.weight = self.tgt_emb.weight

    def forward(self, src, tgt, src_pad, tgt_pad):
        src = self.pos(self.src_emb(src))
        tgt = self.pos(self.tgt_emb(tgt))

        tgt_mask = self.transformer.generate_square_subsequent_mask(
            tgt.size(0)
        ).to(DEVICE)

        out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad
        )
        return self.fc(out)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab, pad_id, eps=0.1):
        super().__init__()
        self.pad = pad_id
        self.eps = eps
        self.vocab = vocab

    def forward(self, logits, target):
        logp = torch.log_softmax(logits, dim=-1)
        nll = -logp.gather(-1, target.unsqueeze(1)).squeeze(1)
        smooth = -logp.mean(dim=-1)
        mask = target != self.pad
        return ((1 - self.eps) * nll + self.eps * smooth)[mask].mean()

class TransformerScheduler:
    def __init__(self, opt, d_model):
        self.opt = opt
        self.step_num = 0
        self.d_model = d_model

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * WARMUP_STEPS ** -1.5
        )
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.step()

@torch.no_grad()
def bleu_eval(model, loader, tok):
    smooth = SmoothingFunction().method4
    refs, hyps = [], []

    for src, tgt in loader:
        src_pad = (src == 0).transpose(0, 1)
        ys = torch.tensor([[tok.token_to_id(SOS)]], device=DEVICE)

        for _ in range(MAX_LEN):
            tgt_pad = (ys == 0).transpose(0, 1)
            out = model(src, ys, src_pad, tgt_pad)[-1]
            next_tok = out.argmax(-1).item()
            ys = torch.cat([ys, torch.tensor([[next_tok]], device=DEVICE)], 0)
            if next_tok == tok.token_to_id(EOS):
                break

        hyp = [
            tok.id_to_token(i)
            for i in ys.squeeze().tolist()
            if i not in (0, tok.token_to_id(SOS), tok.token_to_id(EOS))
        ]
        ref = [
            tok.id_to_token(i)
            for i in tgt[:, 0].tolist()
            if i not in (0, tok.token_to_id(SOS), tok.token_to_id(EOS))
        ]

        refs.append([ref])
        hyps.append(hyp)

    return corpus_bleu(refs, hyps, smoothing_function=smooth)

def save_checkpoint(epoch, model, opt, sched, scaler):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.step_num,
        "scaler": scaler.state_dict()
    }, CKPT_DIR / "latest.pt")

def load_checkpoint(model, opt, sched, scaler):
    ckpt = CKPT_DIR / "latest.pt"
    if not ckpt.exists():
        return 1
    data = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(data["model"])
    opt.load_state_dict(data["optimizer"])
    scaler.load_state_dict(data["scaler"])
    sched.step_num = data["scheduler"]
    return data["epoch"] + 1

print("Loading datasets...")
ca = load_dataset("opus_books", "ca-en")["train"]
de = load_dataset("opus_books", "de-en")["train"]

full = concatenate_datasets([ca, de]).train_test_split(0.1, seed=42)

tok_src = get_tokenizer(TOKENIZER_DIR / "src.json", full["train"], "ca")
tok_en = get_tokenizer(TOKENIZER_DIR / "en.json", full["train"], "en")

train_ds = TranslationDataset(full["train"], tok_src, tok_en)
val_ds = TranslationDataset(full["test"], tok_src, tok_en)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, 1, collate_fn=collate)

model = MultilingualTransformer(
    tok_src.get_vocab_size(),
    tok_en.get_vocab_size()
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
scheduler = TransformerScheduler(optimizer, 512)
scaler = torch.cuda.amp.GradScaler()
criterion = LabelSmoothingLoss(tok_en.get_vocab_size(), 0)

start_epoch = load_checkpoint(model, optimizer, scheduler, scaler)

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    total = 0

    for src, tgt in train_loader:
        optimizer.zero_grad()
        src_pad = (src == 0).transpose(0, 1)
        tgt_pad = (tgt[:-1] == 0).transpose(0, 1)

        with torch.cuda.amp.autocast():
            out = model(src, tgt[:-1], src_pad, tgt_pad)
            loss = criterion(
                out.reshape(-1, out.size(-1)),
                tgt[1:].reshape(-1)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item()

    bleu = bleu_eval(model, val_loader, tok_en)
    save_checkpoint(epoch, model, optimizer, scheduler, scaler)

    print(
        f"Epoch {epoch:03d} | "
        f"Loss {total/len(train_loader):.3f} | "
        f"BLEU {bleu:.3f}"
    )

torch.save(model.state_dict(), MODEL_DIR / "multilingual_ca_de_en.pt")
print("Model and tokenizers saved. Ready for inference or deployment.")
