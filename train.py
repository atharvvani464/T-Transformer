import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from model import build_transformer

# ensure directories exist
Path("models").mkdir(exist_ok=True)
Path("tokenizers").mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
bleu_smoother = SmoothingFunction().method1

def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]

def get_or_create_tokenizer(tokenizer_path, ds, lang):
    tokenizer_path = Path(tokenizer_path)
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
            min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    return Tokenizer.from_file(str(tokenizer_path))

def get_ds(lang_src, lang_tgt, val_split=0.1):
    ds_raw = load_dataset("opus_books", f"{lang_src}-{lang_tgt}")
    ds = ds_raw["train"]
    split = ds.train_test_split(test_size=val_split, seed=42)
    return split["train"], split["test"], lang_src, lang_tgt

def encode_sentence(tokenizer, text, seq_len):
    ids = tokenizer.encode(text).ids
    pad_id = tokenizer.token_to_id("[PAD]")
    if len(ids) < seq_len:
        ids += [pad_id] * (seq_len - len(ids))
    else:
        ids = ids[:seq_len]
    return torch.tensor(ids, dtype=torch.long)

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, ds, tok_src, tok_tgt, src_lang, tgt_lang, seq_len):
        self.ds = ds
        self.tok_src = tok_src
        self.tok_tgt = tok_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        src = encode_sentence(self.tok_src, item["translation"][self.src_lang], self.seq_len)
        tgt = encode_sentence(self.tok_tgt, item["translation"][self.tgt_lang], self.seq_len)
        return src, tgt

def create_padding_mask(seq, pad_id):
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)

def create_tgt_mask(tgt, pad_id):
    batch, seq_len = tgt.shape
    pad_mask = create_padding_mask(tgt, pad_id)
    causal = torch.tril(torch.ones(seq_len, seq_len)).bool()
    causal = causal.unsqueeze(0).unsqueeze(1)
    return pad_mask & causal

def build_language_transformer(lang_src, lang_tgt, seq_len=64):
    train_ds, val_ds, src_key, tgt_key = get_ds(lang_src, lang_tgt)

    tok_src = get_or_create_tokenizer(f"tokenizers/{lang_src}_tokenizer.json", train_ds, src_key)
    tok_tgt = get_or_create_tokenizer(f"tokenizers/{lang_tgt}_tokenizer.json", train_ds, tgt_key)

    model = build_transformer(
        src_vocab_size=tok_src.get_vocab_size(),
        tgt_vocab_size=tok_tgt.get_vocab_size(),
        src_seq_len=seq_len,
        tgt_seq_len=seq_len,
        d_model=256,
        N=4,
        h=4,
        dropout=0.1,
        d_ff=512
    ).to(device)

    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(train_ds, tok_src, tok_tgt, src_key, tgt_key, seq_len),
        batch_size=16, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        TranslationDataset(val_ds, tok_src, tok_tgt, src_key, tgt_key, seq_len),
        batch_size=16
    )

    return model, tok_src, tok_tgt, train_loader, val_loader

def train_one_epoch(model, loader, optimizer, tok_tgt):
    model.train()
    pad_id = tok_tgt.token_to_id("[PAD]")
    total_loss = 0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

        src_mask = create_padding_mask(src, pad_id).to(device)
        tgt_mask = create_tgt_mask(tgt_in, pad_id).to(device)

        enc = model.encode(src, src_mask)
        dec = model.decode(enc, src_mask, tgt_in, tgt_mask)
        logits = model.project(dec)

        loss = nn.CrossEntropyLoss(ignore_index=pad_id)(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate_bleu(model, loader, tok_src, tok_tgt):
    model.eval()
    pad_id = tok_tgt.token_to_id("[PAD]")
    scores = []

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask = create_padding_mask(src, pad_id).to(device)

            enc = model.encode(src, src_mask)
            dec = model.decode(enc, src_mask, tgt[:, :-1], None)
            preds = model.project(dec).argmax(dim=-1)

            for i in range(src.size(0)):
                hyp = [t for t in preds[i].tolist() if t != pad_id]
                ref = [[t for t in tgt[i].tolist() if t != pad_id]]
                scores.append(sentence_bleu(ref, hyp, smoothing_function=bleu_smoother))

    return sum(scores) / max(len(scores), 1)

# build models
model_ca, tok_ca_src, tok_ca_tgt, loader_ca_tr, loader_ca_val = build_language_transformer("ca", "en")
model_de, tok_de_src, tok_de_tgt, loader_de_tr, loader_de_val = build_language_transformer("de", "en")

opt_ca = torch.optim.Adam(model_ca.parameters(), lr=1e-4)
opt_de = torch.optim.Adam(model_de.parameters(), lr=1e-4)

epochs = 20

for epoch in range(epochs):
    loss_ca = train_one_epoch(model_ca, loader_ca_tr, opt_ca, tok_ca_tgt)
    loss_de = train_one_epoch(model_de, loader_de_tr, opt_de, tok_de_tgt)

    bleu_ca = evaluate_bleu(model_ca, loader_ca_val, tok_ca_src, tok_ca_tgt)
    bleu_de = evaluate_bleu(model_de, loader_de_val, tok_de_src, tok_de_tgt)

    print(
        f"Epoch {epoch+1:02d} | "
        f"CA Loss: {loss_ca:.3f} | DE Loss: {loss_de:.3f} | "
        f"CA BLEU: {bleu_ca:.3f} | DE BLEU: {bleu_de:.3f}"
    )

torch.save(model_ca.state_dict(), "models/transformer_ca_en.pth")
torch.save(model_de.state_dict(), "models/transformer_de_en.pth")

print("Models saved to models/")
