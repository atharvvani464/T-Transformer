import torch 
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from pathlib import Path
from model import build_transformer

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_create_tokenizer(tokenizer_path, ds, lang):
    tokenizer_path = Path(tokenizer_path)
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
            min_frequency=2
        )
        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang),
            trainer=trainer
        )
        tokenizer.save(str(tokenizer_path))
    return Tokenizer.from_file(str(tokenizer_path))

def get_ds(lang_src="ta", lang_tgt="en"):
    try:
        ds_raw = load_dataset("opus_books", lang_src + "-" + lang_tgt)
        ds = ds_raw["train"]
        if len(ds) == 0:
            raise ValueError("Empty dataset")
        return ds, lang_src, lang_tgt
    except:
        print(f"Dataset for {lang_src}-{lang_tgt} not found. Using de-en instead.")
        ds_raw = load_dataset("opus_books", "de-en")
        return ds_raw["train"], "de", "en"

def encode_sentence(tokenizer, text, seq_len):
    tokens = tokenizer.encode(text).ids
    if len(tokens) < seq_len:
        tokens = tokens + [tokenizer.token_to_id("[PAD]")] * (seq_len - len(tokens))
    else:
        tokens = tokens[:seq_len]
    return torch.tensor(tokens, dtype=torch.long)

# creates src-tgt pairs
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
        src_text = item["translation"][self.src_lang]
        tgt_text = item["translation"][self.tgt_lang]

        src_tokens = encode_sentence(self.tok_src, src_text, self.seq_len)
        tgt_tokens = encode_sentence(self.tok_tgt, tgt_text, self.seq_len)

        return src_tokens, tgt_tokens
    
def create_padding_mask(seq, pad_id):
    # seq: (batch, seq_len)
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)

def create_tgt_mask(tgt, pad_id):
    batch, seq_len = tgt.shape

    pad_mask = create_padding_mask(tgt, pad_id)  # (batch,1,1,seq)
    causal = torch.tril(torch.ones((seq_len, seq_len))).bool()
    causal = causal.unsqueeze(0).unsqueeze(1)     # (1,1,seq,seq)
    return pad_mask & causal

def build_language_transformer(lang_src, lang_tgt, seq_len=64):
    # Load raw dataset
    ds, src_key, tgt_key = get_ds(lang_src, lang_tgt)

    # Load or create tokenizer
    tok_src = get_or_create_tokenizer(f"{lang_src}_tokenizer.json", ds, src_key)
    tok_tgt = get_or_create_tokenizer(f"{lang_tgt}_tokenizer.json", ds, tgt_key)

    # Build vocab sizes
    src_vocab = tok_src.get_vocab_size()
    tgt_vocab = tok_tgt.get_vocab_size()

    # Build transformer model
    model = build_transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        src_seq_len=seq_len,
        tgt_seq_len=seq_len,
        d_model=256,
        N=4,
        h=4,
        dropout=0.1,
        d_ff=512
    )

    # Build dataset + dataloader
    dataset = TranslationDataset(ds, tok_src, tok_tgt, src_key, tgt_key, seq_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    return model, tok_src, tok_tgt, loader

def train_one_epoch(model, data_loader, optimizer, tok_tgt, device):

    PAD = tok_tgt.token_to_id("[PAD]")
    model.train()

    total_loss = 0

    for src, tgt in data_loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = create_padding_mask(src, PAD).to(device)
        tgt_mask = create_tgt_mask(tgt_input, PAD).to(device)

        encoder_output = model.encode(src, src_mask)
        decoder_output = model.decode(encoder_output, src_mask, tgt_input, tgt_mask)
        logits = model.project(decoder_output)

        loss = nn.CrossEntropyLoss(ignore_index=PAD)(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Tamil transformer
model_ta, tok_ta_src, tok_ta_tgt, loader_ta = build_language_transformer("ta", "en")
model_ta.to(device)
opt_ta = torch.optim.Adam(model_ta.parameters(), lr=1e-4)

# Telugu transformer
model_te, tok_te_src, tok_te_tgt, loader_te = build_language_transformer("te", "en")
model_te.to(device)
opt_te = torch.optim.Adam(model_te.parameters(), lr=1e-4)

# Train for a few epochs each
for epoch in range(5):
    loss_ta = train_one_epoch(model_ta, loader_ta, opt_ta, tok_ta_tgt, device)
    loss_te = train_one_epoch(model_te, loader_te, opt_te, tok_te_tgt, device)

    print(f"Epoch {epoch+1} | Tamil Loss: {loss_ta:.3f} | Telugu Loss: {loss_te:.3f}")
