# train_story.py
import os
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from story_model import StoryModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

VOCAB_FILE = "tokenizer/vocab.json"
STORY_DATA_FILE = "story_data.json"
CHECKPOINT_DIR = "checkpoints_story"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 8        # stories can be long, keep batch small
EPOCHS = 10
LR = 3e-5
MAX_SEQ_LEN = 512     # adjust if you want longer stories
VAL_SPLIT = 0.1
RANDOM_SEED = 42


class StoryDataset(Dataset):
    """
    Builds training samples from story_data.json.

    Each item in story_data.json:
      {
        "base_caption": "...",
        "short_text": "...",
        "description_text": "...",
        "story_text": "..."
      }

    For each such entry, we create 3 samples:
      mode=0, len_bin=0, target=short_text
      mode=1, len_bin=1, target=description_text
      mode=2, len_bin=2, target=story_text

    Sequence format (token IDs):
      [<start>, base_caption_ids..., target_ids..., <end>, <pad>...]
      length = MAX_SEQ_LEN (padded/truncated)
    """

    def __init__(self, vocab_file, story_data_file, max_seq_len=512):
        super().__init__()
        self.max_seq_len = max_seq_len

        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.itos = vocab["itos"]
        self.stoi = vocab["stoi"]

        self.pad_idx = self.stoi["<pad>"]
        self.start_idx = self.stoi["<start>"]
        self.end_idx = self.stoi["<end>"]
        self.unk_idx = self.stoi["<unk>"]

        with open(story_data_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.samples = []
        for item in raw:
            base = item["base_caption"]
            short = item["short_text"]
            desc = item["description_text"]
            story = item["story_text"]

            self.samples.append((base, short, 0, 0))  # mode=0 short, len_bin=0
            self.samples.append((base, desc, 1, 1))   # mode=1 desc, len_bin=1
            self.samples.append((base, story, 2, 2))  # mode=2 story, len_bin=2

        print(f"Loaded {len(raw)} base items -> {len(self.samples)} (mode,length) samples")

    def __len__(self):
        return len(self.samples)

    def encode_text(self, text):
        tokens = text.strip().split()
        ids = [self.stoi.get(t, self.unk_idx) for t in tokens]
        return ids

    def __getitem__(self, idx):
        base_caption, target_text, mode_id, len_bin = self.samples[idx]

        base_ids = self.encode_text(base_caption)
        target_ids = self.encode_text(target_text)

        # [<start>] + base + target + [<end>]
        seq = [self.start_idx] + base_ids + target_ids + [self.end_idx]

        # truncate / pad to max_seq_len
        if len(seq) > self.max_seq_len:
            seq = seq[: self.max_seq_len]
            if seq[-1] != self.end_idx:
                seq[-1] = self.end_idx
        else:
            seq = seq + [self.pad_idx] * (self.max_seq_len - len(seq))

        input_ids = torch.tensor(seq, dtype=torch.long)
        mode_id = torch.tensor(mode_id, dtype=torch.long)
        len_bin = torch.tensor(len_bin, dtype=torch.long)

        return input_ids, mode_id, len_bin


def compute_token_accuracy(logits, target, pad_idx):
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        mask = target != pad_idx
        correct = (preds == target) & mask
        total = mask.sum().item()
        if total == 0:
            return 0.0
        return correct.sum().item() / total


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # load vocab for indices
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    itos = vocab["itos"]
    stoi = vocab["stoi"]

    vocab_size = len(itos)
    pad_idx = stoi["<pad>"]
    start_idx = stoi["<start>"]
    end_idx = stoi["<end>"]
    print("Vocab size:", vocab_size)
    print("PAD:", pad_idx, "START:", start_idx, "END:", end_idx)

    # dataset
    full_dataset = StoryDataset(
        vocab_file=VOCAB_FILE,
        story_data_file=STORY_DATA_FILE,
        max_seq_len=MAX_SEQ_LEN,
    )

    n_total = len(full_dataset)
    n_val = max(1, int(VAL_SPLIT * n_total))
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
    print(f"Total samples: {n_total} | Train: {n_train} | Val: {n_val}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    # model
    model = StoryModel(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        start_idx=start_idx,
        end_idx=end_idx,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        max_seq_len=MAX_SEQ_LEN,
        num_modes=3,
        num_len_bins=3,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    try:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.05)
        print("Using CE with label_smoothing=0.05")
    except TypeError:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        print("Using plain CE (no label_smoothing).")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    best_val_loss = float("inf")
    best_path = None

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        # TRAIN
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        n_batches = 0

        for i, (input_ids, mode_ids, len_bins) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            mode_ids = mode_ids.to(device)
            len_bins = len_bins.to(device)

            # LM: predict next token
            inp = input_ids[:, :-1]     # (B, T-1)
            target = input_ids[:, 1:]   # (B, T-1)

            logits = model(inp, mode_ids, len_bins)  # (B, T-1, V)
            Bsz, Tm1, V = logits.shape

            # ✅ use reshape instead of view (handles non-contiguous)
            logits_flat = logits.reshape(Bsz * Tm1, V)
            target_flat = target.reshape(-1)

            loss = criterion(
                logits_flat,
                target_flat,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            acc = compute_token_accuracy(logits, target, pad_idx)
            train_loss += loss.item()
            train_acc += acc
            n_batches += 1

            if (i + 1) % 50 == 0:
                print(
                    f"Epoch {epoch+1} | Iter {i+1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f} | Avg Loss: {train_loss/n_batches:.4f} "
                    f"| Avg Token Acc: {100*train_acc/n_batches:.2f}%"
                )

        train_loss /= max(1, n_batches)
        train_acc = 100 * train_acc / max(1, n_batches)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Train Token Acc: {train_acc:.2f}%")

        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for input_ids, mode_ids, len_bins in val_loader:
                input_ids = input_ids.to(device)
                mode_ids = mode_ids.to(device)
                len_bins = len_bins.to(device)

                inp = input_ids[:, :-1]
                target = input_ids[:, 1:]

                logits = model(inp, mode_ids, len_bins)
                Bsz, Tm1, V = logits.shape

                logits_flat = logits.reshape(Bsz * Tm1, V)
                target_flat = target.reshape(-1)

                loss = criterion(
                    logits_flat,
                    target_flat,
                )

                acc = compute_token_accuracy(logits, target, pad_idx)
                val_loss += loss.item()
                val_acc += acc
                n_val_batches += 1

        val_loss /= max(1, n_val_batches)
        val_acc = 100 * val_acc / max(1, n_val_batches)
        print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f} | Val Token Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        # save epoch checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"story_epoch_{epoch+1}.pt")
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1,
                "vocab_file": VOCAB_FILE,
            },
            ckpt_path,
        )
        print("Saved checkpoint:", ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "story_best.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "vocab_file": VOCAB_FILE,
                },
                best_path,
            )
            print(f"*** New best StoryModel saved: {best_path} (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()
