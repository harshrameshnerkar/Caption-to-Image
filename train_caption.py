# train_caption.py
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import Flickr30kFeaturesDataset
from model import CaptionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Paths
CAPTIONS_FILE = "captions.json"
VOCAB_FILE = "tokenizer/vocab.json"
FEATURES_DIR = "image_features"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 25           # total epochs target
LR = 2e-5             # smaller LR for fine-tuning
D_MODEL = 512
MAX_LEN = 30
R_MAX = 36
VAL_SPLIT = 0.05
RESUME_CKPT = "checkpoints/caption_best.pt"  # resume from best style-aware v2 checkpoint


def compute_token_accuracy(logits, target, pad_idx):
    """
    logits: (B, T, V), target: (B, T)
    Ignores pad positions.
    """
    with torch.no_grad():
        preds = logits.argmax(dim=-1)         # (B, T)
        mask = target != pad_idx              # (B, T)
        correct = (preds == target) & mask
        total = mask.sum().item()
        if total == 0:
            return 0.0
        return correct.sum().item() / total


def main():
    # ----- load vocab -----
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    itos = vocab["itos"]
    stoi = vocab["stoi"]

    vocab_size = len(itos)
    pad_idx = stoi["<pad>"]
    start_idx = stoi["<start>"]
    end_idx = stoi["<end>"]
    print("Vocab size:", vocab_size)
    print("PAD idx:", pad_idx, "START idx:", start_idx, "END idx:", end_idx)
    print(f"Training with LR = {LR}")

    # ----- full dataset -----
    full_dataset = Flickr30kFeaturesDataset(
        captions_file=CAPTIONS_FILE,
        vocab_file=VOCAB_FILE,
        features_dir=FEATURES_DIR,
        max_len=MAX_LEN,
        r_max=R_MAX,
    )

    # train/val split
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

    # ----- model -----
    model = CaptionModel(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        start_idx=start_idx,
        end_idx=end_idx,
        d_model=D_MODEL,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        max_len=MAX_LEN,
        r_max=R_MAX,
        dropout=0.1,
        num_styles=3,
    ).to(device)

    start_epoch = 0

    # ----- resume from checkpoint if available -----
    if RESUME_CKPT is not None and os.path.exists(RESUME_CKPT):
        ckpt = torch.load(RESUME_CKPT, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        prev_epoch = ckpt.get("epoch", 0)
        start_epoch = prev_epoch
        print(f"Resumed weights from {RESUME_CKPT} (previous epoch = {prev_epoch})")
    else:
        print("Training style-aware model from scratch (no RESUME_CKPT found).")

    # ----- optimizer & loss -----
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    try:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
        print("Using CrossEntropyLoss with label_smoothing=0.1")
    except TypeError:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        print("label_smoothing not supported in this torch version, using plain CE.")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
    )

    best_val_loss = float("inf")
    best_ckpt_path = None

    # ----- training loop -----
    for epoch in range(start_epoch, EPOCHS):
        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_train_batches = 0

        for i, (g, l, caps, style_ids) in enumerate(train_loader):
            g = g.to(device).float()          # (B, 2048)
            l = l.to(device).float()          # (B, R, 2048)
            caps = caps.to(device).long()     # (B, T)
            style_ids = style_ids.to(device).long()  # (B,)

            # teacher forcing: input = caps[:, :-1], target = caps[:, 1:]
            inp = caps[:, :-1]
            target = caps[:, 1:].contiguous()

            logits = model(
                global_feat=g,
                region_feats=l,
                captions=inp,
                style_ids=style_ids,
            )  # (B, T-1, V)
            Bsz, Tm1, V = logits.shape

            loss = criterion(
                logits.view(Bsz * Tm1, V),
                target.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            acc = compute_token_accuracy(logits, target, pad_idx)

            running_loss += loss.item()
            running_acc += acc
            n_train_batches += 1

            if (i + 1) % 100 == 0:
                avg_loss = running_loss / n_train_batches
                avg_acc = running_acc / n_train_batches * 100.0
                print(
                    f"Epoch {epoch+1} | Iter {i+1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} "
                    f"| Avg Token Acc: {avg_acc:.2f}%"
                )

        train_loss = running_loss / max(1, n_train_batches)
        train_acc = running_acc / max(1, n_train_batches) * 100.0
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Train Token Acc: {train_acc:.2f}%")

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for g, l, caps, style_ids in val_loader:
                g = g.to(device).float()
                l = l.to(device).float()
                caps = caps.to(device).long()
                style_ids = style_ids.to(device).long()

                inp = caps[:, :-1]
                target = caps[:, 1:].contiguous()

                logits = model(
                    global_feat=g,
                    region_feats=l,
                    captions=inp,
                    style_ids=style_ids,
                )
                Bsz, Tm1, V = logits.shape

                loss = criterion(
                    logits.view(Bsz * Tm1, V),
                    target.view(-1)
                )

                acc = compute_token_accuracy(logits, target, pad_idx)
                val_loss += loss.item()
                val_acc += acc
                n_val_batches += 1

        val_loss = val_loss / max(1, n_val_batches)
        val_acc = val_acc / max(1, n_val_batches) * 100.0
        print(f"[Epoch {epoch+1}] Val  Loss: {val_loss:.4f} | Val Token Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        # ----- save checkpoint -----
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"caption_epoch_{epoch+1}.pt")
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
            best_ckpt_path = os.path.join(CHECKPOINT_DIR, "caption_best.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "vocab_file": VOCAB_FILE,
                },
                best_ckpt_path,
            )
            print(f"*** New best model saved: {best_ckpt_path} (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()
