# story_model.py
import torch
import torch.nn as nn


class StoryModel(nn.Module):
    """
    Text-only Transformer language model that:
      - Conditions on: base caption (from Stage A)
      - Uses mode_id: 0=short, 1=description, 2=story
      - Uses length_id: 0=short (10-20), 1=medium (50-100), 2=long (1000+ target)
    Input sequence format (tokens):
        [<start>, base_caption_tokens..., target_tokens..., <end>]
    We add mode + length embeddings to every position.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        start_idx: int,
        end_idx: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        max_seq_len: int = 512,
        num_modes: int = 3,      # short / description / story
        num_len_bins: int = 3,   # short / medium / long
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token and positional embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Mode + length embeddings
        self.mode_embed = nn.Embedding(num_modes, d_model)
        self.len_embed = nn.Embedding(num_len_bins, d_model)

        # Transformer encoder used as decoder-only LM (causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Final projection to vocab
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def _generate_square_subsequent_mask(self, sz: int, device):
        # causal mask: position i can only attend to <= i
        mask = torch.triu(torch.ones(sz, sz, device=device) == 1, diagonal=1)
        mask = mask.float().masked_fill(mask, float("-inf"))
        return mask

    def forward(self, input_ids, mode_ids, length_ids):
        """
        input_ids: (B, T)
        mode_ids:  (B,)  ints in {0,1,2}
        length_ids:(B,)  ints in {0,1,2}
        Returns:
          logits: (B, T, vocab_size)
        """
        device = input_ids.device
        B, T = input_ids.shape

        # embeddings
        positions = torch.arange(0, T, device=device).unsqueeze(0).expand(B, T)
        tok_emb = self.token_embed(input_ids)           # (B, T, d)
        pos_emb = self.pos_embed(positions)            # (B, T, d)

        mode_emb = self.mode_embed(mode_ids)           # (B, d)
        len_emb = self.len_embed(length_ids)           # (B, d)

        mode_emb = mode_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, d)
        len_emb = len_emb.unsqueeze(1).expand(-1, T, -1)    # (B, T, d)

        x = tok_emb + pos_emb + mode_emb + len_emb     # (B, T, d)

        # causal mask
        src_mask = self._generate_square_subsequent_mask(T, device=device)

        out = self.transformer(
            src=x,
            mask=src_mask,
        )  # (B, T, d)

        logits = self.output_proj(out)  # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        base_ids,
        mode_id: int,
        length_id: int,
        max_new_tokens: int = 256,
    ):
        """
        Autoregressive generation.
        base_ids: list[int] - tokenized base caption (without <start>/<end>)
        mode_id: 0=short, 1=description, 2=story
        length_id: 0=short, 1=medium, 2=long
        Returns list[int] of all tokens including base + generated.
        """
        self.eval()
        device = next(self.parameters()).device

        # initial sequence: [<start>] + base_tokens
        seq = [self.start_idx] + base_ids
        input_ids = torch.tensor([seq], dtype=torch.long, device=device)  # (1, T0)
        mode_ids = torch.tensor([mode_id], dtype=torch.long, device=device)
        length_ids = torch.tensor([length_id], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            B, T = input_ids.shape
            if T >= self.max_seq_len:
                break

            positions = torch.arange(0, T, device=device).unsqueeze(0).expand(B, T)
            tok_emb = self.token_embed(input_ids)
            pos_emb = self.pos_embed(positions)

            mode_emb = self.mode_embed(mode_ids).unsqueeze(1).expand(-1, T, -1)
            len_emb = self.len_embed(length_ids).unsqueeze(1).expand(-1, T, -1)

            x = tok_emb + pos_emb + mode_emb + len_emb

            src_mask = self._generate_square_subsequent_mask(T, device=device)
            out = self.transformer(x, mask=src_mask)  # (1, T, d)
            logits = self.output_proj(out[:, -1, :])  # (1, vocab_size)

            next_token = torch.argmax(logits, dim=-1)  # (1,)
            next_id = next_token.item()

            # append
            input_ids = torch.cat(
                [input_ids, next_token.unsqueeze(0)], dim=1
            )  # (1, T+1)

            if next_id == self.end_idx:
                break

        return input_ids[0].tolist()
