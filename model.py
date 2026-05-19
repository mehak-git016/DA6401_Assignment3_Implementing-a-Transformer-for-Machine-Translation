"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
from typing import Optional, Tuple

import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_UNK_IDX = 0
DEFAULT_PAD_IDX = 1
DEFAULT_SOS_IDX = 2
DEFAULT_EOS_IDX = 3
DEFAULT_SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]
DEFAULT_CHECKPOINT_NAME = "best_checkpoint.pt"
DEFAULT_CHECKPOINT_DRIVE_ID = os.getenv(
    "TRANSFORMER_CHECKPOINT_DRIVE_ID",
    "1xwlyFgMNo6QTwTlXAuJb8G1lFz_2hHZH",
)


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if scale:
        scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    attn_w = torch.softmax(scores, dim=-1)
    if mask is not None:
        attn_w = attn_w.masked_fill(mask, 0.0)
        normalizer = attn_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        attn_w = attn_w / normalizer
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    batch_size, tgt_len = tgt.shape
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(1)
    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        scale_attention: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale_attention = scale_attention
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]

        """
        batch_size = query.size(0)

        def shape_projection(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
            x = linear(x)
            x = x.view(batch_size, -1, self.num_heads, self.d_k)
            return x.transpose(1, 2)

        q = shape_projection(query, self.w_q)
        k = shape_projection(key, self.w_k)
        v = shape_projection(value, self.w_v)

        attn_output, attn_weights = scaled_dot_product_attention(
            q,
            k,
            v,
            mask,
            scale=self.scale_attention,
        )
        attn_output = self.dropout(attn_output)
        self.last_attn_weights = attn_weights.detach()

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        return self.w_o(attn_output)


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]  

        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: Task 2.3 — define:
        #   self.linear1 = nn.Linear(d_model, d_ff)
        #   self.linear2 = nn.Linear(d_ff, d_model)
        #   self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        scale_attention: bool = True,
    ) -> None:
        super().__init__()
        # TODO:instantiate:
        self.self_attn = MultiHeadAttention(
            d_model,
            num_heads,
            dropout,
            scale_attention=scale_attention,
        )
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        scale_attention: bool = True,
    ) -> None:
        super().__init__()
        # TODO: instantiate:
        self.self_attn = MultiHeadAttention(
            d_model,
            num_heads,
            dropout,
            scale_attention=scale_attention,
        )
        self.cross_attn = MultiHeadAttention(
            d_model,
            num_heads,
            dropout,
            scale_attention=scale_attention,
        )
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        cross_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout2(cross_out))

        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(N))
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(N))
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
        checkpoint_path: Optional[str] = None,
        checkpoint_drive_id: Optional[str] = None,
        max_len: int = 50,
        positional_encoding_type: str = "sinusoidal",
        scale_attention: bool = True,
    ) -> None:
        super().__init__()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_checkpoint_path = os.path.join(base_dir, DEFAULT_CHECKPOINT_NAME)
        self.checkpoint_path = checkpoint_path or default_checkpoint_path
        self.max_len = max_len

        auto_load_checkpoint = checkpoint_path is not None or (
            src_vocab_size is None and tgt_vocab_size is None
        )
        checkpoint = None

        if auto_load_checkpoint:
            checkpoint_drive_id = checkpoint_drive_id or DEFAULT_CHECKPOINT_DRIVE_ID

            if checkpoint_drive_id and not os.path.exists(self.checkpoint_path):
                try:
                    import gdown
                except ImportError as exc:
                    raise ImportError(
                        "gdown is required to download the Transformer checkpoint."
                    ) from exc

                gdown.download(
                    id=checkpoint_drive_id,
                    output=self.checkpoint_path,
                    quiet=False,
                )

            if os.path.exists(self.checkpoint_path):
                checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "model_config" in checkpoint:
            model_config = checkpoint["model_config"]
            if src_vocab_size is None:
                src_vocab_size = model_config.get("src_vocab_size")
            if tgt_vocab_size is None:
                tgt_vocab_size = model_config.get("tgt_vocab_size")
            d_model = model_config.get("d_model", d_model)
            N = model_config.get("N", N)
            num_heads = model_config.get("num_heads", num_heads)
            d_ff = model_config.get("d_ff", d_ff)
            dropout = model_config.get("dropout", dropout)
            positional_encoding_type = model_config.get(
                "positional_encoding_type",
                positional_encoding_type,
            )
            scale_attention = model_config.get("scale_attention", scale_attention)

        if src_vocab_size is None:
            src_vocab_size = len(checkpoint.get("src_vocab_stoi", {})) if isinstance(checkpoint, dict) else None
        if tgt_vocab_size is None:
            tgt_vocab_size = len(checkpoint.get("tgt_vocab_stoi", {})) if isinstance(checkpoint, dict) else None

        self.src_vocab_size = src_vocab_size or len(DEFAULT_SPECIAL_TOKENS)
        self.tgt_vocab_size = tgt_vocab_size or len(DEFAULT_SPECIAL_TOKENS)
        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.positional_encoding_type = positional_encoding_type
        self.scale_attention = scale_attention

        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, d_model)
        if positional_encoding_type == "learned":
            self.positional_encoding = LearnedPositionalEncoding(d_model, dropout)
        else:
            self.positional_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout,
            scale_attention=scale_attention,
        )
        decoder_layer = DecoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout,
            scale_attention=scale_attention,
        )

        self.encoder = Encoder(encoder_layer, N)
        self.decoder = Decoder(decoder_layer, N)
        self.generator = nn.Linear(d_model, self.tgt_vocab_size)

        self.src_vocab_stoi = {
            token: idx for idx, token in enumerate(DEFAULT_SPECIAL_TOKENS)
        }
        self.tgt_vocab_stoi = {
            token: idx for idx, token in enumerate(DEFAULT_SPECIAL_TOKENS)
        }
        self.src_vocab_itos = list(DEFAULT_SPECIAL_TOKENS)
        self.tgt_vocab_itos = list(DEFAULT_SPECIAL_TOKENS)

        self.pad_idx = DEFAULT_PAD_IDX
        self.sos_idx = DEFAULT_SOS_IDX
        self.eos_idx = DEFAULT_EOS_IDX
        self.unk_idx = DEFAULT_UNK_IDX

        self.src_tokenizer = self._load_tokenizer("de_core_news_sm", "de")
        self.tgt_tokenizer = self._load_tokenizer("en_core_web_sm", "en")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self._restore_vocab_state(checkpoint)
            state_dict = checkpoint.get("model_state_dict")
            if state_dict is not None:
                self.load_state_dict(state_dict)
        elif isinstance(checkpoint, dict) and checkpoint:
            self.load_state_dict(checkpoint)

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        src_embed = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embed = self.positional_encoding(src_embed)
        return self.encoder(src_embed, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        tgt_embed = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embed = self.positional_encoding(tgt_embed)
        decoder_output = self.decoder(tgt_embed, memory, src_mask, tgt_mask)
        return self.generator(decoder_output)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.

        Args:
            src_sentence: The raw German text.

        Returns:
            The fully translated English string, detokenized and clean.
        """
        self.eval()

        src_tokens = self._tokenize(self.src_tokenizer, src_sentence)
        src_ids = [self.sos_idx]
        src_ids.extend(self.src_vocab_stoi.get(token, self.unk_idx) for token in src_tokens)
        src_ids.append(self.eos_idx)

        device = next(self.parameters()).device
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = make_src_mask(src, pad_idx=self.pad_idx)

        with torch.no_grad():
            memory = self.encode(src, src_mask)
            generated = torch.full(
                (1, 1),
                self.sos_idx,
                dtype=torch.long,
                device=device,
            )

            max_decode_len = min(self.max_len, max(20, 3 * len(src_tokens) + 10))

            for _ in range(max_decode_len - 1):
                tgt_mask = make_tgt_mask(generated, pad_idx=self.pad_idx)
                logits = self.decode(memory, src_mask, generated, tgt_mask)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == self.eos_idx:
                    break

        return self._detokenize(generated.squeeze(0).tolist())

    @staticmethod
    def _load_tokenizer(model_name: str, lang_code: str):
        try:
            nlp = spacy.load(model_name)
        except OSError:
            nlp = spacy.blank(lang_code)
        return nlp.tokenizer

    @staticmethod
    def _tokenize(tokenizer, text: str) -> list[str]:
        return [token.text.lower() for token in tokenizer(text.strip()) if token.text.strip()]

    def _restore_vocab_state(self, checkpoint: dict) -> None:
        src_stoi = checkpoint.get("src_vocab_stoi")
        tgt_stoi = checkpoint.get("tgt_vocab_stoi")

        if src_stoi:
            self.src_vocab_stoi = src_stoi
            self.src_vocab_itos = self._build_inverse_vocab(src_stoi)

        if tgt_stoi:
            self.tgt_vocab_stoi = tgt_stoi
            self.tgt_vocab_itos = self._build_inverse_vocab(tgt_stoi)

        self.unk_idx = self.src_vocab_stoi.get("<unk>", DEFAULT_UNK_IDX)
        self.pad_idx = self.src_vocab_stoi.get("<pad>", DEFAULT_PAD_IDX)
        self.sos_idx = self.src_vocab_stoi.get("<sos>", DEFAULT_SOS_IDX)
        self.eos_idx = self.src_vocab_stoi.get("<eos>", DEFAULT_EOS_IDX)

    @staticmethod
    def _build_inverse_vocab(stoi: dict) -> list[str]:
        itos = [None] * len(stoi)
        for token, idx in stoi.items():
            if idx >= len(itos):
                itos.extend([None] * (idx - len(itos) + 1))
            itos[idx] = token
        return [token if token is not None else "<unk>" for token in itos]

    def _detokenize(self, token_ids: list[int]) -> str:
        words = []
        for idx in token_ids:
            if idx == self.eos_idx:
                break
            if idx in (self.pad_idx, self.sos_idx):
                continue
            if 0 <= idx < len(self.tgt_vocab_itos):
                words.append(self.tgt_vocab_itos[idx])
            else:
                words.append("<unk>")

        text = " ".join(words)
        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")
        text = text.replace(" ;", ";")
        text = text.replace(" :", ":")
        text = text.replace(" n't", "n't")
        text = text.replace(" 's", "'s")
        text = text.replace(" 're", "'re")
        text = text.replace(" 've", "'ve")
        text = text.replace(" 'm", "'m")
        return text.strip()

    def get_last_encoder_self_attention(self) -> Optional[torch.Tensor]:
        if not self.encoder.layers:
            return None
        return self.encoder.layers[-1].self_attn.last_attn_weights
