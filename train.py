"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import math
import os
import plotly.graph_objects as go

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from lr_scheduler import NoamScheduler

from model import Transformer, make_src_mask, make_tgt_mask


PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # TODO: Task 3.1
        log_probs = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.full_like(
                log_probs,
                self.smoothing / (self.vocab_size - 2),
            )
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0.0

            pad_positions = target == self.pad_idx
            true_dist[pad_positions] = 0.0

        loss = -(true_dist * log_probs).sum(dim=1)
        non_pad = target != self.pad_idx

        if non_pad.sum() == 0:
            return loss.sum() * 0.0

        return loss[non_pad].mean()


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    tracking: Optional[dict] = None,
) -> dict:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        Dictionary of epoch metrics.

    """
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0
    total_correct = 0
    total_tokens = 0
    total_confidence = 0.0
    progress = tqdm(data_iter, desc=f"{'train' if is_train else 'eval'} {epoch_num}", leave=False)

    for src, tgt in progress:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = make_src_mask(src, pad_idx=PAD_IDX)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx=PAD_IDX)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )

            probs = torch.softmax(logits, dim=-1)
            predictions = probs.argmax(dim=-1)
            non_pad_mask = tgt_output != PAD_IDX
            total_correct += ((predictions == tgt_output) & non_pad_mask).sum().item()
            total_tokens += non_pad_mask.sum().item()

            correct_token_probs = probs.gather(-1, tgt_output.unsqueeze(-1)).squeeze(-1)
            total_confidence += correct_token_probs.masked_select(non_pad_mask).sum().item()

            if is_train:
                optimizer.zero_grad()
                loss.backward()

                if tracking is not None:
                    tracking["global_step"] += 1
                    if (
                        tracking.get("wandb") is not None
                        and tracking["global_step"] <= tracking.get("gradient_log_steps", 0)
                    ):
                        last_layer = model.encoder.layers[-1].self_attn
                        q_grad = last_layer.w_q.weight.grad
                        k_grad = last_layer.w_k.weight.grad
                        if q_grad is not None and k_grad is not None:
                            tracking["wandb"].log(
                                {
                                    "global_step": tracking["global_step"],
                                    "grad_norm_q": q_grad.norm().item(),
                                    "grad_norm_k": k_grad.norm().item(),
                                }
                            )

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        total_loss += loss.item()
        total_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    if total_batches == 0:
        return {"loss": 0.0, "accuracy": 0.0, "prediction_confidence": 0.0}

    return {
        "loss": total_loss / total_batches,
        "accuracy": total_correct / max(total_tokens, 1),
        "prediction_confidence": total_confidence / max(total_tokens, 1),
    }


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    # TODO: Task 3.3 — implement token-by-token greedy decoding
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=PAD_IDX)
            out = model.decode(memory, src_mask, ys, tgt_mask)
            next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)

            if next_token.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    # TODO: Task 3 — loop test set, decode, compute and return BLEU
    def lookup_token(idx: int) -> str:
        if hasattr(tgt_vocab, "lookup_token"):
            return tgt_vocab.lookup_token(idx)
        if hasattr(tgt_vocab, "itos"):
            return tgt_vocab.itos[idx]
        raise AttributeError("tgt_vocab must provide lookup_token() or itos")

    def ids_to_tokens(ids: list[int]) -> list[str]:
        tokens = []
        for idx in ids:
            if idx == EOS_IDX:
                break
            if idx in (PAD_IDX, SOS_IDX):
                continue
            tokens.append(lookup_token(idx))
        return tokens

    references = []
    hypotheses = []

    model.eval()
    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="bleu", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)
            for src_item, tgt_item in zip(src, tgt):
                src_item = src_item.unsqueeze(0)
                tgt_item = tgt_item.unsqueeze(0)

                src_mask = make_src_mask(src_item, pad_idx=PAD_IDX)
                pred = greedy_decode(
                    model,
                    src_item,
                    src_mask,
                    max_len=max_len,
                    start_symbol=SOS_IDX,
                    end_symbol=EOS_IDX,
                    device=device,
                )

                hypotheses.append(ids_to_tokens(pred.squeeze(0).tolist()))
                references.append([ids_to_tokens(tgt_item.squeeze(0).tolist())])

    return _corpus_bleu(references, hypotheses) * 100.0


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # TODO: implement using torch.save({...}, path)
    model_config = {
        "src_vocab_size": model.src_vocab_size,
        "tgt_vocab_size": model.tgt_vocab_size,
        "d_model": model.d_model,
        "N": model.N,
        "num_heads": model.num_heads,
        "d_ff": model.d_ff,
        "dropout": model.dropout,
        "positional_encoding_type": getattr(model, "positional_encoding_type", "sinusoidal"),
        "scale_attention": getattr(model, "scale_attention", True),
    }

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config": model_config,
        "src_vocab_stoi": getattr(model, "src_vocab_stoi", None),
        "tgt_vocab_stoi": getattr(model, "tgt_vocab_stoi", None),
    }
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"]


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # TODO: implement full experiment
    config_dict = {
        "batch_size": int(os.getenv("BATCH_SIZE", "64")),
        "num_epochs": int(os.getenv("NUM_EPOCHS", "20")),
        "d_model": int(os.getenv("D_MODEL", "512")),
        "N": int(os.getenv("NUM_LAYERS", "6")),
        "num_heads": int(os.getenv("NUM_HEADS", "8")),
        "d_ff": int(os.getenv("D_FF", "2048")),
        "dropout": float(os.getenv("DROPOUT", "0.1")),
        "warmup_steps": int(os.getenv("WARMUP_STEPS", "4000")),
        "label_smoothing": float(os.getenv("LABEL_SMOOTHING", "0.1")),
        "scheduler_type": os.getenv("SCHEDULER_TYPE", "noam"),
        "fixed_lr": float(os.getenv("FIXED_LR", "1e-4")),
        "scale_attention": os.getenv("SCALE_ATTENTION", "true").lower() == "true",
        "positional_encoding_type": os.getenv("POSITIONAL_ENCODING_TYPE", "sinusoidal"),
        "gradient_log_steps": int(os.getenv("GRADIENT_LOG_STEPS", "1000")),
        "log_attention_maps": os.getenv("LOG_ATTENTION_MAPS", "true").lower() == "true",
    }

    try:
        import wandb
        from dataset import build_dataloaders
        wandb_run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "da6401-a3"),
            name=os.getenv("WANDB_RUN_NAME"),
            mode=os.getenv("WANDB_MODE"),
            config=config_dict,
        )
        config = wandb.config
    except Exception:
        wandb = None
        wandb_run = None
        from dataset import build_dataloaders
        config = config_dict

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=config["batch_size"]
    )

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        positional_encoding_type=config["positional_encoding_type"],
        scale_attention=config["scale_attention"],
    ).to(device)
    model.src_vocab_stoi = src_vocab.stoi
    model.tgt_vocab_stoi = tgt_vocab.stoi
    model.src_vocab_itos = src_vocab.itos
    model.tgt_vocab_itos = tgt_vocab.itos

    if wandb_run is not None:
        wandb.watch(model, log="all", log_freq=100)

    base_lr = 1.0 if config["scheduler_type"] == "noam" else config["fixed_lr"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = None
    if config["scheduler_type"] == "noam":
        scheduler = NoamScheduler(
            optimizer,
            d_model=config["d_model"],
            warmup_steps=config["warmup_steps"],
        )
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=config["label_smoothing"],
    )

    best_val_bleu = float("-inf")
    best_path = "best_checkpoint.pt"
    tracking = {
        "global_step": 0,
        "gradient_log_steps": config["gradient_log_steps"],
        "wandb": wandb if wandb_run is not None else None,
    }

    for epoch in range(config["num_epochs"]):
        train_metrics = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
            tracking=tracking,
        )
        val_metrics = run_epoch(
            val_loader,
            model,
            loss_fn,
            optimizer=None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device,
            tracking=tracking,
        )
        bleu = evaluate_bleu(model, val_loader, tgt_vocab, device=device)

        if bleu > best_val_bleu:
            best_val_bleu = bleu
            save_checkpoint(model, optimizer, scheduler, epoch, path=best_path)
            print(f"New best BLEU checkpoint saved: val_bleu={bleu:.2f}")

        print(
            f"Epoch {epoch + 1}/{config['num_epochs']} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"bleu={bleu:.2f}"
        )

        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "train_prediction_confidence": train_metrics["prediction_confidence"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_prediction_confidence": val_metrics["prediction_confidence"],
                    "val_bleu": bleu,
                    "best_val_bleu": best_val_bleu,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

    load_checkpoint(best_path, model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)

    if wandb_run is not None:
        wandb.log({"test_bleu": bleu, "best_val_bleu": best_val_bleu})
        if config["log_attention_maps"]:
            _log_attention_maps(wandb, model, val_loader, src_vocab, device)
        wandb.finish()

    print(f"Test BLEU: {bleu:.2f}")


def _extract_ngrams(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    counts = {}
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def _corpus_bleu(
    list_of_references: list[list[list[str]]],
    hypotheses: list[list[str]],
    max_order: int = 4,
) -> float:
    clipped_counts = [0] * max_order
    total_counts = [0] * max_order
    ref_length = 0
    hyp_length = 0

    for references, hypothesis in zip(list_of_references, hypotheses):
        hyp_length += len(hypothesis)
        ref_lengths = [len(reference) for reference in references]
        ref_length += min(ref_lengths, key=lambda length: (abs(length - len(hypothesis)), length))

        for order in range(1, max_order + 1):
            hyp_ngrams = _extract_ngrams(hypothesis, order)
            total_counts[order - 1] += max(len(hypothesis) - order + 1, 0)

            max_ref_counts = {}
            for reference in references:
                ref_ngrams = _extract_ngrams(reference, order)
                for ngram, count in ref_ngrams.items():
                    max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), count)

            for ngram, count in hyp_ngrams.items():
                clipped_counts[order - 1] += min(count, max_ref_counts.get(ngram, 0))

    precisions = []
    for clipped, total in zip(clipped_counts, total_counts):
        if total == 0:
            continue
        if clipped == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped / total)

    if not precisions or min(precisions) == 0.0:
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))

    if hyp_length == 0:
        return 0.0

    if hyp_length > ref_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - (ref_length / hyp_length))

    return brevity_penalty * geo_mean


def _tokens_from_vocab(vocab, ids: list[int], stop_at_eos: bool = True) -> list[str]:
    tokens = []
    for idx in ids:
        if stop_at_eos and idx == EOS_IDX:
            break
        if idx in (PAD_IDX, SOS_IDX):
            continue
        if hasattr(vocab, "lookup_token"):
            tokens.append(vocab.lookup_token(idx))
        else:
            tokens.append(vocab.itos[idx])
    return tokens


def _log_attention_maps(wandb, model: Transformer, dataloader: DataLoader, src_vocab, device: str) -> None:
    model.eval()
    src_batch, _ = next(iter(dataloader))
    src = src_batch[:1].to(device)
    src_mask = make_src_mask(src, pad_idx=PAD_IDX)

    with torch.no_grad():
        model.encode(src, src_mask)
        attn = model.get_last_encoder_self_attention()

    if attn is None:
        return

    attn = attn.squeeze(0).cpu().numpy()
    src_tokens = _tokens_from_vocab(src_vocab, src.squeeze(0).tolist())
    if not src_tokens:
        src_tokens = ["<empty>"]

    for head_idx in range(attn.shape[0]):
        head_map = attn[head_idx]
        token_count = min(len(src_tokens), head_map.shape[0], head_map.shape[1])
        labels = src_tokens[:token_count]
        values = head_map[:token_count, :token_count]
        fig = go.Figure(
            data=go.Heatmap(
                z=values,
                x=labels,
                y=labels,
                colorscale="Viridis",
                colorbar=dict(title="Attention"),
            )
        )
        fig.update_layout(
            title=f"Last encoder self-attention head {head_idx}",
            xaxis_title="Key Tokens",
            yaxis_title="Query Tokens",
            width=850,
            height=700,
        )
        wandb.log(
            {
                f"attention_head_{head_idx}": wandb.Html(
                    fig.to_html(include_plotlyjs="cdn")
                )
            }
        )


if __name__ == "__main__":
    run_training_experiment()
