# DA6401 Assignment 3: Transformer for German-to-English Translation

## Links

- GitHub Repository: [DA6401_Assignment3_Implementing-a-Transformer-for-Machine-Translation](https://github.com/mehak-git016/DA6401_Assignment3_Implementing-a-Transformer-for-Machine-Translation)
- W&B Report: [Weights & Biases Report](https://api.wandb.ai/links/ma25m016mehak-indian-institute-of-technology-madras/m8g3s0lk)

## Assignment Overview

This project implements the base Transformer architecture from *Attention Is All You Need* in PyTorch for German-to-English neural machine translation on the Multi30k dataset. The implementation covers the three core components:

1. Scaled dot-product attention, multi-head attention, and masking
2. Transformer encoder-decoder stacks with positional encoding and feed-forward layers
3. Training pipeline with label smoothing, Noam scheduler, greedy decoding, checkpointing and BLEU evaluation

The model is trained on the Hugging Face `bentrevett/multi30k` dataset using `spacy` tokenization, and the submission path supports single-sentence inference through `Transformer().infer(...)` as required by the autograder.

## Base Paper

- Paper: [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

## Dataset

- Dataset: [bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k)
- Train split: `29,000`
- Validation split: `1,014`
- Test split: `1,000`

## Default Training Configuration

The current default training configuration is:

- `batch_size = 64`
- `num_epochs = 20`
- `d_model = 512`
- `N = 6`
- `num_heads = 8`
- `d_ff = 2048`
- `dropout = 0.1`
- `warmup_steps = 4000`
- `label_smoothing = 0.1`
- Optimizer: `Adam(beta1=0.9, beta2=0.98, eps=1e-9)`

These defaults are defined in [train.py].
