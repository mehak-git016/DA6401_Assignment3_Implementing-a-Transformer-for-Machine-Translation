from collections import Counter
from typing import Callable

import torch
from datasets import load_dataset
import spacy


SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


class Vocab:
    def __init__(self, stoi: dict[str, int]) -> None:
        self.stoi = stoi
        self.itos = [None] * len(stoi)
        for token, idx in stoi.items():
            self.itos[idx] = token

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.stoi["<unk>"])

    def lookup_token(self, idx: int) -> str:
        return self.itos[idx]


class Multi30kDataset(torch.utils.data.Dataset):
    _dataset_cache = None

    def __init__(self, split: str = "train", min_freq: int = 1):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        self.min_freq = min_freq

        dataset = self._load_hf_dataset()
        if split not in dataset:
            raise ValueError(f"Unknown split: {split}")

        self.raw_split = dataset[split]
        self._train_split = dataset["train"]

        self.src_tokenizer = self._load_tokenizer("de_core_news_sm", "de")
        self.tgt_tokenizer = self._load_tokenizer("en_core_web_sm", "en")

        self.src_vocab = None
        self.tgt_vocab = None
        self.examples = None

    @classmethod
    def _load_hf_dataset(cls):
        if cls._dataset_cache is None:
            cls._dataset_cache = load_dataset("bentrevett/multi30k")
        return cls._dataset_cache

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

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        src_counter = Counter()
        tgt_counter = Counter()

        for sample in self._train_split:
            src_counter.update(self._tokenize(self.src_tokenizer, sample["de"]))
            tgt_counter.update(self._tokenize(self.tgt_tokenizer, sample["en"]))

        self.src_vocab = self._build_single_vocab(src_counter)
        self.tgt_vocab = self._build_single_vocab(tgt_counter)
        return self.src_vocab, self.tgt_vocab

    def _build_single_vocab(self, counter: Counter) -> Vocab:
        stoi = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        sorted_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))

        for token, freq in sorted_tokens:
            if freq < self.min_freq or token in stoi:
                continue
            stoi[token] = len(stoi)

        return Vocab(stoi)

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary.
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            self.build_vocab()

        examples = []
        for sample in self.raw_split:
            src_tokens = self._tokenize(self.src_tokenizer, sample["de"])
            tgt_tokens = self._tokenize(self.tgt_tokenizer, sample["en"])

            src_ids = [SOS_IDX]
            src_ids.extend(self.src_vocab[token] for token in src_tokens)
            src_ids.append(EOS_IDX)

            tgt_ids = [SOS_IDX]
            tgt_ids.extend(self.tgt_vocab[token] for token in tgt_tokens)
            tgt_ids.append(EOS_IDX)

            examples.append(
                (
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                )
            )

        self.examples = examples
        return examples

    def __len__(self) -> int:
        if self.examples is None:
            self.process_data()
        return len(self.examples)

    def __getitem__(self, idx: int):
        if self.examples is None:
            self.process_data()
        return self.examples[idx]

    def collate_fn(self, batch):
        src_batch, tgt_batch = zip(*batch)
        src_batch = torch.nn.utils.rnn.pad_sequence(
            src_batch,
            batch_first=True,
            padding_value=PAD_IDX,
        )
        tgt_batch = torch.nn.utils.rnn.pad_sequence(
            tgt_batch,
            batch_first=True,
            padding_value=PAD_IDX,
        )
        return src_batch, tgt_batch


def build_dataloaders(batch_size: int = 64):
    train_dataset = Multi30kDataset(split="train")
    src_vocab, tgt_vocab = train_dataset.build_vocab()
    train_dataset.process_data()

    val_dataset = Multi30kDataset(split="validation")
    val_dataset.src_vocab = src_vocab
    val_dataset.tgt_vocab = tgt_vocab
    val_dataset.process_data()

    test_dataset = Multi30kDataset(split="test")
    test_dataset.src_vocab = src_vocab
    test_dataset.tgt_vocab = tgt_vocab
    test_dataset.process_data()

    collate: Callable = train_dataset.collate_fn

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate,
    )

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
