from typing import Optional
import logging

import datasets
import torch
import transformers


logger = logging.getLogger(__name__)


def prepare_test_dataloader(
    dataset: torch.utils.data.Dataset,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_seqlen: int = 2048,
    batch_size: int = 1,
) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
    """
    Get a DataLoader from a test dataset. This dataloader should be used when comparing WikiText2 perplexities with other papers, e.g. SparseGPT (arxiv.org/abs/2301.00774).

    Args:
        dataset: The dataset to create a dataloader from.
        tokenizer: The tokenizer to use.
        seqlen: The sequence length of sequences in the dataset.
        batch_size: The batch size.

    Returns:
        A DataLoader.
    """

    logger.info(f"Preparing test dataloader")

    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, ds, tokenizer, seqlen=2048):
            """Tokenize the entire dataset and reshape it into sequences of length seqlen."""

            tokenized_ds = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
            nsamples = tokenized_ds.input_ids.numel() // seqlen

            input_ids = tokenized_ds.input_ids[0, : nsamples * seqlen]
            input_ids = input_ids.reshape(nsamples, seqlen)
            attn_mask = tokenized_ds.attention_mask[0, : nsamples * seqlen]
            attn_mask = attn_mask.reshape(nsamples, seqlen)

            self.input_ids = input_ids
            self.attn_mask = attn_mask

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attn_mask[idx],
            }

        def __len__(self):
            return len(self.input_ids)

    test_ds = TestDataset(dataset, tokenizer, max_seqlen)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    logger.info(f"Preparing test dataloader done")
    return loader


def prepare_dataloader_v1(
    dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_seqlen: int = 2048,
    batch_size: int = 1,
    nsamples: Optional[int] = None,
    varied_seqlen: bool = False,
    seed=42,
) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
    logger.info(f"Preparing dataloader")
    if nsamples is None:
        nsamples = len(dataset)

    if not varied_seqlen and not nsamples:
        logger.warning(
            "varied_seqlen=False, but nsamples is not specified. This will lead to tokenization of the entire dataset, which will be slow."
        )

    data_name = dataset.column_names[0]
    ds = dataset.filter(lambda x: len(x[data_name]) > 0)

    if not varied_seqlen:
        # create a new dataset where each example is a concatenation of multiple examples of total length = max_seqlen.
        data_list = ds[data_name]
        new_data_list = []

        gen = torch.Generator()
        gen.manual_seed(seed)

        indices = list(range(len(data_list)))

        while len(new_data_list) < nsamples and len(indices) > 0:
            start_idx = torch.randint(0, len(indices), (1,), generator=gen).item()
            idx = start_idx
            tokens = []
            while len(tokens) < max_seqlen and idx < len(indices):
                item = data_list[indices[idx]]
                sep = "" if not tokens else "\n\n"
                tokens += tokenizer.tokenize(sep + item)
                idx += 1

            indices = indices[:start_idx] + indices[idx:]  # remove the used indices

            if len(tokens) >= max_seqlen:
                tokens = tokens[:max_seqlen]  # truncate to max_seqlen
                new_data_list.append(tokenizer.convert_tokens_to_string(tokens))

        ds = datasets.Dataset.from_dict({data_name: new_data_list})

    raw = False

    def tokenize(data_batch):
        # tokenize then pad each batch according to the longest sequence in the batch
        batch = tokenizer(
            data_batch[data_name],
            padding="longest",
            max_length=max_seqlen,
            truncation=True,
            return_tensors="pt",
        )
        if raw:
            return batch["input_ids"]
        else:
            batch["labels"] = batch["input_ids"].clone()
            return batch

    # tokenize lazily
    ds.set_transform(tokenize)

    gen = torch.Generator()
    gen.manual_seed(seed)

    indices = torch.randperm(len(ds), generator=gen)[:nsamples]
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler)

    logger.info(f"Preparing dataloader done")
    return loader


def prepare_dataloader_v2(
    dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_seqlen: int = 2048,
    batch_size: int = 1,
    varied_seqlen: bool = False,
    seed=42,
    separator: str = "\n\n",
) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:

    if separator == "eos":
        separator = tokenizer.eos_token
    logging.info(f"Preparing dataloader v2")

    data_name = dataset.column_names[0]
    ds = dataset.filter(lambda x: len(x[data_name]) > 0)

    if not varied_seqlen:
        # create a new dataset where each example is a concatenation of multiple examples of total length = max_seqlen.
        data_list = ds[data_name]
        new_data_list = []

        gen = torch.Generator()
        gen.manual_seed(seed)

        indices = list(range(len(data_list)))

        while len(indices) > 0:
            start_idx = torch.randint(0, len(indices), (1,), generator=gen).item()
            idx = start_idx
            tokens = []
            while len(tokens) < max_seqlen and idx < len(indices):
                item = data_list[indices[idx]]
                input_to_tokenize = tokenizer.bos_token + item + tokenizer.eos_token
                tokens += tokenizer.tokenize(input_to_tokenize)
                idx += 1

            indices = indices[:start_idx] + indices[idx:]  # remove the used indices

            if len(tokens) >= max_seqlen:
                tokens = tokens[:max_seqlen]  # truncate to max_seqlen
                new_data_list.append(tokenizer.convert_tokens_to_string(tokens))

        ds = datasets.Dataset.from_dict({data_name: new_data_list})

    def tokenize(data_batch):
        # tokenize then pad each batch according to the longest sequence in the batch
        batch = tokenizer(
            data_batch[data_name],
            padding="longest",
            max_length=max_seqlen,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # tokenize lazily
    ds.set_transform(tokenize)
    gen = torch.Generator()
    gen.manual_seed(seed)

    sampler = torch.utils.data.SubsetRandomSampler(
        torch.randperm(len(ds), generator=gen)
    )

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler)
    logging.info(f"Preparing dataloader v2 done")
    return loader
