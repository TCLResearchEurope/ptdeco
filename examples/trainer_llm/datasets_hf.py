import codecs
import collections.abc
import logging
from typing import Any

import datasets  # type: ignore
import torch
import transformers  # type: ignore

# TODO: Remove the dataset.config setup when the PR below gets packaged
# TODO: https://github.com/EleutherAI/lm-evaluation-harness/pull/2092

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

logger = logging.getLogger(__name__)


def _remove_all_but_selected_columns(
    ds: datasets.Dataset,
    split_name: str,
    selected_columns: collections.abc.Iterable | collections.abc.Container,
) -> datasets.Dataset:
    cols_to_remove = [c for c in ds[split_name].column_names if c in selected_columns]
    if cols_to_remove:
        cols_to_remove_str = ", ".join(cols_to_remove)
        logger.info(f"Removing columns {cols_to_remove_str}")
        ds = ds.remove_columns(cols_to_remove)
    return ds


def _get_dataset_from_json(fname: str) -> datasets.Dataset:
    ds = datasets.load_dataset("json", data_files=fname)
    split_name = "train"
    data_column = "text"
    ds = _remove_all_but_selected_columns(
        ds, split_name=split_name, selected_columns={data_column}
    )
    res = ds[split_name]
    assert len(res.column_names) == 1
    return res


def _is_json_fname(fname: str) -> bool:
    return (
        fname.endswith(".json")
        or fname.endswith(".json.gz")
        or fname.endswith(".jsonl")
        or fname.endswith(".jsonl.gz")
    )


def _get_dataset_from_hf(dataset_and_split_name: str) -> datasets.Dataset:
    DS_PROPERTIES: dict[str, dict[str, Any]] = {
        "wikitext2": {"path": "wikitext", "config_name": "wikitext-2-raw-v1"},
        "alpaca": {
            "path": "tatsu-lab/alpaca",
            "data_column": "text",
        },
    }
    ds_available = set(DS_PROPERTIES.keys())
    dataset_name, split_name = dataset_and_split_name.split(".")
    if dataset_name not in ds_available:
        raise ValueError(f"Unkown dataset {dataset_name}, available are {ds_available}")

    properties = DS_PROPERTIES[dataset_name]
    ds = datasets.load_dataset(
        properties["path"],
        name=properties.get("config_name"),
        data_files=properties.get("data_files"),
    )

    if dataset_name == "alpaca":
        if split_name == "full":
            split_name = "train"
        else:
            # Alpaca does not have valid/test, so create custom valid/test 10% splits
            ds = ds["train"].train_test_split(test_size=0.2, seed=42)
            temp_ds = ds.pop("test")
            temp_ds = temp_ds.train_test_split(test_size=0.5, seed=42)
            ds["test"] = temp_ds["train"]
            ds["validation"] = temp_ds["test"]

    if "data_column" in properties:
        ds = _remove_all_but_selected_columns(
            ds, split_name=split_name, selected_columns=properties["data_column"]
        )
    res = ds[split_name]
    assert len(res.column_names) == 1
    return res


def get_dataset(dataset_and_split_name: str) -> datasets.Dataset:

    if _is_json_fname(dataset_and_split_name):
        return _get_dataset_from_json(dataset_and_split_name)
    return _get_dataset_from_hf(dataset_and_split_name)


def _normalize_separator(
    separator: str, tokenizer: transformers.PreTrainedTokenizerBase
) -> str:

    allowed_separators = {"\n\n", " ", "", "eos"}

    # Hmm, ... brutal but it might work
    if separator not in allowed_separators:
        raise ValueError(f"{separator=} not in {allowed_separators=}")
    if separator == "eos":
        separator = tokenizer.eos_token
    return separator


def _escape_separator(separator: str) -> str:
    return codecs.escape_encode(separator.encode("utf-8"))[0].decode("utf-8")


def prepare_dataloader_v1(
    *,
    dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizerBase,
    separator: str,
    max_seqlen: int = 2048,
    batch_size: int = 1,
    nsamples: int = 128,
    varied_seqlen: bool = False,
    seed: int = 42,
) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:

    separator = _normalize_separator(separator, tokenizer)

    logger.info(f"v1 dataloader - using sep={_escape_separator(separator)}")

    if not varied_seqlen and not nsamples:
        logger.warning(
            "varied_seqlen=False, but nsamples is not specified. This will lead to "
            "tokenization of the entire dataset, which will be slow."
        )
    assert len(dataset.column_names) == 1
    data_name = dataset.column_names[0]
    logger.info(f"v1 dataloader - using data column={data_name}")
    ds = dataset.filter(lambda x: len(x[data_name]) > 0)

    if not varied_seqlen:
        # Create a new dataset where each example is a concatenation of multiple
        # examples of total length = max_seqlen
        data_list = ds[data_name]
        new_data_list: list[str] = []
        generator = torch.Generator()
        generator.manual_seed(seed)

        indices = list(range(len(data_list)))

        while (nsamples < 0 or len(new_data_list) < nsamples) and len(indices) > 0:
            start_idx = int(
                torch.randint(0, len(indices), (1,), generator=generator).item()
            )
            idx = start_idx
            tokens: list[str] = []
            while len(tokens) < max_seqlen and idx < len(indices):
                item = data_list[indices[idx]]
                sep = "" if not tokens else separator
                tokens += tokenizer.tokenize(sep + item)
                idx += 1
            # logger.info(f"Used {idx-start_idx} examples")
            indices = indices[:start_idx] + indices[idx:]  # remove the used indices

            if len(tokens) >= max_seqlen:
                tokens = tokens[:max_seqlen]  # truncate to max_seqlen
                new_data_list.append(tokenizer.convert_tokens_to_string(tokens))
        msg = f"v1 dataloader - created dataset of size {len(new_data_list)}"
        logger.info(msg)
        ds = datasets.Dataset.from_dict({data_name: new_data_list})

    def tokenize(
        data_batch: dict[str, torch.Tensor]
    ) -> transformers.tokenization_utils_base.BatchEncoding:
        # tokenize then pad each batch according to the longest sequence in the batch
        batch = tokenizer(
            data_batch[data_name],
            padding="longest",
            max_length=max_seqlen,
            truncation=True,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # tokenize lazily
    ds.set_transform(tokenize)
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(ds), generator=generator)[:nsamples].tolist()
    sampler = torch.utils.data.SubsetRandomSampler(indices)

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return loader


def prepare_dataloader_v2(
    *,
    dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_seqlen: int = 2048,
    batch_size: int = 1,
    seed: int = 42,
    separator: str,
) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:

    logger.info(f"v2 dataloader - using sep={_escape_separator(separator)}")

    separator = _normalize_separator(separator, tokenizer)
    eos_string = separator
    eos_tokens = tokenizer(
        eos_string,
        truncation=False,
        padding=False,
        add_special_tokens=False,
    )["input_ids"]
    assert len(dataset.column_names) == 1
    data_name = dataset.column_names[0]
    ds = dataset.filter(lambda x: len(x[data_name]) > 0)

    data_list = ds[data_name]
    new_data_list = []

    idx = 0
    buffer: list[int] = []
    while idx < len(data_list) - 1:
        while len(buffer) <= max_seqlen and idx < len(data_list) - 1:
            encoded = tokenizer(
                data_list[idx],
                truncation=False,
                padding=False,
                add_special_tokens=False,
            )
            iids = encoded["input_ids"]
            buffer += iids + eos_tokens
            idx += 1
        tokens = buffer[:max_seqlen]
        new_data_list.append(tokenizer.decode(tokens))
        buffer = []

    ds = datasets.Dataset.from_dict({data_name: new_data_list})

    def tokenize(
        data_batch: dict[str, torch.Tensor]
    ) -> transformers.tokenization_utils_base.BatchEncoding:
        # tokenize then pad each batch according to the longest sequence in the batch
        batch = tokenizer(
            data_batch[data_name],
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # tokenize lazily
    ds.set_transform(tokenize)

    gen = torch.Generator()
    gen.manual_seed(seed)
    indices = torch.randperm(len(ds), generator=gen).tolist()
    sampler = torch.utils.data.SubsetRandomSampler(indices)

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return loader
