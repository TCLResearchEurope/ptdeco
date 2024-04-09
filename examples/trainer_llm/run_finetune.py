import json
import logging
import pathlib
import time
from typing import Any

import torch
import transformers  # type: ignore

import builder
import configurator
import datasets_hf
import metrics
import utils

PPL_EVAL_VARIED_SEQLEN = False
LOADER_SEED = 42


logger = logging.getLogger(__name__)


def make_dataloaders(
    config: configurator.FinetuneConfig,
    tokenizer: transformers.PreTrainedTokenizer,
) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:

    perplexity_ds = datasets_hf.get_dataset(config.perplexity_data_name)

    logger.info(
        f"Created perplexity dataset {config.perplexity_data_name}, "
        f"{len(perplexity_ds)} examples"
    )

    perplexity_dl = datasets_hf.prepare_slicegpt_dataloader(
        dataset=perplexity_ds,
        tokenizer=tokenizer,
        max_seqlen=config.perplexity_data_max_length,
        batch_size=config.perplexity_data_batch_size,
        separator=config.perplexity_data_separator,
        nsamples=1000,
        seed=LOADER_SEED,
    )

    return perplexity_dl


def main(config_raw: dict[str, Any], output_path: pathlib.Path) -> None:
    start = time.perf_counter()
    transformers.utils.logging.disable_progress_bar()
    config = configurator.FinetuneConfig(**config_raw)
    dtype = utils.conv_str_to_dtype(config.decomposed_model_dtype)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    egc = config.decomposed_model_enable_gradient_checkpointing
    model, tokenizer = builder.make_model_and_tokenizer(
        model_name=config.decomposed_model_name,
        model_revision=config.decomposed_model_revision,
        decompose_config_path=config.decompose_config,
        state_dict_path=config.decompose_state_dict,
        enable_gradient_checkpointing=egc,
        device=device,
        dtype=dtype,
        log_linears=True,
    )

    perplexity_dl = make_dataloaders(config, tokenizer)
    with torch.no_grad():
        perplexity_initial = metrics.calc_perplexity(
            model, perplexity_dl, device, model.config.pad_token_id
        )
    logger.info(f"{perplexity_initial=}")

    if config.lm_eval_initial and config.lm_eval_tasks:
        start = time.perf_counter()

        lm_eval_results, lm_eval_results_str = metrics.calc_lm_eval_metrics(
            model=model,
            tokenizer=tokenizer,
            device=device,
            tasks=config.lm_eval_tasks,
        )
        logger.info("\n" + lm_eval_results_str)
        lm_eval_path = output_path / "lm_eval_initial.json"
        lm_eval_results["config"]["device"] = str(lm_eval_results["config"]["device"])
        with open(lm_eval_path, "wt") as f:
            json.dump(lm_eval_results, f)
        logger.info(f"Initial lm_eval results saved to {lm_eval_path}")
        time_lm_eval = time.perf_counter() - start
        logger.info(f"Initial lm_eval took {time_lm_eval:.2f} s")
