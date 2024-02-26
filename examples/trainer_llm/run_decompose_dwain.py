from typing import Any
import json
import logging
import pathlib

import datasets
import torch
import transformers
import ptdeco

import configurator
import datasets_hf
import metrics


PPL_EVAL_VARIED_SEQLEN = False
LOADER_SEED = 42


logger = logging.getLogger(__name__)


def setup_logging():
    # TENSORFLOW style format
    fmt = "%(asctime)s.%(msecs)03d: %(levelname).1s %(name)s.py:%(lineno)d] %(message)s"

    # SIMPLER style format
    # fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.WARNING,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Herer you put modules that you want more verbose logging

    for module_name in [__name__, "datasets_hf", "metrics", "ptdeco"]:
        logging.getLogger(module_name).setLevel(logging.INFO)


def get_params(m: torch.nn.Module, only_trainable: bool = False) -> int:
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def make_inifinte_iterator(dl):
    while True:
        for x in dl:
            yield x


def make_padding_tokenizer(
    model: torch.nn.Module, tokenizer: transformers.PreTrainedTokenizer, model_name: str
) -> transformers.PreTrainedTokenizer:
    if model_name in (
        "meta-llama/Llama-2-7b-hf",
        "microsoft/phi-2",
        "Qwen/Qwen1.5-1.8B",
        "upstage/SOLAR-10.7B-v1.0",
        "mistralai/Mistral-7B-Instruct-v0.2",
    ):
        tokenizer.pad_token = (
            tokenizer.eos_token
        )  # Phi-2 and LLama2 models don't have a pad token by default
        model.config.pad_token_id = tokenizer.pad_token_id  # llama, phi
        logger.warning(f"Setting pad_token to eos_token")

    if model_name == "Qwen/Qwen-1_8B":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen-1.8B", trust_remote_code=True, pad_token="<|endoftext|>"
        )
    return tokenizer


def conv_str_to_dtype(s: str) -> torch.dtype:
    if s == "torch.float32":
        return torch.float32
    elif s == "torch.bfloat16":
        return torch.bfloat16
    elif s == "torch.float16":
        return torch.float16
    raise ValueError(f"Unknown dtype {s}")


def main(config: dict[str, Any], output_path: pathlib.Path) -> None:
    transformers.utils.logging.disable_progress_bar()
    config_parsed = configurator.DecomposeDWAINConfig(**config)

    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     "facebook/opt-125m", trust_remote_code=True
    # )
    # model = transformers.AutoModel.from_pretrained(
    #     "facebook/opt-125m", trust_remote_code=True
    # )
    # print(type(model))

    # message = ["The largest lake on earth is "]
    # inputs = tokenizer(message, return_tensors="pt", return_token_type_ids=False)
    # response = model.generate(
    #     **inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95
    # )
    # print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     "facebook/opt-125m", trust_remote_code=True
    # )
    # model = transformers.AutoModel.from_pretrained(
    #     "facebook/opt-125m", trust_remote_code=True
    # )

    # GENERATION:
    #
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")

    # model = transformers.OPTForCausalLM.from_pretrained(
    #     "facebook/opt-125m", torch_dtype=torch.float32,
    # )
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     "facebook/opt-125m", trust_remote_code=True
    # )
    # generator = transformers.pipeline(
    #     "text-generation", model=model, tokenizer=tokenizer, do_sample=True
    # )

    ds = datasets.load_dataset("wikitext", name="wikitext-2-raw-v1")
    ds_train, ds_valid, ds_test = ds["train"], ds["validation"], ds["test"]
    logger.info(f"{len(ds_train)=}, {len(ds_valid)=}, {len(ds_test)=}")
    # sys.exit()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # model = transformers.OPTForCausalLM.from_pretrained(
    #     "facebook/opt-125m",
    #     torch_dtype=torch.float32,
    # )
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     "facebook/opt-125m", trust_remote_code=True
    # )
    model_name = config_parsed.decomposed_model_name
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )

    dtype = conv_str_to_dtype(config_parsed.decomposed_model_dtype)
    logger.info(f"Using {dtype=}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    )
    tokenizer = make_padding_tokenizer(
        model_name=model_name, model=model, tokenizer=tokenizer
    )

    model.to(device)
    model.to(dtype)
    model.eval()

    test_datasetloader = False
    logger.info(f"{test_datasetloader=}")
    if test_datasetloader:
        ppl_eval_loader = datasets_hf.prepare_test_dataloader(
            dataset=ds_test,
            tokenizer=tokenizer,
            max_seqlen=config_parsed.metric_max_length,
            batch_size=config_parsed.metric_batch_size,
        )
    else:
        ppl_eval_loader = datasets_hf.prepare_dataloader(
            dataset=ds_test,
            tokenizer=tokenizer,
            max_seqlen=config_parsed.metric_max_length,
            batch_size=config_parsed.metric_batch_size,
            nsamples=len(ds_test),
            varied_seqlen=PPL_EVAL_VARIED_SEQLEN,
            seed=LOADER_SEED,
        )
    train_dataloader = datasets_hf.prepare_dataloader(
        dataset=ds_train,
        tokenizer=tokenizer,
        max_seqlen=config_parsed.data_max_length,
        batch_size=config_parsed.data_batch_size,
        nsamples=len(ds_test),
        varied_seqlen=False,
        seed=LOADER_SEED,
    )
    valid_dataloader = datasets_hf.prepare_dataloader(
        dataset=ds_valid,
        tokenizer=tokenizer,
        max_seqlen=config_parsed.metric_max_length,
        batch_size=config_parsed.metric_batch_size,
        nsamples=len(ds_valid),
        varied_seqlen=PPL_EVAL_VARIED_SEQLEN,
        seed=LOADER_SEED,
    )
    with torch.no_grad():
        perplexity_orig = metrics.calc_perplexity(
            model, ppl_eval_loader, device, model.config.pad_token_id
        )
    params_orig = get_params(model) / 1.0e6
    logger.info(f"{perplexity_orig=} {params_orig=}")

    class WrapperModule(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.config = model.config

        def forward(self, x):
            return self.model(**x).logits

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return self.model.prepare_inputs_for_generation(input_ids, **kwargs)

    model_wrapped = WrapperModule(model)

    for n, m in model_wrapped.named_modules():
        if isinstance(m, torch.nn.Linear):
            logger.info(f"{n}, {m.weight.shape}")
    num_layers = config_parsed.num_last_decomposed_layers_to_finetune
    decompose_config = ptdeco.dwain.decompose_in_place(
        module=model_wrapped,
        device=device,
        dtype=dtype,
        blacklisted_module_names=config_parsed.blacklisted_module_names,
        data_iterator=make_inifinte_iterator(train_dataloader),
        ft_iterator=iter(train_dataloader),
        metric_iterator=make_inifinte_iterator(valid_dataloader),
        nsr_final_threshold=config_parsed.nsr_final_threshold,
        ppl_diff_threshold=config_parsed.ppl_diff_threshold,
        num_data_steps=config_parsed.num_data_steps,
        num_metric_steps=config_parsed.num_metric_steps,
        num_ft_steps=config_parsed.num_ft_steps,
        ft_lr=config_parsed.ft_lr,
        min_rank=config_parsed.min_rank,
        trade_off_factor=config_parsed.trade_off_factor,
        num_last_decomposed_layers_to_finetune=num_layers,
        run_finetuning=config_parsed.run_finetuning,
        lora_finetuning=config_parsed.lora_finetuning,
    )

    # Serialize model

    out_decompose_config_path = output_path / "decompose_config.json"
    with open(out_decompose_config_path, "wt") as f:
        json.dump(decompose_config, f)
    out_decompose_state_dict_path = output_path / "decompose_state_dict.pt"
    torch.save(model.state_dict(), out_decompose_state_dict_path)

    # Evaluate model

    with torch.no_grad():
        perplexity_final = metrics.calc_perplexity(
            model, ppl_eval_loader, device, model.config.pad_token_id
        )
    params_final = get_params(model) / 1.0e6
    logger.info(f"{perplexity_orig=} -> {perplexity_final=}")
    logger.info(f"{params_orig=} -> {params_final=}")

    if config_parsed.lm_eval_tasks is not None and len(config_parsed.lm_eval_tasks) > 0:
        lm_eval_results, lm_eval_results_str = metrics.calc_lm_eval_metrics(
            model=model_wrapped.model,
            tokenizer=tokenizer,
            device=device,
            tasks=config_parsed.lm_eval_tasks,
        )
        logger.info(lm_eval_results_str)
        lm_eval_path = output_path / "lm_eval.json"
        with open(lm_eval_path, "wt") as f:
            json.dump(lm_eval_results, f)
        logger.info(f"lm_eval results saved to {lm_eval_path}")


if __name__ == "__main__":
    main()
