from typing import Any
import logging
import pathlib
import sys


import torch
import transformers

import datasets

import ptdeco

import datasets_hf
import metrics

PPL_EAVAL_SEQLEN = 2048
PPL_EVAL_BATCH_SIZE = 1
PPL_EVAL_VARIED_SEQLEN = False
PPL_EVAL_SEED = 42




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


def main(config: dict[str, Any], output_path: pathlib.Path) -> None:

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
    #sys.exit()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = transformers.OPTForCausalLM.from_pretrained(
        "facebook/opt-125m",
        torch_dtype=torch.float32,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "facebook/opt-125m", trust_remote_code=True
    )
    model.to(device)
    model.eval()
    ds = datasets.load_dataset("wikitext", name="wikitext-2-raw-v1")

    test_datasetloader = False
    logger.info(f"{test_datasetloader=}")
    if test_datasetloader:
        ppl_eval_loader = datasets_hf.prepare_test_dataloader(
            dataset=ds_test,
            tokenizer=tokenizer,
            max_seqlen=PPL_EAVAL_SEQLEN,
            batch_size=PPL_EVAL_BATCH_SIZE,
        )
    else:
        ppl_eval_loader = datasets_hf.prepare_dataloader(
            dataset=ds_test,
            tokenizer=tokenizer,
            max_seqlen=PPL_EAVAL_SEQLEN,
            batch_size=PPL_EVAL_BATCH_SIZE,
            nsamples=len(ds_test),
            varied_seqlen=PPL_EVAL_VARIED_SEQLEN,
            seed=PPL_EVAL_SEED,
        )
    with torch.no_grad():
        perplexity = metrics.calc_perplexity(
            model, ppl_eval_loader, device, model.config.pad_token_id
        )
    logger.info(f"{perplexity=}")
    # ptdeco.dwain.decompose_in_place(
    #     model=model,
    #     device=device,
    # )

if __name__ == "__main__":
    main()
