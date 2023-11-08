import collections
import json
import logging
import pathlib
from typing import Any

import composer
import composer.algorithms
import composer.callbacks
import composer.core
import composer.devices
import composer.optim
import nvidia.dali.plugin.pytorch
import ptdeco
import torch
import torchmetrics

import builder
import configurator
import datasets_dali

TENSORBOARD_DIRNAME = "tensorboard"
CHECKPOINTS_DIRNAME = "checkpoints"


logger = logging.getLogger(__name__)


def kl_divergence(
    q_logits: torch.Tensor,
    p_logits: torch.Tensor,
) -> torch.Tensor:
    q_prob = torch.softmax(q_logits, dim=-1)
    p_prob = torch.softmax(p_logits, dim=-1)
    return torch.multiply(p_prob, torch.log(torch.divide(p_prob, q_prob))).sum(dim=1)


def kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return torch.maximum(
        kl_divergence(student_logits, teacher_logits),
        kl_divergence(teacher_logits, student_logits),
    ).mean()


class KdClassificationModel(composer.ComposerModel):
    def __init__(
        self,
        student_model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        output_path: pathlib.Path,
        eval_mode: bool = False,
    ):
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.accuracy = torchmetrics.classification.MulticlassAccuracy(
            num_classes=1000, average="micro"
        )
        self.writer = torch.utils.tensorboard.SummaryWriter(
            log_dir=output_path / TENSORBOARD_DIRNAME
        )
        self.eval_mode = eval_mode

    def forward(self, batch: composer.core.Batch) -> tuple[torch.Tensor, torch.Tensor]:
        if self.eval_mode:
            self.student_model.eval()
        self.teacher_model.eval()
        inputs = batch["inputs"]
        inputs = inputs.permute(0, 3, 1, 2)

        with torch.no_grad():
            y_teacher = self.teacher_model(inputs)
        y_student = self.student_model.forward(inputs)

        return y_student, y_teacher

    def loss(self, outputs: Any, batch: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        y_student = outputs[0]
        y_teacher = outputs[1]
        loss = kl_loss(student_logits=y_student, teacher_logits=y_teacher)
        return loss

    def get_metrics(self, is_train: bool = False) -> dict[str, torchmetrics.Metric]:
        if is_train:
            metrics = {
                "accuracy": self.accuracy,
            }
        else:
            metrics = self.accuracy
        metrics_dict: dict[str, torchmetrics.Metric] = {}
        if isinstance(metrics, torchmetrics.Metric):
            metrics_dict[metrics.__class__.__name__] = metrics
        else:
            for name, metric in metrics.items():
                assert isinstance(metric, torchmetrics.Metric)
                metrics_dict[name] = metric

        return metrics_dict

    def update_metric(
        self, batch: Any, outputs: Any, metric: torchmetrics.Metric
    ) -> None:
        targets = torch.argmax(batch["targets"], dim=1)
        predictions = torch.softmax(outputs[0], dim=1)
        metric.update(predictions, targets)


class TensorboardCallBack(composer.Callback):
    def __init__(self, interval: int = 100):
        super().__init__()
        self.interval = interval

    def batch_end(self, state: composer.State, logger: composer.Logger) -> None:
        batch_num = state.timestamp.batch.value
        if batch_num % self.interval == 0:
            y_teacher = state.outputs[1]
            target = torch.argmax(state.batch["targets"], dim=1)
            predicted_teacher = torch.argmax(torch.softmax(y_teacher, dim=1), dim=1)
            teacher_accuracy = (predicted_teacher == target).to(torch.float32).mean()
            # log eval metrics
            train_accuracy = state.train_metric_values["accuracy"]
            state.model.writer.add_scalar(
                "train/teacher_acc", teacher_accuracy, batch_num
            )
            state.model.writer.add_scalar(
                "train/student_acc", train_accuracy, batch_num
            )
            state.model.writer.add_scalar("train/loss", state.loss, batch_num)
            state.model.writer.add_scalar(
                "train/lr", state.schedulers[0].get_last_lr()[0], batch_num
            )

    def eval_end(self, state: composer.State, logger: composer.Logger) -> None:
        epoch = state.timestamp.epoch.value
        eval_accuracy = list(state.eval_metric_values.values())[0]
        state.model.writer.add_scalar("valid/accuracy", eval_accuracy, epoch)


def filter_decompose_config(
    decompose_config: dict[str, Any],
    proportion_threshold: float,
    blacklisted_module_names: list[str],
) -> tuple[dict[str, Any], list[str]]:
    decompose_config_filtered = {}
    skipped_module_names = []

    for module_name, module_data in decompose_config.items():
        meta = module_data[ptdeco.MODCONFIG_META_KEY]
        proportion = meta["proportion"]
        if module_name in blacklisted_module_names:
            logger.info(f"Skippping decomposition - {module_name} module blacklisted")
            skipped_module_names.append(module_name)
            continue
        if proportion > proportion_threshold:
            msg = f"{proportion=:.3} > {proportion_threshold} proportion too large"
            logger.info(f"Skippping decomposition - {module_name} {msg}")
            skipped_module_names.append(module_name)
            continue

        decompose_config_filtered[module_name] = module_data

    return decompose_config_filtered, skipped_module_names


def filter_state_dict(
    sd: collections.OrderedDict[str, torch.Tensor], skipped_module_names: list[str]
) -> collections.OrderedDict[str, torch.Tensor]:
    filtered_sd = collections.OrderedDict()
    for k, v in sd.items():
        keep = True
        for m in skipped_module_names:
            if k.startswith(m):
                logger.info(f"Removing {k} from model state_dict ({m} not decomposed)")
                keep = False
                break
        if keep:
            filtered_sd[k] = v
    return filtered_sd


def create_decomposed_model(config) -> torch.nn.Module:
    model_name = config["model_name"]
    model = builder.create_model(model_name)
    with open(config["decompose_config"], "rt") as f:
        decompose_config = json.load(f)

    decompose_state_dict = torch.load(config["decompose_state_dict"])
    proportion_threshold = config["decompose_proportion_threshold"]
    blacklisted_module_names = config.get("decompose_blaclisted_modules")

    decompose_config, skipped_module_names = filter_decompose_config(
        decompose_config, proportion_threshold, blacklisted_module_names
    )
    decompose_state_dict = filter_state_dict(decompose_state_dict, skipped_module_names)
    ptdeco.apply_decompose_config_in_place(model, decompose_config)
    model.load_state_dict(decompose_state_dict, strict=False)
    return model


def create_student_teacher_models(
    config: dict[str, Any]
) -> tuple[torch.nn.Module, torch.nn.Module]:
    model_name = config["model_name"]
    teacher_model = builder.create_model(model_name)
    student_model = create_decomposed_model(config)

    # Compute statistics of teacher model

    b_c_h_w = (1, 3, int(config["input_h_w"][0]), int(config["input_h_w"][1]))

    teacher_model.eval()

    teacher_model_gflops = builder.get_fpops(
        teacher_model, b_c_h_w=b_c_h_w, units="gflops"
    )
    teacher_model_kmapps = builder.get_fpops(
        teacher_model, b_c_h_w=b_c_h_w, units="kmapps"
    )
    teacher_model_params = builder.get_params(teacher_model) / 1.0e6
    msg_ops = f"gflops={teacher_model_gflops:.2f} kmapps={teacher_model_kmapps:.2f}"
    msg_par = f"params={teacher_model_params:.2f}"

    # Compute statistics of student model model

    student_model.eval()
    student_model_gflops = builder.get_fpops(
        student_model, b_c_h_w=b_c_h_w, units="gflops"
    )
    student_model_kmapps = builder.get_fpops(
        student_model, b_c_h_w=b_c_h_w, units="kmapps"
    )
    student_model_params = builder.get_params(student_model) / 1.0e6
    student_model.train()

    logger.info(f"Teacher model {msg_ops} {msg_par}")
    msg_ops = f"gflops={student_model_gflops:.2f} kmapps={student_model_kmapps:.2f}"
    msg_par = f"params={student_model_params:.2f}"
    logger.info(f"Student model {msg_ops} {msg_par}")

    return student_model, teacher_model


def get_callbacks(
    config: dict[str, Any], output_path: pathlib.Path
) -> list[composer.Callback]:
    speed_monitor = composer.callbacks.SpeedMonitor(window_size=50)
    lr_monitor = composer.callbacks.LRMonitor()
    tb_callback = TensorboardCallBack()
    # valid_callback = ValidationCallBack(valid_pipeline)
    return [speed_monitor, lr_monitor, tb_callback]


def main(config: dict[str, Any], output_path: pathlib.Path) -> None:
    config_class = configurator.FinetuneConfig(**config)
    print(config_class)

    train_pipeline, valid_pipeline = datasets_dali.make_imagenet_pipelines(
        imagenet_root_dir=config["imagenet_root_dir"],
        trn_image_classes_fname=config["trn_imagenet_classes_fname"],
        val_image_classes_fname=config["val_imagenet_classes_fname"],
        batch_size=config["batch_size"],
        normalization=config["normalization"],
        h_w=(int(config["input_h_w"][0]), int(config["input_h_w"][1])),
    )

    train_dataloader = datasets_dali.DaliGenericIteratorWrapper(
        nvidia.dali.plugin.pytorch.DALIGenericIterator(
            [train_pipeline], ["inputs", "targets"]
        )
    )

    valid_dataloader = datasets_dali.DaliGenericIteratorWrapper(
        nvidia.dali.plugin.pytorch.DALIGenericIterator(
            [valid_pipeline], ["inputs", "targets"]
        )
    )

    student_model, teacher_model = create_student_teacher_models(config)

    model = KdClassificationModel(
        student_model=student_model,
        teacher_model=teacher_model,
        output_path=output_path,
    )

    device = composer.devices.DeviceGPU()

    optimizers = configurator.get_optimizer(model.student_model.parameters(), config)
    lr_schedulers = configurator.get_lr_scheduler(config)
    algorithms = configurator.get_algorithms(config)
    precision = configurator.get_precision(config)
    compile_config = configurator.get_compile_config(config)
    callbacks = get_callbacks(config, output_path)

    evaluator = composer.Evaluator(
        dataloader=valid_dataloader,
        label="eval",
    )

    logger.info(f"Using {precision=}")

    trainer = composer.Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=evaluator,
        max_duration=config["max_duration"],
        optimizers=optimizers,
        schedulers=lr_schedulers,
        device=device,
        algorithms=algorithms,
        autoresume=True,
        save_folder=str(output_path / CHECKPOINTS_DIRNAME),
        save_overwrite=False,
        run_name="train_decomposed",
        save_interval="1ep",
        callbacks=callbacks,
        eval_interval="1ep",
        precision=precision,
        compile_config=compile_config,
        # Setup capture friendly logging to console
        loggers=None,
        log_to_console=True,
        progress_bar=False,
        console_log_interval="1000ba",
    )
    trainer.fit()