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

import configurator
import datasets_dali
import models

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


def create_student_teacher_models(
    config: dict[str, Any], out_decompose_config_path: pathlib.Path
) -> tuple[torch.nn.Module, torch.nn.Module]:
    teacher_model = models.create_model(config)
    b_c_h_w = (1, 3, int(config["input_h_w"][0]), int(config["input_h_w"][1]))

    student_model = models.create_model(config)

    ptdeco.wrap_in_place(student_model)
    sd = torch.load(config["model"]["wrapped_model_checkpoint"])
    student_model.load_state_dict(sd)
    proportion_threshold = config["model"][
        "wrapped_model_decomposition_proportion_threshold"
    ]

    blacklist = config["model"].get("wrapped_model_decomposition_blacklisted_modules")

    decompose_config = ptdeco.decompose_in_place(
        student_model,
        proportion_threshold=proportion_threshold,
        blacklisted_module_names=blacklist,
    )
    student_model.eval()

    with open(out_decompose_config_path, "wt") as f:
        json.dump(decompose_config, f)

    # Compute & log statistics of teacher and student model

    teacher_model.eval()

    teacher_model_gflops = models.get_fpops(
        teacher_model, b_c_h_w=b_c_h_w, units="gflops"
    )
    teacher_model_kmapps = models.get_fpops(
        teacher_model, b_c_h_w=b_c_h_w, units="kmapps"
    )
    teacher_model_params = models.get_params(teacher_model) / 1.0e6
    msg_ops = f"gflops={teacher_model_gflops:.2f} kmapps={teacher_model_kmapps:.2f}"
    msg_par = f"params={teacher_model_params:.2f}"

    student_model_gflops = models.get_fpops(
        student_model, b_c_h_w=b_c_h_w, units="gflops"
    )
    student_model_kmapps = models.get_fpops(
        student_model, b_c_h_w=b_c_h_w, units="kmapps"
    )
    student_model_params = models.get_params(student_model) / 1.0e6
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

    student_model, teacher_model = create_student_teacher_models(
        config, output_path / "decompose_config.json"
    )
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
        max_duration=config["epochs"],
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
