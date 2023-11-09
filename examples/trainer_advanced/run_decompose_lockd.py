# flake8: noqa

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
import composer.optim.scheduler
import nvidia.dali.plugin.pytorch
import ptdeco
import torch
import torch.utils.tensorboard
import torchmetrics

import builder
import configurator
import datasets_dali

TENSORBOARD_DIRNAME = "tensorboard"
CHECKPOINTS_DIRNAME = "checkpoints"


logger = logging.getLogger(__name__)


class ComposerWrappedModel(composer.ComposerModel):
    def __init__(
        self,
        wrapped_model: torch.nn.Module,
        proportion_lambda: float,
        nsr_threshold: float,
        output_path: pathlib.Path,
    ):
        super().__init__()
        self.wrapped_model = wrapped_model
        self.accuracy = torchmetrics.classification.MulticlassAccuracy(
            num_classes=1000, average="micro"
        )
        self.writer = torch.utils.tensorboard.SummaryWriter(
            log_dir=output_path / TENSORBOARD_DIRNAME
        )
        self.proportion_lambda = proportion_lambda
        self.nsr_threshold = nsr_threshold

    def forward(self, batch: composer.core.Batch) -> torch.Tensor:
        self.wrapped_model.eval()
        inputs = batch["inputs"]
        inputs = inputs.permute(0, 3, 1, 2)
        return self.wrapped_model.forward(inputs)

    def loss(
        self, outputs: Any, batch: composer.core.Batch, *args, **kwargs
    ) -> torch.Tensor:
        model_size_loss = ptdeco.lockd.get_proportion_loss(self.wrapped_model)
        nsr_loss = ptdeco.lockd.get_nsr_loss(self.wrapped_model, self.nsr_threshold)
        loss = nsr_loss + self.proportion_lambda * model_size_loss
        return loss

    def get_metrics(self, is_train: bool = False) -> dict[str, torchmetrics.Metric]:
        if is_train:
            metrics: dict[str, torchmetrics.Metric] = {
                "accuracy": self.accuracy,
            }
        else:
            metrics = self.accuracy

        if isinstance(metrics, torchmetrics.Metric):
            metrics_dict = {metrics.__class__.__name__: metrics}
        else:
            metrics_dict = {}
            for name, metric in metrics.items():
                assert isinstance(metric, torchmetrics.Metric)
                metrics_dict[name] = metric
        return metrics_dict

    def update_metric(
        self, batch: Any, outputs: Any, metric: torchmetrics.Metric
    ) -> None:
        targets = torch.argmax(batch["targets"], dim=1)
        predictions = torch.softmax(outputs, dim=1)
        metric.update(predictions, targets)


class TensorboardCallBack(composer.Callback):
    def __init__(self, interval: int = 100):
        super().__init__()
        self.interval = interval

    def batch_end(self, state: composer.State, logger: composer.Logger) -> None:
        batch_num = state.timestamp.batch.value
        if batch_num % self.interval == 0:
            # LOSSES

            state.model.writer.add_scalar("train/loss", state.loss, batch_num)
            with torch.no_grad():
                loss_nsr = ptdeco.lockd.get_nsr_loss(
                    state.model.wrapped_model, state.model.nsr_threshold
                )
            state.model.writer.add_scalar("train/loss_nsr", loss_nsr, batch_num)
            with torch.no_grad():
                loss_proportion = ptdeco.lockd.get_proportion_loss(
                    state.model.wrapped_model
                )
            state.model.writer.add_scalar(
                "train/loss_proportion", loss_proportion, batch_num
            )
            with torch.no_grad():
                loss_entropy = ptdeco.lockd.get_entropy_loss(state.model.wrapped_model)
            state.model.writer.add_scalar("train/loss_entropy", loss_entropy, batch_num)

            # METRICS

            train_accuracy = state.train_metric_values["accuracy"]
            state.model.writer.add_scalar(
                "train/student_acc", train_accuracy, batch_num
            )

            # LEARNING RATE

            state.model.writer.add_scalar(
                "train/lr", state.schedulers[0].get_last_lr()[0], batch_num
            )

            # LOSSES - PARTIAL

            with torch.no_grad():
                nsr_dict = ptdeco.lockd.get_nsr_dict(state.model.wrapped_model)
            for key, nsr in nsr_dict.items():
                key_tb = key.replace(".", "_")
                state.model.writer.add_scalar(
                    f"stage_zero/{key_tb}_nsr", nsr, batch_num
                )

            with torch.no_grad():
                proportion_dict = ptdeco.lockd.get_proportion_dict(
                    state.model.wrapped_model
                )
            for key, prop in proportion_dict.items():
                key_tb = key.replace(".", "_")
                state.model.writer.add_scalar(f"stage_zero/{key_tb}_p", prop, batch_num)


class SaveStageResult(composer.Callback):
    def __init__(self, suffix: str, output_path: pathlib.Path):
        super().__init__()
        self.suffix = suffix
        self.output_path = output_path

    def fit_end(self, state: composer.State, logger: composer.Logger) -> None:
        fname = self.suffix + "_final_sd.pt"
        saving_path = self.output_path / fname
        print(f"Saving stage zero result to: {saving_path}")
        torch.save(state.model.wrapped_model.state_dict(), saving_path)


def get_callbacks(
    config: configurator.DecomposeLOCKDConfig, output_path: pathlib.Path
) -> list[composer.Callback]:
    lr_monitor = composer.callbacks.LRMonitor()
    speed_monitor = composer.callbacks.SpeedMonitor(window_size=50)
    tb_callback = TensorboardCallBack()
    save_result = SaveStageResult(suffix="wrapped", output_path=output_path)
    return [speed_monitor, lr_monitor, tb_callback, save_result]


def main(config: dict[str, Any], output_path: pathlib.Path) -> None:
    config_parsed = configurator.DecomposeLOCKDConfig(**config)
    b_c_h_w = (1, 3, *config_parsed.input_h_w)

    train_pipeline, _ = datasets_dali.make_imagenet_pipelines(
        imagenet_root_dir=config_parsed.imagenet_root_dir,
        trn_image_classes_fname=config_parsed.trn_imagenet_classes_fname,
        val_image_classes_fname=config_parsed.val_imagenet_classes_fname,
        batch_size=config_parsed.batch_size,
        normalization=config_parsed.normalization,
        h_w=config_parsed.input_h_w,
    )

    train_dataloader = datasets_dali.DaliGenericIteratorWrapper(
        nvidia.dali.plugin.pytorch.DALIGenericIterator(
            [train_pipeline], ["inputs", "targets"]
        )
    )

    torch_wrapped_model = builder.create_model(config_parsed.decompose_model_name)
    builder.validate_module_names(
        torch_wrapped_model, config_parsed.blacklisted_modules
    )
    model_orig_stats = builder.get_model_stats(torch_wrapped_model, b_c_h_w)

    ptdeco.lockd.wrap_in_place(
        torch_wrapped_model, blacklisted_module_names=config_parsed.blacklisted_modules
    )
    torch_model_trainable_params = ptdeco.lockd.get_parameters_trainable(
        torch_wrapped_model
    )

    model = ComposerWrappedModel(
        wrapped_model=torch_wrapped_model,
        proportion_lambda=config_parsed.lmbda,
        nsr_threshold=config_parsed.nsr_threshold,
        output_path=output_path,
    )

    device = composer.devices.DeviceGPU()

    optimizers = configurator.get_optimizer(torch_model_trainable_params, config_parsed)
    lr_schedulers = configurator.get_lr_scheduler(config_parsed)
    algorithms = configurator.get_algorithms(config_parsed)
    precision = configurator.get_precision(config_parsed)
    compile_config = configurator.get_compile_config(config_parsed)
    callbacks = get_callbacks(config_parsed, output_path)

    stage_zero_trainer = composer.Trainer(
        model=model,
        train_dataloader=train_dataloader,
        max_duration=config_parsed.max_duration,
        optimizers=optimizers,
        schedulers=lr_schedulers,
        device=device,
        loggers=None,
        algorithms=algorithms,
        autoresume=True,
        save_folder=str(output_path / CHECKPOINTS_DIRNAME),
        save_overwrite=False,
        run_name="decompose_trainable",
        save_interval="1ep",
        callbacks=callbacks,
        precision=precision,
        compile_config=compile_config,
        # Setup capture friendly logging to console
        log_to_console=True,
        progress_bar=False,
        console_log_interval="1000ba",
    )
    stage_zero_trainer.fit()

    # Decompose model
    decompose_config = ptdeco.lockd.decompose_in_place(
        torch_wrapped_model,
        proportion_threshold=config_parsed.proportion_threshold,
        blacklisted_module_names=config_parsed.blacklisted_modules,
    )
    model_deco_stats = builder.get_model_stats(torch_wrapped_model, b_c_h_w)

    # Save decompose_config and state_dict
    out_decompose_config_path = output_path / "decompose_config.json"
    with open(out_decompose_config_path, "wt") as f:
        json.dump(decompose_config, f)
    out_decompose_state_dict_path = output_path / "decompose_state_dict.pt"
    torch.save(model.state_dict(), out_decompose_state_dict_path)

    # Log statistics
    builder.log_model_stats(logger, "Original model  :", model_orig_stats)
    builder.log_model_stats(logger, "Decomposed model:", model_deco_stats)
