import collections.abc
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import nvidia.dali.fn  # type:ignore
import nvidia.dali.ops  # type:ignore
import nvidia.dali.pipeline  # type:ignore
import nvidia.dali.plugin.pytorch  # type:ignore
import nvidia.dali.types  # type:ignore
import torch

logger = logging.getLogger(__name__)

DEVICE_MEMORY_PADDING = 211025920
HOST_MEMORY_PADDING = 140544512

DEFAULT_READER_SEED = 1
CPU_DEBUG_MODE = False
DEFAULT_NUMBER_OF_THREADS = 4
DEFAULT_NORMALIZATION = "imagenet"
DEFAULT_RESIZE_INTERP = "cubic"
DEFAULT_RESIZE_SUBPIXEL_SCALE = False
CPU_DEVICE = "cpu"
GPU_DEVICE = "gpu"
CPU_TO_GPU_DEVICE = "mixed"
SHUFFLE_AFTER_EPOCH = True


@dataclass
class DaliDevices:
    cpu_device: str
    gpu_device: str
    cpu_to_gpu: str
    device_id: Optional[int]

    @classmethod
    def create_gpu(cls) -> "DaliDevices":
        return cls(
            cpu_device=CPU_DEVICE,
            gpu_device=GPU_DEVICE,
            cpu_to_gpu=CPU_TO_GPU_DEVICE,
            device_id=0,
        )

    @classmethod
    def create_cpu(cls) -> "DaliDevices":
        return cls(
            cpu_device=CPU_DEVICE,
            gpu_device=CPU_DEVICE,
            cpu_to_gpu=CPU_DEVICE,
            device_id=None,
        )


@dataclass
class ImageNormalizationParams:
    mean: Union[float, tuple[float, float, float]]
    stddev: Union[float, tuple[float, float, float]]
    scale: Union[float, tuple[float, float, float]]
    offset: Union[float, tuple[float, float, float]]


NORMALIZATION = {
    "zero_to_one": ImageNormalizationParams(mean=0, stddev=255, scale=1, offset=0),
    "negative_one_to_one": ImageNormalizationParams(
        mean=0, stddev=255, scale=2, offset=-1
    ),
    "imagenet": ImageNormalizationParams(
        mean=(255 * 0.485, 255 * 0.456, 255 * 0.406),
        stddev=(255 * 0.229, 255 * 0.224, 255 * 0.225),
        scale=1,
        offset=0,
    ),
    "identity": ImageNormalizationParams(stddev=1, scale=1, mean=0, offset=0),
}

INTERPOLATION = {
    "nn": nvidia.dali.types.INTERP_NN,
    "linear": nvidia.dali.types.INTERP_LINEAR,
    "cubic": nvidia.dali.types.INTERP_CUBIC,
    "lanczos3": nvidia.dali.types.INTERP_LANCZOS3,
    "triangular": nvidia.dali.types.INTERP_TRIANGULAR,
    "gaussian": nvidia.dali.types.INTERP_GAUSSIAN,
}


def define_bool_ops() -> nvidia.dali.ops.Cast:
    return nvidia.dali.ops.Cast(dtype=nvidia.dali.types.DALIDataType.BOOL)


class ImagenetPipeline(nvidia.dali.pipeline.Pipeline):
    @staticmethod
    def mux(
        condition: nvidia.dali.pipeline.DataNode,
        true_case: nvidia.dali.pipeline.DataNode,
        false_case: nvidia.dali.pipeline.DataNode,
    ) -> nvidia.dali.pipeline.DataNode:
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case

    def __init__(
        self,
        *,
        batch_size: int,
        resize_image_size: Tuple[int, int],
        num_classes: int,
        shuffle: bool,
        in_training: bool,
        num_threads: int = DEFAULT_NUMBER_OF_THREADS,
        dataset_dir: str,
        image_classes_fname: str,
        normalization: str = DEFAULT_NORMALIZATION,
        resize_before_crop_size: int = 256,
        resize_interp: str = DEFAULT_RESIZE_INTERP,
        resize_subpixel_scale: bool = DEFAULT_RESIZE_SUBPIXEL_SCALE,
        random_crop_in_training: bool = True,
    ):
        if not CPU_DEBUG_MODE:
            self.dali_devices = DaliDevices.create_gpu()
        else:
            self.dali_devices = DaliDevices.create_cpu()

        self.num_classes = num_classes
        self.resize_before_crop_size = resize_before_crop_size
        super().__init__(batch_size, num_threads, device_id=self.dali_devices.device_id)
        self.dataset_dir = dataset_dir
        self.in_training = in_training
        self.shuffle = shuffle

        logger.info(f"Using {normalization=}")

        if normalization not in NORMALIZATION:
            msg_norms = ", ".join(NORMALIZATION.keys())
            msg = f"{normalization=} not supported, use {msg_norms}"
            raise ValueError(msg)
        self.normalization_params = NORMALIZATION[normalization]

        self.image_filenames_and_classes_txt_filepath = image_classes_fname
        self.images_list, self.classes_list = self.get_image_paths_and_classes(
            shuffle=self.shuffle
        )
        self.use_rotation = False
        self.resize_image_size = resize_image_size

        # if shuffle:
        #     self.img_reader = self.define_reader(
        #         shuffle_after_epoch=SHUFFLE_AFTER_EPOCH
        #     )
        # else:
        #     # If shuffle_after_epoch is enabled dataset is also shuffled
        #     # before first epoch (!)
        #     # If you do not want shuffling you need to disable this option
        #     self.img_reader = self.define_reader(shuffle_after_epoch=False)

        self.decode_image_op = nvidia.dali.ops.ImageDecoder(
            device=self.dali_devices.cpu_to_gpu,
            output_type=nvidia.dali.types.RGB,
        )
        if resize_interp not in INTERPOLATION:
            msg_interp = ", ".join(INTERPOLATION.keys())
            msg = f"{resize_interp=} not supported, use {msg_interp}"
            raise ValueError(msg)
        self.resize_interp = INTERPOLATION[resize_interp]

        self.resize_subpixel_scale = resize_subpixel_scale
        self.resize_image_op = nvidia.dali.ops.Resize(
            device=self.dali_devices.gpu_device,
            size=self.resize_before_crop_size,
            mode="not_smaller",
            interp_type=self.resize_interp,
            subpixel_scale=self.resize_subpixel_scale,
        )
        self.bool_op = define_bool_ops()
        self.random_crop_in_training = random_crop_in_training

    @property
    def epoch_size(self) -> int:
        return len(self.images_list) // self.max_batch_size

    @property
    def num_examples(self) -> int:
        return len(self.classes_list)

    def get_image_paths_and_classes(
        self,
        shuffle: Optional[bool] = True,
    ) -> tuple[list[str], list[int]]:
        with open(self.image_filenames_and_classes_txt_filepath, "r") as f:
            lines = [line.rstrip() for line in f]
        if shuffle:
            np.random.shuffle(lines)
        image_names_ = [e.split(" ")[0] for e in lines]
        image_classes = [int(e.split(" ")[1]) for e in lines]
        image_paths = [
            os.path.join(self.dataset_dir, image_name) for image_name in image_names_
        ]
        return image_paths, image_classes

    def define_graph(self) -> list[nvidia.dali.pipeline.DataNode]:
        images, classes = nvidia.dali.fn.readers.file(
            seed=DEFAULT_READER_SEED,
            files=self.images_list,
            shuffle_after_epoch=SHUFFLE_AFTER_EPOCH,
            labels=self.classes_list,
        )
        if not self.in_training:
            images = self.decode_image_op(images)
            images = self.resize_image_op(images)

            images = nvidia.dali.fn.crop_mirror_normalize(
                images,
                device=self.dali_devices.gpu_device,
                dtype=nvidia.dali.types.FLOAT,
                output_layout=nvidia.dali.types.NHWC,
                crop=self.resize_image_size,
                mean=self.normalization_params.mean,
                std=self.normalization_params.stddev,
                scale=self.normalization_params.scale,
                shift=self.normalization_params.offset,
            )

        else:
            if self.random_crop_in_training:
                images = nvidia.dali.fn.decoders.image_random_crop(
                    images,
                    device=self.dali_devices.cpu_to_gpu,
                    output_type=nvidia.dali.types.RGB,
                    device_memory_padding=DEVICE_MEMORY_PADDING,
                    host_memory_padding=HOST_MEMORY_PADDING,
                    random_aspect_ratio=[0.8, 1.25],
                    random_area=[0.1, 1.0],
                    num_attempts=100,
                )
            else:
                images = self.decode_image_op(images)
            images = nvidia.dali.fn.resize(
                images,
                device=self.dali_devices.gpu_device,
                resize_x=self.resize_image_size[0],
                resize_y=self.resize_image_size[1],
                interp_type=self.resize_interp,
                subpixel_scale=self.resize_subpixel_scale,
            )
            mirror = nvidia.dali.fn.random.coin_flip(probability=0.5)
            images = nvidia.dali.fn.crop_mirror_normalize(
                images,
                device=self.dali_devices.gpu_device,
                dtype=nvidia.dali.types.FLOAT,
                output_layout=nvidia.dali.types.NHWC,
                crop=self.resize_image_size,
                mean=self.normalization_params.mean,
                std=self.normalization_params.stddev,
                scale=self.normalization_params.scale,
                shift=self.normalization_params.offset,
                mirror=mirror,
            )
            if self.use_rotation:
                rotate = nvidia.dali.fn.coin_flip(probability=0.5)
                angle_sample = nvidia.dali.fn.random.uniform(range=(-30, 30))
                images_rotated = nvidia.dali.fn.rotate(
                    images,
                    device=self.dali_devices.gpu_device,
                    keep_size=True,
                    fill_value=0,
                    interp_type=nvidia.dali.types.INTERP_LINEAR,
                    angle=angle_sample,
                )

                images = self.mux(self.bool_op(rotate), images_rotated, images)

        classes = classes.gpu() if not CPU_DEBUG_MODE else classes
        classes = nvidia.dali.fn.one_hot(
            classes, num_classes=self.num_classes, device=self.dali_devices.gpu_device
        )
        images = images.gpu() if not CPU_DEBUG_MODE else images
        return [images, classes]


def make_imagenet_trn_pipeline(
    *,
    batch_size: int,
    normalization: str = "IMAGENET",
    trn_image_classes_fname: str,
    imagenet_root_dir: str,
    h_w: tuple[int, int] = (224, 224),
    resize_interp: str = DEFAULT_RESIZE_INTERP,
    resize_subpixel_scale: bool = DEFAULT_RESIZE_SUBPIXEL_SCALE,
) -> ImagenetPipeline:

    return ImagenetPipeline(
        batch_size=batch_size,
        resize_image_size=h_w,
        in_training=True,
        normalization=normalization,
        num_classes=1000,
        shuffle=True,
        image_classes_fname=trn_image_classes_fname,
        dataset_dir=os.path.join(imagenet_root_dir, "train"),
        resize_interp=resize_interp,
        resize_subpixel_scale=resize_subpixel_scale,
    )


def make_imagenet_val_pipeline(
    *,
    batch_size: int,
    normalization: str = "IMAGENET",
    val_image_classes_fname: str,
    imagenet_root_dir: str,
    h_w: tuple[int, int] = (224, 224),
    resize_interp: str = DEFAULT_RESIZE_INTERP,
    resize_subpixel_scale: bool = DEFAULT_RESIZE_SUBPIXEL_SCALE,
) -> ImagenetPipeline:

    return ImagenetPipeline(
        batch_size=batch_size,
        resize_image_size=h_w,
        in_training=False,
        normalization=normalization,
        num_classes=1000,
        shuffle=False,
        image_classes_fname=val_image_classes_fname,
        dataset_dir=os.path.join(imagenet_root_dir, "valid"),
        resize_interp=resize_interp,
        resize_subpixel_scale=resize_subpixel_scale,
    )


def make_imagenet_pipelines(
    *,
    batch_size: int,
    normalization: str = "IMAGENET",
    trn_image_classes_fname: str,
    val_image_classes_fname: str,
    imagenet_root_dir: str,
    h_w: tuple[int, int] = (224, 224),
    resize_interp: str = DEFAULT_RESIZE_INTERP,
    resize_subpixel_scale: bool = DEFAULT_RESIZE_SUBPIXEL_SCALE,
) -> tuple[ImagenetPipeline, ImagenetPipeline]:

    train_pipeline = make_imagenet_trn_pipeline(
        batch_size=batch_size,
        normalization=normalization,
        trn_image_classes_fname=trn_image_classes_fname,
        imagenet_root_dir=imagenet_root_dir,
        h_w=h_w,
        resize_interp=resize_interp,
        resize_subpixel_scale=resize_subpixel_scale,
    )

    valid_pipeline = make_imagenet_val_pipeline(
        batch_size=batch_size,
        normalization=normalization,
        val_image_classes_fname=val_image_classes_fname,
        imagenet_root_dir=imagenet_root_dir,
        h_w=h_w,
        resize_interp=resize_interp,
        resize_subpixel_scale=resize_subpixel_scale,
    )

    return train_pipeline, valid_pipeline


class DaliGenericIteratorWrapper:
    def __init__(self, pt_dali_iter: nvidia.dali.plugin.pytorch.DALIGenericIterator):
        self.pt_dali_iter = pt_dali_iter

    def __len__(self) -> int:
        return self.pt_dali_iter._pipes[0].epoch_size

    def __next__(self) -> dict[str, torch.Tensor]:
        return next(self.pt_dali_iter)[0]

    def __iter__(self) -> collections.abc.Iterator[dict[str, torch.Tensor]]:
        while True:
            try:
                yield self.__next__()
            except StopIteration:
                return

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self.pt_dali_iter, name)
        except AttributeError:
            raise
