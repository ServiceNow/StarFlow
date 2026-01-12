from abc import ABC, abstractmethod
from dataclasses import dataclass
from datasets import DatasetDict, load_dataset, load_from_disk
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Optional
from starvlm.model.coordinates_adapter.base import VLCoordinatesAdapter
import os


@dataclass
class VLExample:
    identifier: str
    images: list[Image.Image]
    queries: list[str]
    annotations: list[str]


class VLDataset(ABC, Dataset):
    def __init__(self, **kwargs) -> None:
        dataset_path = kwargs["dataset_path"]
        subset = kwargs["subset"]
        split = kwargs["split"]
        is_local = kwargs["is_local"]
        filter_fn = kwargs["filter_fn"]
        min_image_size = kwargs["min_image_size"]
        max_image_size = kwargs["max_image_size"]
        dummy_image_size = kwargs["dummy_image_size"]
        if is_local:
            dataset_dir = Path(dataset_path).expanduser()
            if subset is not None:
                dataset_dir = dataset_dir.joinpath(subset)
            if not dataset_dir.is_dir():
                raise ValueError(f"dataset_dir '{dataset_dir}' is not valid directory")
            try:
                dataset = load_from_disk(str(dataset_dir))
            except Exception as e:
                raise RuntimeError(
                    f"failed to load dataset from '{dataset_dir}' on local file system"
                ) from e
            if isinstance(dataset, DatasetDict):
                if split is None:
                    raise ValueError("split is None but type of dataset is DatasetDict")
                if split not in dataset:
                    raise KeyError(f"dataset does not contain split '{split}'")
                dataset = dataset[split]
            else:
                if split is not None:
                    raise ValueError("split is not None but type of dataset is Dataset")
        else:
            try:
                dataset = load_dataset(
                    dataset_path, name=subset, split=split, num_proc=os.cpu_count()
                )
            except Exception as e:
                raise RuntimeError(
                    f"failed to load dataset from '{dataset_path}' on Hugging Face Hub and get split '{split}' of subset '{subset}'"
                ) from e
        if filter_fn is not None:
            try:
                dataset = dataset.filter(
                    eval(filter_fn), with_indices=True, num_proc=os.cpu_count()
                )
            except Exception as e:
                raise RuntimeError(
                    f"failed to filter dataset with filter_fn '{filter_fn}'"
                ) from e
        self.dataset = dataset
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        self.dummy_image_size = dummy_image_size
        self._coordinates_adapter = None
        self.post_init(**kwargs)

    @property
    def coordinates_adapter(self) -> Optional[VLCoordinatesAdapter]:
        return self._coordinates_adapter

    @coordinates_adapter.setter
    def coordinates_adapter(
        self, coordinates_adapter: Optional[VLCoordinatesAdapter]
    ) -> None:
        self._coordinates_adapter = coordinates_adapter

    @abstractmethod
    def post_init(self, **kwargs) -> None:
        raise NotImplementedError

    def __getitem__(self, index: int) -> VLExample:
        example = self.dataset[index]
        use_dummy = False
        identifier = self.get_identifier(example)
        if not isinstance(identifier, str):
            raise TypeError(
                f"type of identifier must be str, got {type(identifier).__name__}"
            )
        try:
            images = self.get_images(example)
        except:
            images = []
        if not isinstance(images, list):
            raise TypeError(f"type of images must be list, got {type(images).__name__}")
        if images:
            for image in images:
                if not isinstance(image, Image.Image):
                    raise TypeError(
                        f"type of each image in images must be Image.Image, got {type(image).__name__}"
                    )
                if (
                    min(image.size) < self.min_image_size
                    or max(image.size) > self.max_image_size
                ):
                    use_dummy = True
        else:
            use_dummy = True
        try:
            queries = self.get_queries(example)
        except:
            queries = []
        if not isinstance(queries, list):
            raise TypeError(
                f"type of queries must be list, got {type(queries).__name__}"
            )
        if queries:
            for query in queries:
                if not isinstance(query, str):
                    raise TypeError(
                        f"type of each query in queries must be str, got {type(query).__name__}"
                    )
        else:
            use_dummy = True
        try:
            annotations = self.get_annotations(example)
        except:
            annotations = []
        if not isinstance(annotations, list):
            raise TypeError(
                f"type of annotations must be list, got {type(annotations).__name__}"
            )
        if annotations:
            for annotation in annotations:
                if not isinstance(annotation, str):
                    raise TypeError(
                        f"type of each annotation in annotations must be str, got {type(annotation).__name__}"
                    )
        else:
            use_dummy = True
        if len(queries) != len(annotations):
            use_dummy = True
        if use_dummy:
            images = [Image.new("RGB", (self.dummy_image_size, self.dummy_image_size))]
            queries = ["Dummy query."]
            annotations = ["Dummy annotation."]
        vl_example = VLExample(
            identifier=identifier,
            images=images,
            queries=queries,
            annotations=annotations,
        )
        return vl_example

    def __len__(self) -> int:
        length = len(self.dataset)
        return length

    @abstractmethod
    def get_identifier(self, example: dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_images(self, example: dict[str, Any]) -> list[Image.Image]:
        raise NotImplementedError

    @abstractmethod
    def get_queries(self, example: dict[str, Any]) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_annotations(self, example: dict[str, Any]) -> list[str]:
        raise NotImplementedError
