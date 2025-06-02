from abc import ABC, abstractmethod
from dataclasses import dataclass
from datasets import concatenate_datasets, load_dataset, load_from_disk
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from typing import Callable
import os


@dataclass(kw_only=True)
class VLExample:
    identifier: str
    images: list[Image.Image]
    queries: list[str]
    annotations: list[str]
    task: str
    source: str


class VLDataset(ABC, Dataset):
    def __init__(self, **kwargs):
        storage = kwargs["storage"]
        tasks = kwargs["tasks"]
        splits = kwargs["splits"]
        is_local = kwargs["is_local"]
        filter_fn = kwargs["filter_fn"]
        min_image_size = kwargs["min_image_size"]
        dummy_image_size = kwargs["dummy_image_size"]
        if not isinstance(tasks, (list, tuple)):
            tasks = [tasks]
        if not isinstance(splits, (list, tuple)):
            splits = [splits]
        subsets = []
        for task in tasks:
            for split in splits:
                if is_local:
                    subset = load_from_disk(os.path.join(storage, task))[split]
                else:
                    subset = load_dataset(
                        storage, task, split=split, num_proc=os.cpu_count()
                    )
                subsets.append(subset)
        dataset = concatenate_datasets(subsets)
        if filter_fn is not None:
            if isinstance(filter_fn, str):
                filter_fn = eval(filter_fn)
            dataset = dataset.filter(
                filter_fn, with_indices=True, num_proc=os.cpu_count()
            )
        self.dataset = dataset
        self.min_image_size = min_image_size
        self.dummy_image_size = dummy_image_size
        self.post_init(**kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        example = self.dataset[index]
        identifier = self.get_identifier(example)
        assert isinstance(identifier, str)
        images = self.get_images(example)
        assert isinstance(images, list)
        for image in images:
            assert isinstance(image, Image.Image)
        queries = self.get_queries(example)
        assert isinstance(queries, list)
        for query in queries:
            assert isinstance(query, str)
        annotations = self.get_annotations(example)
        assert isinstance(annotations, list)
        for annotation in annotations:
            assert isinstance(annotation, str)
        task = self.get_task(example)
        assert isinstance(task, str)
        source = self.get_source(example)
        assert isinstance(source, str)
        try:
            assert len(images) != 0
            for image in images:
                assert min(image.size) >= self.min_image_size
            assert len(queries) == len(annotations) != 0
        except:
            images = [Image.new("RGB", (self.dummy_image_size, self.dummy_image_size))]
            queries = ["Dumy query."]
            annotations = ["Dummy annotation."]
        return VLExample(
            identifier=identifier,
            images=images,
            queries=queries,
            annotations=annotations,
            task=task,
            source=source,
        )

    def get_data_loader(self, collate_fn: Callable, **kwargs):
        return DataLoader(self, collate_fn=collate_fn, **kwargs)

    @abstractmethod
    def post_init(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_identifier(self, example: dict):
        raise NotImplementedError

    @abstractmethod
    def get_images(self, example: dict):
        raise NotImplementedError

    @abstractmethod
    def get_queries(self, example: dict):
        raise NotImplementedError

    @abstractmethod
    def get_annotations(self, example: dict):
        raise NotImplementedError

    @abstractmethod
    def get_task(self, example: dict):
        raise NotImplementedError

    @abstractmethod
    def get_source(self, example: dict):
        raise NotImplementedError
