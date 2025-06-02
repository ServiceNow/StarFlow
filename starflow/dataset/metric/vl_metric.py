from abc import ABC, abstractmethod


class VLMetric(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute(self, candidates: list[str], references: list[list[str]]):
        raise NotImplementedError
