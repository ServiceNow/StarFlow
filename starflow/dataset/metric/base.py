from abc import ABC, abstractmethod
from typing import Any


class VLMetric(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self, candidates: list[str], references: list[list[str]]
    ) -> dict[str, Any]:
        raise NotImplementedError
