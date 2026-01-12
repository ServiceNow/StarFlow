from abc import ABC, abstractmethod


class VLCoordinatesAdapter(ABC):
    @abstractmethod
    def __init__(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        x: int,
        y: int,
        original_width: int,
        original_height: int,
        to_original: bool,
    ) -> tuple[int, int]:
        raise NotImplementedError
