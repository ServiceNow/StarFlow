from starvlm.model.coordinates_adapter.base import VLCoordinatesAdapter


class QwenCoordinatesAdapter(VLCoordinatesAdapter):
    def __init__(self, **kwargs) -> None:
        pass

    def __call__(
        self,
        x: int,
        y: int,
        original_width: int,
        original_height: int,
        to_original: bool,
    ) -> tuple[int, int]:
        if to_original:
            x = int(x / 1000 * original_width)
            y = int(y / 1000 * original_height)
        else:
            x = int(x / original_width * 1000)
            y = int(y / original_height * 1000)
        return x, y
