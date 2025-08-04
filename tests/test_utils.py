import numpy as np
import pytest

from src.pctl.dataset.utils import get_mosaic_of_centers


@pytest.mark.parametrize(
    "tile_width, subtile_width, subtile_overlap",
    [(1000, 50, 25), (500, 50, 10), (2000, 100, 50)],
)
def test_get_mosaic_of_centers(tile_width, subtile_width, subtile_overlap):
    mosaic = get_mosaic_of_centers(
        tile_width, subtile_width, subtile_overlap=subtile_overlap
    )
    centers = np.stack(mosaic).T
    for x, y in centers:
        assert x - subtile_width / 2 >= 0
        assert x + subtile_width / 2 <= tile_width
        assert y - subtile_width / 2 >= 0
        assert y + subtile_width / 2 <= tile_width
