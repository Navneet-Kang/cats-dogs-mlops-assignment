from io import BytesIO

import numpy as np
from PIL import Image

from src.catsdogs.features import preprocess_image


def test_preprocess_converts_grayscale_to_224_rgb_normalized():
    stream = BytesIO()
    Image.new("L", (40, 80), color=128).save(stream, format="PNG")
    stream.seek(0)
    array = preprocess_image(stream)
    assert array.shape == (224, 224, 3)
    assert array.dtype == np.float32
    assert 0.49 < float(array.mean()) < 0.51
    assert 0.0 <= float(array.min()) <= float(array.max()) <= 1.0

