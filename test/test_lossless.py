import os
import pytest

from conftest import ENCODER_VARIANTS, DECODER_VARIANTS, ALL_BACKENDS
from test_helpers import (
    get_image_dimensions,
    load_codec_or_skip,
    check_cupy_or_skip,
    encode_image,
    decode_image,
    compare_image_to_raw,
)

# Build test combinations
LOSSLESS_PARAMS = [
    (enc, dec, enc_backend, dec_backend)
    for enc in ENCODER_VARIANTS
    for dec in DECODER_VARIANTS
    for enc_backend in ALL_BACKENDS
    for dec_backend in ALL_BACKENDS
]

LOSSLESS_IDS = [
    f"{enc.name}({enc_be})_{dec.name}({dec_be})"
    for enc, dec, enc_be, dec_be in LOSSLESS_PARAMS
]


@pytest.mark.parametrize("enc,dec,enc_backend,dec_backend", LOSSLESS_PARAMS, ids=LOSSLESS_IDS)
def test_lossless_encode_decode(temp_dir, enc, dec, enc_backend, dec_backend, input_image):
    """Test lossless encoding and decoding with all encoder/decoder/backend variants."""

    # Skip unsupported configurations
    if not enc.supports_lossless:
        pytest.skip(f"{enc.name} does not support lossless encoding")
    if not enc.supports_cpu and enc_backend == "cpu":
        pytest.skip(f"{enc.name} does not support CPU backend")
    if dec.cuda_only and dec_backend != "cuda":
        pytest.skip(f"{dec.name} only supports CUDA backend")
    if not dec.supports_cpu and dec_backend == "cpu":
        pytest.skip(f"{dec.name} does not support CPU backend")

    # Load codecs (skips if backend unavailable)
    enc_codec, _, _ = load_codec_or_skip(enc_backend)
    dec_codec, _, _ = load_codec_or_skip(dec_backend)

    if dec.name == "BatchDecoder_exp":
        check_cupy_or_skip()

    # Setup directories
    encoded_dir = os.path.join(temp_dir, "encoded")
    decoded_dir = os.path.join(temp_dir, "decoded")
    os.makedirs(encoded_dir, exist_ok=True)
    os.makedirs(decoded_dir, exist_ok=True)

    width, height = get_image_dimensions(input_image)
    enc_use_gpu = enc_backend in ("cuda", "opencl")
    dec_use_gpu = dec_backend in ("cuda", "opencl")

    # Encode
    encoded_file = encode_image(
        enc_codec, enc, input_image, width, height, encoded_dir, enc_use_gpu, "lossless"
    )
    assert os.path.exists(encoded_file), f"Encoded file was not created by {enc.name}"

    # Decode
    decoded_file = decode_image(
        dec_codec, dec, encoded_file, width, height, decoded_dir, dec_use_gpu
    )
    assert os.path.exists(decoded_file), f"Decoded file was not created by {dec.name}"

    # Verify lossless roundtrip
    assert compare_image_to_raw(input_image, decoded_file), \
        f"Decoded bytes do not match original ({enc.name}[{enc_backend}] -> {dec.name}[{dec_backend}])"
