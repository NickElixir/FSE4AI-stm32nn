import pytest
import torch

from stm32nn import (
    check_export,
    compare,
    conv2d,
    linear,
    max_pool2d,
    quantize_per_channel,
    round_fp16,
    to_nchw,
    to_nhwc,
)


def test_linear_shape_and_dtype():
    y = linear(torch.zeros(3, 4), torch.zeros(5, 4))
    assert y.shape == (3, 5)
    assert y.dtype == torch.float32


def test_linear_matches_hand_values():
    x = torch.tensor([[1.0, 2.0], [0.0, -1.0]])
    w = torch.tensor([[1.0, 0.0], [0.5, -1.0]])
    b = torch.tensor([0.25, 0.0])
    y = linear(x, w, b)
    expect = torch.tensor([[1.25, -1.5], [0.25, 1.0]])
    assert torch.equal(y, expect)


def test_linear_fused_relu_is_not_a_copy_of_the_input():
    x = torch.tensor([[1.0, -2.0]])
    w = torch.tensor([[1.0, 0.0], [-1.0, 1.0]])
    y = linear(x, w, relu=True)
    assert torch.equal(y, torch.tensor([[1.0, 0.0]]))


def test_conv2d_one_pixel_kernel():
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    w = torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]]])
    y = conv2d(x, w, bias=torch.tensor([0.5]), stride_h=1, stride_w=1, pad_h=0, pad_w=0)
    # (0,0): 1*1 + 4*(-1) + 0.5 = -2.5
    assert y.shape == (1, 1, 1, 1)
    assert y.dtype == torch.float32
    assert torch.equal(y, torch.tensor([[[[-2.5]]]]))


def test_conv2d_relu_zeros_a_negative_site():
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    w = torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]]])
    y = conv2d(x, w, relu=True)
    assert torch.equal(y, torch.tensor([[[[0.0]]]]))


def test_compare_gap_and_disagreement():
    logits = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    reference = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    mse, disagree, cosine = compare(logits, reference)
    assert mse == pytest.approx(6.5)
    assert disagree == pytest.approx(0.5)
    assert cosine == pytest.approx(0.5)


def test_integer_input_is_rejected():
    with pytest.raises(RuntimeError):
        linear(torch.ones(2, 2, dtype=torch.int32), torch.ones(2, 2))


def test_round_fp16_matches_pytorch_cast():
    value = torch.tensor([0.1, 1.0, -2.5])
    got = round_fp16(value)
    expect = value.to(torch.float16).to(torch.float32)
    assert torch.equal(got, expect)
    assert got.dtype == torch.float32


def test_nhwc_roundtrip_keeps_values_and_channel_last_stride():
    image = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    packed = to_nhwc(image)
    assert packed.shape == (2, 4, 5, 3)
    assert packed.stride(-1) == 1
    assert torch.equal(to_nchw(packed), image)


def test_quantize_per_channel_uses_127_levels():
    weight = torch.tensor([[0.5, -0.25], [0.0, 0.0]])
    got = quantize_per_channel(weight, axis=0)
    scale = 0.5 / 127.0
    expect = torch.tensor([[127.0 * scale, -64.0 * scale], [0.0, 0.0]])
    assert torch.allclose(got, expect)


def test_max_pool_takes_the_window_max():
    image = torch.tensor([[[[1.0, 3.0], [2.0, 0.0]]]])
    pooled = max_pool2d(image, kernel_h=2, kernel_w=2, stride_h=2, stride_w=2)
    assert torch.equal(pooled, torch.tensor([[[[3.0]]]]))


def test_check_export_flags_range_and_prime_channels():
    fine = check_export([torch.zeros(32, 4)], torch.zeros(1, 1, 8, 8))
    assert fine == []
    flagged = check_export(
        [torch.zeros(257, 3, 3, 3)],
        torch.tensor([255.0]),
    )
    assert any("257" in item for item in flagged)
    assert any("255" in item for item in flagged)
