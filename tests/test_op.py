import pytest
import torch

from stm32nn import round_fp16


def test_output_shape():
    value = torch.tensor([[0.1, -1.0, 2.0], [4.0, -0.5, 0.25]])
    assert round_fp16(value).shape == (2, 3)


def test_noncontiguous_empty_scalar_shapes():
    value = torch.tensor([[0.1, -1.0, 2.0], [4.0, -0.5, 0.25]])
    for candidate in (value.t(), torch.empty(0, 3), torch.tensor(0.1)):
        result = round_fp16(candidate)
        assert result.shape == candidate.shape
        assert result.is_contiguous()


def test_output_dtype():
    value = torch.tensor([1.0, -2.0, 0.5])
    assert round_fp16(value).dtype == torch.float32


def test_exact_match_on_known_values():
    # Also guards the old subnormal shift bug and binary16 rounding boundaries.
    value = torch.tensor([
        0.1, 1.0, -2.5, 0.0, -0.0, 1e-7, -1e-7, 2**-25, 3 * 2**-25,
        2**-24, 2**-14, 1 + 2**-11, 1 + 3 * 2**-11, 65504.0, 65520.0, -65520.0,
    ])
    expect = value.to(torch.float16).to(torch.float32)
    got = round_fp16(value)
    assert torch.equal(got, expect)
    assert torch.equal(torch.signbit(got), torch.signbit(expect))


def test_distinct_values_are_not_a_copy():
    value = torch.tensor([-1.1, 0.1, 2.2])
    got = round_fp16(value)
    expect = value.to(torch.float16).to(torch.float32)
    assert torch.equal(got, expect)
    assert not torch.equal(got, value)
    assert got[0] != got[1]
    assert got[1] != got[2]


def test_leading_batch_of_four_matches_specification():
    row = torch.tensor([0.1, -1.1, 2.2, -2.5])
    value = torch.stack((row, row, row, row))
    expect = value.to(torch.float16).to(torch.float32)
    assert torch.equal(round_fp16(value), expect)


def test_integer_tensor_raises_runtime_error():
    with pytest.raises(RuntimeError, match="float32"):
        round_fp16(torch.tensor([1, 2, 3]))
