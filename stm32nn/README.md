# stm32nn

CPU PyTorch extension. `stm32nn.linear` and `stm32nn.conv2d` are FP32 kernels with an explicit accumulation order and an optional fused ReLU. `stm32nn.compare` reports the mean squared Euclidean gap of the last dimension and the fraction of rows whose argmax differs.

`linear(input, weight, bias=None, relu=False)` expects `input` of shape `(N, in)` and `weight` of shape `(out, in)`. The sum is over the input feature.

`conv2d(input, weight, bias=None, stride_h=1, stride_w=1, pad_h=0, pad_w=0, relu=False)` expects NCHW `input` and weight `(out_channels, in_channels, kH, kW)`, one group, dilation 1. For each output site the sum walks input channel, then kernel row, then kernel column.

`max_pool2d` is a separate NCHW pool, matching a generated graph that does not fuse the pool into the convolution.

`round_fp16` casts a float32 tensor to IEEE FP16, nearest, and stores it back in float32.

`to_nhwc` and `to_nchw` reorder a 4-D tensor. NHWC is the channel-last layout used by the generated CNN buffers.

`quantize_per_channel` applies symmetric int8 along one axis (scale = max-abs / 127, stored codes -127..127) and returns the dequantized float32 tensor.

`compare` returns the mean squared Euclidean gap, the fraction of rows whose argmax differs, and the mean per-row cosine.

`check_export(weights, sample, onnx_path=None)` reports float16 parameters, a prime channel count above 64, and a sample that is not a float image divided by 255. An ONNX path is also rejected when it stores FP16.

Build from this directory, with the environment that has PyTorch:

```bash
cmake -S . -B build
cmake --build build
```

The shared library is written to `stm32nn/_C.so`. Then `PYTHONPATH=. pytest -q tests`.
