from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    ext_modules=[
        CppExtension(
            name="stm32nn._C",
            sources=["src/stm32nn/ops.cpp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
