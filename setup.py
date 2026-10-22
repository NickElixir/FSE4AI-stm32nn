import os
import platform

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# CPython from python.org may be universal2, but torch is a native wheel.
# Build and label this extension for the current Mac architecture only.
options = {}
if platform.system() == "Darwin":
    os.environ.setdefault("ARCHFLAGS", f"-arch {platform.machine()}")
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "14.0")
    options["bdist_wheel"] = {"plat_name": f"macosx_14_0_{platform.machine()}"}

setup(
    options=options,
    ext_modules=[
        CppExtension(
            name="stm32nn._C",
            sources=["src/stm32nn/ops.cpp"],
            depends=["src/stm32nn/fp16.h"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
