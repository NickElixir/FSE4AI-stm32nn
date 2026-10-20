cmake_minimum_required(VERSION 3.18)
project(stm32nn LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE Torch_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${Torch_DIR}")
find_package(Torch REQUIRED)

find_library(
  TORCH_PYTHON_LIB
  torch_python
  PATHS "${TORCH_INSTALL_PREFIX}/lib"
  REQUIRED
)

add_library(stm32nn_C MODULE src/ops.cpp)
target_compile_features(stm32nn_C PRIVATE cxx_std_17)
target_compile_definitions(stm32nn_C PRIVATE TORCH_EXTENSION_NAME=_C)
target_compile_options(stm32nn_C PRIVATE ${TORCH_CXX_FLAGS})
target_include_directories(stm32nn_C PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(stm32nn_C PRIVATE "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIB}" Python3::Python)
set_target_properties(stm32nn_C PROPERTIES
  PREFIX ""
  OUTPUT_NAME "_C"
  SUFFIX ".so"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/stm32nn"
)
