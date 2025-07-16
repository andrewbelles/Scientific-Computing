cmake_minimum_required(VERSION 3.24) 
project(Fluid LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Tested on ada (89) 
set(CMAKE_CUDA_ARCHITECTURES 89)
# -rdc=true 
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# needed for lambda expressions and math in __device__ 
add_compile_options(
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -expt-relaxed-constexpr>
)

add_executable(fluid main.cu)

# Get libraries 
find_package(CUDAToolkit       REQUIRED)   # CUDA::cudart
find_package(PkgConfig         REQUIRED)
pkg_check_modules(RAYLIB raylib REQUIRED)  # raylib via pkg‑config
find_package(OpenGL            REQUIRED)   # OpenGL::GL
find_package(GLEW              REQUIRED)   # GLEW::GLEW
find_package(X11               REQUIRED)   # X11::X11

# Link
target_include_directories(fluid PRIVATE
  ${RAYLIB_INCLUDE_DIRS}
)

target_link_libraries(fluid PRIVATE
  CUDA::cudart
  ${RAYLIB_LINK_LIBRARIES}      # usually “raylib”
  GLEW::GLEW
  OpenGL::GL
  X11::X11
  m                             # libm
)
