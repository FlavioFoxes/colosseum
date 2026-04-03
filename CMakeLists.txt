# ================== PROJECT SETUP ===================
cmake_minimum_required(VERSION 3.22)

project(arena VERSION 0.1 LANGUAGES CXX)

# Use the conda sysroot glibc at runtime so the binary doesn't fall back to
# the host system glibc. Mirrors external/spqrbooster2026 behavior.
if(DEFINED ENV{CONDA_PREFIX} AND NOT CMAKE_CROSSCOMPILING)
    set(_sysroot "$ENV{CONDA_PREFIX}/x86_64-conda-linux-gnu/sysroot")
    if(EXISTS "${_sysroot}/lib64/ld-linux-x86-64.so.2")
        set(CMAKE_BUILD_RPATH "${_sysroot}/lib64;$ENV{CONDA_PREFIX}/lib")
        string(APPEND CMAKE_EXE_LINKER_FLAGS
            " -Wl,-dynamic-linker,${_sysroot}/lib64/ld-linux-x86-64.so.2")
    endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_definitions(PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

add_subdirectory(src/arena)
# ====================================================