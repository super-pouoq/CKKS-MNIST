@echo off
REM ===== Phantom-FHE build environment setup =====
REM CUDA Toolkit (conda env on F:) + MSVC Build Tools (F:) + Windows SDK
call "F:\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=F:\Miniconda3\envs\phantom-cuda\Library"
set "CUDAToolkit_ROOT=%CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%PATH%"
echo ===== Environment ready =====
where cl
where nvcc
