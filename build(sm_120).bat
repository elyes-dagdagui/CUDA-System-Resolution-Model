@echo off
echo "Blackwell Architecture Binary Compilation"

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "MSVC_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build"

call "%MSVC_PATH%\vcvarsall.bat" amd64

set "PATH=%CUDA_PATH%\bin;%PATH%"

nvcc -arch=sm_120 equation_predict.cu -o builds\equation_predict_sm89.exe > builds\build_log_sm120.txt

timeout /t 15