@echo off
if exist build rmdir /s /q build
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE="../vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build --config Release
