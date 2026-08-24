mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=D:/code/project/veryBegin/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release