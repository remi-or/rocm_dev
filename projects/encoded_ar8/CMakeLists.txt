cmake_minimum_required(VERSION 3.18)
project(msccl_allreduce CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find HIP package
find_package(hip REQUIRED)
find_package(mscclpp REQUIRED)

# Enable HIP
enable_language(HIP)

# Set HIP flags for conversion
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --cuda-path=/opt/rocm/hip")

# Add executable - note that we'll convert .cpp and .cu files to .hip
add_executable(allreduce
    main.cpp
    allreduce.cpp
)

# Set source file properties for HIPIFY conversion
set_source_files_properties(
    allreduce.cpp
    main.cpp
    PROPERTIES
    HIP_SOURCE_PROPERTY_FORMAT 1
)

# Link libraries
target_link_libraries(allreduce
    PRIVATE
    mscclpp::mscclpp
    hip::device
)

# Convert CUDA syntax to HIP
set_target_properties(allreduce PROPERTIES
    HIP_ARCHITECTURES "gfx908;gfx90a"  # Adjust these based on your AMD GPU architecture
)
