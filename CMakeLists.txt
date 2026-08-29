cmake_minimum_required(VERSION 3.24)

project(
    FluidSimulator
    VERSION 0.1.0
    DESCRIPTION "Position Based Fluids simulator"
    LANGUAGES CXX CUDA
)

find_package(CUDAToolkit REQUIRED)

add_library(pbf INTERFACE)

add_library(FluidSimulator::pbf ALIAS pbf)

target_compile_features(
    pbf
    INTERFACE
        cxx_std_17
)

target_include_directories(
    pbf
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    pbf
    INTERFACE
        CUDA::cudart
)

# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------

include(CTest)

if(BUILD_TESTING)

    find_package(GTest REQUIRED)

    add_executable(
        test_cuda_buffer
        test/test_cuda_buffer.cu
    )

    target_link_libraries(
        test_cuda_buffer
        PRIVATE
            FluidSimulator::pbf
            CUDA::cudart
            GTest::gtest_main
    )

    set_target_properties(
        test_cuda_buffer
        PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
    )

    add_test(
        NAME CudaBufferIntegration
        COMMAND test_cuda_buffer
    )

endif()
