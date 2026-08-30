cmake_minimum_required(VERSION 3.24)

project(
    FluidSimulator
    VERSION 0.1.0
    DESCRIPTION "Position Based Fluids simulator"
    LANGUAGES CXX CUDA
)

find_package(CUDAToolkit REQUIRED)

file(
    GLOB_RECURSE PBF_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

add_library(pbf STATIC ${PBF_SOURCES})

add_library(FluidSimulator::pbf ALIAS pbf)

target_compile_features(
    pbf
    PUBLIC
        cxx_std_20
        cuda_std_20
)

target_include_directories(
    pbf
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    pbf
    PUBLIC
        CUDA::cudart
)

# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------

include(CTest)

if(BUILD_TESTING)

    find_package(GTest REQUIRED)
    include(GoogleTest)

    file(
        GLOB PBF_TEST_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/test/test_*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/test_*.cu"
    )

    function(add_pbf_test test_source)
        get_filename_component(test_target "${test_source}" NAME_WE)

        add_executable("${test_target}" "${test_source}")

        target_link_libraries(
            "${test_target}"
            PRIVATE
                FluidSimulator::pbf
                GTest::gtest_main
        )

        gtest_discover_tests(
            "${test_target}"
            DISCOVERY_MODE PRE_TEST
        )
    endfunction()

    foreach(test_source IN LISTS PBF_TEST_SOURCES)
        add_pbf_test("${test_source}")
    endforeach()

endif()
