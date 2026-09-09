cmake_minimum_required(VERSION 3.18)
project(Makemore CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Expose both Torch and oneAPI to CMake's find_package and internal module searches
set(CMAKE_PREFIX_PATH 
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/torch/"
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/oneapi/"
)
set(LOCAL_ONEAPI "${CMAKE_CURRENT_SOURCE_DIR}/deps/oneapi")
set(LOCAL_ONEAPI_LIB "${LOCAL_ONEAPI}/lib")
set(LOCAL_ONEAPI_INCLUDE "${LOCAL_ONEAPI}/include")
set(LOCAL_TORCH_LIB "${CMAKE_CURRENT_SOURCE_DIR}/deps/torch/lib")

set(CMAKE_LIBRARY_PATH "${LOCAL_ONEAPI_LIB}")
set(CMAKE_INCLUDE_PATH "${LOCAL_ONEAPI_INCLUDE}")

# PyTorch's XPU CMake checks for a full oneAPI compiler install (SYCL_ROOT + icx).
# deps/oneapi only has pip-extracted runtime headers/libs, so provide the imported
# targets that FindSYCLToolkit/xpu.cmake would normally create.
set(PYTORCH_FOUND_XPU TRUE)

add_library(torch::sycl INTERFACE IMPORTED)
set_property(TARGET torch::sycl PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${LOCAL_ONEAPI_INCLUDE}")
set_property(TARGET torch::sycl PROPERTY
    INTERFACE_LINK_LIBRARIES
    "${LOCAL_ONEAPI_LIB}/libsycl.so;${LOCAL_ONEAPI_LIB}/libOpenCL.so")

add_library(torch::xpurt INTERFACE IMPORTED)
set_property(TARGET torch::xpurt PROPERTY
    INTERFACE_LINK_LIBRARIES torch::sycl)

find_package(Torch REQUIRED)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})

target_compile_definitions(${PROJECT_NAME} PRIVATE USE_XPU)
target_include_directories(${PROJECT_NAME} PRIVATE "${LOCAL_ONEAPI_INCLUDE}")
target_link_directories(${PROJECT_NAME} PRIVATE "${LOCAL_ONEAPI_LIB}")

target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${TORCH_LIBRARIES}
    "${LOCAL_ONEAPI_LIB}/libmkl_sycl_blas.so.5"
    "${LOCAL_ONEAPI_LIB}/libmkl_sycl_dft.so.5"
    "${LOCAL_ONEAPI_LIB}/libmkl_sycl_lapack.so.5"
    "${LOCAL_ONEAPI_LIB}/libmkl_intel_lp64.so.2"
    "${LOCAL_ONEAPI_LIB}/libmkl_core.so.2"
    "${LOCAL_ONEAPI_LIB}/libmkl_gnu_thread.so.2"
    "${LOCAL_ONEAPI_LIB}/libpti_view.so"
)

# Use RPATH (not RUNPATH) so transitive deps from libtorch_xpu.so can find
# pip-extracted oneAPI libs in deps/oneapi/lib (e.g. libsycl.so.8).
target_link_options(${PROJECT_NAME} PRIVATE
    "-Wl,--disable-new-dtags"
    "-Wl,-rpath,${LOCAL_TORCH_LIB}:${LOCAL_ONEAPI_LIB}"
    "-Wl,-rpath-link,${LOCAL_TORCH_LIB}:${LOCAL_ONEAPI_LIB}"
)

if(WIN32)
    file(GLOB TORCH_DLLS ${CMAKE_PREFIX_PATH}/lib/*.dll)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()
