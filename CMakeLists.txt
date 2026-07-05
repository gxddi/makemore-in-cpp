cmake_minimum_required(VERSION 3.18)
project(NameGen CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/deps/torch/)
find_package(Torch REQUIRED)

set(ONEAPI_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/deps/oneapi)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})

# Headers: only needed if your own .cpp files #include <sycl/...> or <CL/...> directly
target_include_directories(${PROJECT_NAME} PRIVATE ${ONEAPI_ROOT}/include)

# Link-time search path for the oneAPI libs
target_link_directories(${PROJECT_NAME} PRIVATE ${ONEAPI_ROOT}/lib)

target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})

# Runtime search path: torch's own libs + the oneAPI runtime libs
set(RUNTIME_LIB_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/deps/torch/lib:${ONEAPI_ROOT}/lib")
target_link_options(${PROJECT_NAME} PRIVATE
    "-Wl,-rpath,${RUNTIME_LIB_PATHS}"
    "-Wl,-rpath-link,${RUNTIME_LIB_PATHS}"
)

if(WIN32)
    file(GLOB TORCH_DLLS ${CMAKE_PREFIX_PATH}/lib/*.dll)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()
