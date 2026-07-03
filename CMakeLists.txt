cmake_minimum_required(VERSION 3.18)
project(NameGen CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point to local torch installation
set(CMAKE_PREFIX_PATH /usr/local/Cellar/pytorch/2.12.1/)

find_package(Torch REQUIRED)

# Glob all source files in src/
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})

if(WIN32)
    file(GLOB TORCH_DLLS ${CMAKE_PREFIX_PATH}/lib/*.dll)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()
