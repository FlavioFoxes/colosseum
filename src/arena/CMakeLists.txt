set(LIBRARY arena)

# ================ Dependencies ======================
find_package(Threads REQUIRED)
find_package(booster_robotics_sdk REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(msgpack-cxx REQUIRED)

# MuJoCo backend (optional — enables --backend mujoco at runtime)
find_package(mujoco QUIET)
find_package(glfw3  QUIET)
if(mujoco_FOUND AND glfw3_FOUND)
    message(STATUS "MuJoCo ${mujoco_VERSION} found — MuJoCo backend enabled")
    set(MUJOCO ON)
else()
    message(STATUS "MuJoCo or GLFW not found — MuJoCo backend disabled")
    set(MUJOCO OFF)
endif()

# ====================================================

# =================== Libraries ====================

# Source files
file(
    GLOB_RECURSE SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
list(FILTER SOURCES EXCLUDE REGEX ".*/main\\.cpp$")
list(FILTER SOURCES EXCLUDE REGEX ".*/tests/.*")

if(NOT MUJOCO)
    list(FILTER SOURCES EXCLUDE REGEX ".*/MujocoPortal\\.cpp$")
endif()

file(
    GLOB_RECURSE HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
)

# Debug message
message(STATUS "Building application library '${LIBRARY}':")
foreach(SOURCE ${SOURCES})
    file(RELATIVE_PATH REL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCE})
    message(STATUS "  - ${REL_SOURCE}")
endforeach()

# Library build
add_library(${LIBRARY} OBJECT ${SOURCES})
target_include_directories(
    ${LIBRARY}
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
    ${LIBRARY}
    PUBLIC
        Eigen3::Eigen
        onnxruntime::onnxruntime
        msgpack-cxx
        booster::booster_robotics_sdk
        -Wl,--no-as-needed
        Threads::Threads
        rt
        -Wl,--as-needed
)
if(MUJOCO)
    target_link_libraries(${LIBRARY} PUBLIC mujoco::mujoco glfw)
    target_compile_definitions(${LIBRARY} PUBLIC MUJOCO)
endif()
# ====================================================



# ============================== TESTS ==============================================
find_package(GTest QUIET)

if(GTest_FOUND)
    message(STATUS "GTest found - building tests for ${PROJECT_NAME}")

    # Automatically discover all test source files
    file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
    file(GLOB TEST_FIXTURES "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures.cpp")

    # Optional: Test common library for shared fixtures (only if fixtures exist)
    if(TEST_FIXTURES)
        add_library(${PROJECT_NAME}_test_common STATIC ${TEST_FIXTURES})
        target_include_directories(
            ${PROJECT_NAME}_test_common
            PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tests
        )
        target_link_libraries(
            ${PROJECT_NAME}_test_common
            PUBLIC GTest::gtest ${LIBRARY}
        )
        set(TEST_COMMON_LIB ${PROJECT_NAME}_test_common)
    endif()

    # Testable sources: only units that don't pull in booster DDS static init
    file(
        GLOB_RECURSE TESTABLE_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/OnnxPolicy.cpp"
    )

    # Main test executable
    add_executable(
        ${PROJECT_NAME}_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/main.cpp
        ${TEST_SOURCES}
        ${TESTABLE_SOURCES}
    )

    target_include_directories(
        ${PROJECT_NAME}_tests
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(
        ${PROJECT_NAME}_tests
        PRIVATE GTest::gtest Eigen3::Eigen onnxruntime::onnxruntime ${TEST_COMMON_LIB}
    )

    # Enable CTest integration
    enable_testing()
    add_test(NAME ${PROJECT_NAME}_tests COMMAND ${PROJECT_NAME}_tests)

    install(TARGETS ${PROJECT_NAME}_tests RUNTIME DESTINATION bin)
else()
    message(STATUS "GTest not found - skipping tests for ${PROJECT_NAME}")
endif()
# ===================================================================================

# ================== Installation ====================
install(
    TARGETS ${LIBRARY}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib
)

# Install models from project-level models/ directory (one subdirectory per task).
# Override at cmake time with -DMODELS_DIR=<path> if needed.
if(NOT DEFINED MODELS_DIR)
    set(MODELS_DIR "${CMAKE_SOURCE_DIR}/models")
endif()
if(EXISTS "${MODELS_DIR}")
    install(DIRECTORY "${MODELS_DIR}/" DESTINATION models)
else()
    message(WARNING "arena: models directory not found at ${MODELS_DIR} — skipping model install")
endif()

# ====================================================

# ====================== Main ======================
add_executable(main ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
target_include_directories(
    main
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(main PRIVATE ${LIBRARY})
install(TARGETS main RUNTIME DESTINATION bin)
# ====================================================
