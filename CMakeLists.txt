# Specify the minimum version of CMake required to build this project.
# Version 3.10 is a reasonable baseline supporting modern C++ standards.
cmake_minimum_required(VERSION 3.10)

# Define the project name, version (optional), and language(s).
# Replace "MyProject" with the actual name of your project.
project(GMPHD_MAF VERSION 1.0 LANGUAGES CXX)

# Set the C++ standard to use. C++17 is a common modern choice.
# CMAKE_CXX_STANDARD_REQUIRED ensures CMake errors if the compiler doesn't support it.
# CMAKE_CXX_EXTENSIONS OFF disables GNU extensions (like -std=gnu++17 vs -std=c++17).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Target Architecture (Linux x86_64) ---
# On a native Linux x86_64 system, CMake and the compiler (like GCC or Clang)
# typically detect and target x86_64 automatically. You usually don't need
# explicit flags unless you are cross-compiling or need specific microarchitecture
# optimizations.
#
# If you needed to add specific flags (e.g., for optimization):
# if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
#   message(STATUS "Configuring for Linux x86_64")
#   # Example: Add optimization flags specific to x86_64 if desired
#   # target_compile_options(your_target PRIVATE -march=native) # Optimize for the build machine
#   # Or specific flags like:
#   # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
# else()
#   # This message might appear during cross-compilation setup
#   message(STATUS "Target system processor: ${CMAKE_SYSTEM_PROCESSOR}. Ensure toolchain is set correctly for x86_64.")
# endif()
# --- End Target Architecture Section ---

# --- Source Files ---
# List your source files here.
# Option 1: List explicitly (Recommended for clarity and avoiding issues if files are added/removed)
# set(SOURCES
#     src/main.cpp
#     src/utils.cpp
#     src/core/data.cpp
# )

# Option 2: Use GLOB to find files automatically (Convenient but requires re-running CMake when files change)
# Assumes source files are in a 'src' directory and its subdirectories. Adjust as needed.
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "GMPHD_MAF/*.cpp")
# CONFIGURE_DEPENDS helps re-run CMake automatically if files matching the pattern change.

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(Boost_USE_STATIC_LIBS ON)  # Use static Boost libraries
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.74.0 COMPONENTS filesystem regex)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

if(Boost_FOUND)
    # --- Executable Target ---
    # Define the executable target and link the source files.
    # Replace "my_executable" with the desired name for your output binary.
    add_executable(${PROJECT_NAME} GMPHD_MAF/demo_GMPHD_MAF.cpp) # Using PROJECT_NAME as executable name is common
    target_link_libraries(${PROJECT_NAME} Boost::filesystem Boost::regex ${OpenCV_LIBS})
endif()

# --- Include Directories ---
# Tell CMake where to find your header files.
# Assumes header files are in an 'include' directory relative to CMakeLists.txt.
target_include_directories(  # Use target_include_directories for better scoping
    ${PROJECT_NAME}          # Apply includes to the target defined below
    # Add other include paths here if necessary
    # PUBLIC /usr/include/x86_64-linux-gnu/sys/ # System includes
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/GMPHD_MAF # If headers are alongside sources
)

# --- Dependencies / Linking Libraries ---
# If your project depends on external libraries, find them and link them.
# Example using find_package for the Threads library (common on Linux):
# find_package(Threads REQUIRED)
# target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

# Example for a hypothetical library "MyLib":
# find_package(MyLib REQUIRED) # Assumes MyLib provides MyLibConfig.cmake or FindMyLib.cmake
# target_link_libraries(${PROJECT_NAME} PRIVATE MyLib::MyLib)

# If linking simple system libraries (less common with modern CMake):
# target_link_libraries(${PROJECT_NAME} PRIVATE m dl) # Example linking math library and dynamic loading library

# --- Optional: Build Types ---
# CMake typically defaults to Debug builds if none is specified.
# You can set a default build type if desired, although it's often
# better to specify it during the configure step (e.g., cmake -DCMAKE_BUILD_TYPE=Release ..)
# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
# endif()

# --- Optional: Installation ---
# Define rules for installing the executable and potentially headers/libraries.
# install(TARGETS ${PROJECT_NAME} DESTINATION bin) # Installs executable to <prefix>/bin
# install(DIRECTORY include/ DESTINATION include)   # Installs contents of include/ dir to <prefix>/include

# Print a message upon configuration completion
message(STATUS "CMake configuration complete for ${PROJECT_NAME}.")

