# Specify the minimum version of CMake required to build this project.
cmake_minimum_required(VERSION 3.10) # 3.15+ recommended for newer features if possible

# Define the project name, version (optional), and language(s).
project(GMPHD_MAF VERSION 1.0 LANGUAGES CXX)

# Set the C++ standard to use.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Find Dependencies ---
# Find OpenMP (using modern targets if available)
find_package(OpenMP)
if(OpenMP_FOUND)
    message(STATUS "OpenMP found and enabled.")
else()
    message(WARNING "OpenMP not found. Building without OpenMP support.")
endif()

# Find Boost
set(Boost_USE_STATIC_LIBS ON)       # Prefer static Boost libraries
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)   # Be cautious with static runtime linking
find_package(Boost 1.74.0 COMPONENTS filesystem regex REQUIRED) # Mark as REQUIRED

# Find OpenCV
find_package(OpenCV REQUIRED) # Mark as REQUIRED

# --- Source Files ---
# Use GLOB_RECURSE to find all .cpp files within the GMPHD_MAF directory.
# CONFIGURE_DEPENDS ensures CMake re-runs if files are added/removed.
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/GMPHD_MAF/*.cpp")

# --- Executable Target ---
# Define the executable using the project name and ALL discovered source files.
add_executable(${PROJECT_NAME} ${SOURCES})

# --- Include Directories ---
# Add include directories needed by the target.
target_include_directories(${PROJECT_NAME} PUBLIC # Or PRIVATE if headers are only internal
    "${CMAKE_CURRENT_SOURCE_DIR}/GMPHD_MAF" # Location of your project headers
    ${OpenCV_INCLUDE_DIRS}                  # OpenCV headers
    # Boost include directories are typically handled automatically via target_link_libraries
)

# --- Link Libraries ---
# Link the necessary libraries to the executable target.
target_link_libraries(${PROJECT_NAME} PRIVATE # PRIVATE is usually correct for executables
    Boost::filesystem   # Modern Boost target
    Boost::regex        # Modern Boost target
    ${OpenCV_LIBS}      # OpenCV libraries (or use modern targets like OpenCV::opencv_core if preferred)
)

# Link OpenMP if found (Modern CMake >= 3.9 approach)
if(OpenMP_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

# --- Optional: Build Types ---
# It's generally better to specify this on the command line: cmake -DCMAKE_BUILD_TYPE=Release ..
# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
# endif()

# --- Optional: Installation ---
# install(TARGETS ${PROJECT_NAME} DESTINATION bin) # Installs executable to <prefix>/bin
# Example: Install headers from GMPHD_MAF directory
# install(
#    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/GMPHD_MAF/
#    DESTINATION include/GMPHD_MAF
#    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
# )

# Print a message upon configuration completion
message(STATUS "CMake configuration complete for ${PROJECT_NAME}.")

