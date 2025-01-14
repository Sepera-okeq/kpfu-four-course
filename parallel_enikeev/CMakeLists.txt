cmake_minimum_required (VERSION 3.10)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("task1")

find_package(Matplot++ CONFIG REQUIRED)
find_package(OpenMP)

add_executable (task1 "task1.cpp")


if(OpenMP_CXX_FOUND)
    target_link_libraries(task1 PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(task1 PRIVATE Matplot++::cimg  Matplot++::matplot)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET task1 PROPERTY CXX_STANDARD 20)
endif()
