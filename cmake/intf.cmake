add_library(intf_pub INTERFACE)
target_compile_options(
  intf_pub
  INTERFACE
    -fPIC
    -fvisibility=hidden
    -fvisibility-inlines-hidden
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-O0
    -g>
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-ftrapv
    -fstack-check>
    $<$<COMPILE_LANGUAGE:C>:-pthread
    -Wfloat-equal
    -Wshadow
    -Wformat=2
    -Wno-deprecated
    -Wextra>
    $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
)
target_compile_definitions(
  intf_pub INTERFACE _GLIBCXX_USE_CXX11_ABI=0
                     $<$<CONFIG:Release>:_FORTIFY_SOURCE=2>)
target_include_directories(
  intf_pub INTERFACE ${ASCEND_CANN_PACKAGE_PATH}/include
                     ${PROJECT_SOURCE_DIR}/include)
# if the CANN_PATHS not empty
if(CANN_PATHS)
  # if the arch is aarch64, add the include path
  if(${ARCH} STREQUAL "aarch64")
    target_include_directories(intf_pub
                               INTERFACE ${CANN_PATHS}/aarch64-linux/include)
    target_link_directories(intf_pub INTERFACE
                            ${CANN_PATHS}/aarch64-linux/lib64)
  else()
    target_include_directories(intf_pub
                               INTERFACE ${CANN_PATHS}/x86_64-linux/include)
    target_link_directories(intf_pub INTERFACE ${CANN_PATHS}/x86_64-linux/lib64)
  endif()
endif()

target_link_options(
  intf_pub
  INTERFACE
  $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-pie>
  $<$<CONFIG:Release>:-s>
  -Wl,-z,relro
  -Wl,-z,now
  -Wl,-z,noexecstack)
