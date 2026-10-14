llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(HSSA_TEST_DEPENDS
        FileCheck count not
        hssa-opt
        hssa-translate
        )

# The CAPI test uses add_mlir_aggregate, which only works when MLIR was built
# with MLIR_INSTALL_AGGREGATE_OBJECTS (the aggregate object targets exist).
# Homebrew's MLIR provides them; apt.llvm.org's packages do not. Detect the
# object targets and only build the CAPI test when they are present.
if(TARGET obj.MLIRCAPIIR AND TARGET obj.MLIRCAPIArith)
  set(HSSA_HAVE_CAPI_AGGREGATE TRUE)
else()
  set(HSSA_HAVE_CAPI_AGGREGATE FALSE)
  message(STATUS "HSSA: skipping CAPI test; MLIR aggregate object targets "
                 "(obj.MLIRCAPIIR, obj.MLIRCAPIArith) are unavailable. This "
                 "MLIR was built without MLIR_INSTALL_AGGREGATE_OBJECTS.")
endif()

if(HSSA_HAVE_CAPI_AGGREGATE)
  list(APPEND HSSA_TEST_DEPENDS hssa-capi-test)
endif()
if(NOT WIN32)
  list(APPEND HSSA_TEST_DEPENDS HssaPlugin)
endif()
if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND HSSA_TEST_DEPENDS HssaPythonModules)
endif()

add_lit_testsuite(check-hssa "Running the hssa regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${HSSA_TEST_DEPENDS}
        )
set_target_properties(check-hssa PROPERTIES FOLDER "Tests")

add_lit_testsuites(HSSA ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${HSSA_TEST_DEPENDS})

if(HSSA_HAVE_CAPI_AGGREGATE)
  add_subdirectory(CAPI)
endif()
