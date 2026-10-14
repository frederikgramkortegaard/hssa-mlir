# Here we create a single aggregate shared library with the parts of the CAPI
# that we want to bundle together. Then we link a simple C executable
# against it to demonstrate that it does have the fully self contained
# core MLIR library and our own hssa dialect.
add_mlir_aggregate(HssaCAPITestLib
  SHARED
  EMBED_LIBS
  MLIRCAPIIR
  MLIRCAPIArith
  HssaCAPI
)

add_llvm_executable(hssa-capi-test
  hssa-capi-test.c
)
llvm_update_compile_flags(hssa-capi-test)
target_link_libraries(hssa-capi-test
  PRIVATE HssaCAPITestLib)
