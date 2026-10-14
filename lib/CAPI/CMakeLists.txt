add_mlir_public_c_api_library(HssaCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRHssa
)
