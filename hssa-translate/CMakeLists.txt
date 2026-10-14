set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(hssa-translate
  hssa-translate.cpp
  )
llvm_update_compile_flags(hssa-translate)
target_link_libraries(hssa-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRHssa
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTranslateLib
  MLIRSupport
  )

mlir_check_link_libraries(hssa-translate)
