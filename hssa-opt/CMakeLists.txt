set(LIBS
    MLIRArithDialect
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRHssa
  )
add_llvm_executable(hssa-opt hssa-opt.cpp)

llvm_update_compile_flags(hssa-opt)
target_link_libraries(hssa-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(hssa-opt)
