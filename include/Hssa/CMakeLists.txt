add_mlir_dialect(HssaOps hssa)
add_mlir_doc(HssaDialect HssaDialect Hssa/ -gen-dialect-doc)
add_mlir_doc(HssaOps HssaOps Hssa/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS HssaPasses.td)
mlir_tablegen(HssaPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRHssaPassesIncGen)


set(LLVM_TARGET_DEFINITIONS HssaOpInterfaces.td)
mlir_tablegen(HssaOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(HssaOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRHssaOpInterfacesIncGen)
