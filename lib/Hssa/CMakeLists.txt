add_mlir_dialect_library(MLIRHssa
        HssaDialect.cpp
        HssaTypes.cpp
        HssaOps.cpp
        HssaInvert.cpp
        HssaLinearity.cpp
        HssaToArith.cpp
        HssaImporter.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Hssa

        DEPENDS
        MLIRHssaOpsIncGen
        MLIRHssaPassesIncGen
        MLIRHssaOpInterfacesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
