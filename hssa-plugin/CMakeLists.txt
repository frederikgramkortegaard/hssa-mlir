add_llvm_library(HssaPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        hssa-plugin.cpp

        DEPENDS
        MLIRHssa
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRHssa
        )
target_include_directories(
        HssaPlugin
        PRIVATE
        "${HSSA_BINARY_DIR}/include"
)
