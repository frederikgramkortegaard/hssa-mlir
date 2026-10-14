# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "HSSA"

# Use lit's internal shell; external-shell execution is deprecated as of LLVM 23.
config.test_format = lit.formats.ShTest(execute_external=False)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".hssa"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.hssa_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

config.hssa_tools_dir = os.path.join(config.hssa_obj_root, "bin")
config.hssa_libs_dir = os.path.join(config.hssa_obj_root, "lib")

config.substitutions.append(("%hssa_libs", config.hssa_libs_dir))

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.hssa_tools_dir, config.llvm_tools_dir]
tools = [
    "mlir-opt",
    "hssa-capi-test",
    "hssa-opt",
    "hssa-translate",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# The CAPI test is only built when MLIR provides the aggregate object targets
# (see test/CMakeLists.txt). Expose a feature so the test is skipped when the
# binary was not built.
for tool_dir in tool_dirs:
    if os.path.exists(os.path.join(tool_dir, "hssa-capi-test")):
        config.available_features.add("capi-test")
        break

python_path = [os.path.join(config.mlir_obj_dir, "python_packages", "hssa")]
if "PYTHONPATH" in os.environ:
    python_path += [os.environ["PYTHONPATH"]]

llvm_config.with_environment("PYTHONPATH", python_path, append_path=True)
