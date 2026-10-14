# Convenience wrapper around the real build system (CMake/ninja).
# `make`       - build, run all tests verbosely, show the passes, run the pipeline.
# `make test`  - build + verbose tests only.
# `make demo`  - print what each pass does to the sample files.
# `make run`   - compile examples/pipeline.mlir to native code via LLVM and execute it.

BUILD       := build
HSSA_CORPUS ?= third_party/hssa-reference/programs/hssa
LLVM_BIN := /opt/homebrew/opt/llvm/bin
MLIR_INC := /opt/homebrew/opt/llvm/include
TBLGEN   := $(LLVM_BIN)/mlir-tblgen

.PHONY: all test demo run demo-corpus docs diff diff-ref diff-live diff-oo

all: test demo run

test:
	ninja -C $(BUILD)
	lit -v $(BUILD)/test

# Auto-generate the language reference from the dialect's TableGen (the single
# source of truth — these cannot drift from the implementation).
docs:
	@mkdir -p docs/spec
	@$(TBLGEN) -gen-op-doc      -I include -I $(MLIR_INC) include/Hssa/HssaOps.td     > docs/spec/ops.md
	@$(TBLGEN) -gen-pass-doc    -I include -I $(MLIR_INC) include/Hssa/HssaPasses.td  > docs/spec/passes.md
	@$(TBLGEN) -gen-dialect-doc -I include -I $(MLIR_INC) include/Hssa/HssaDialect.td > docs/spec/dialect.md
	@echo "generated docs/spec/{ops,passes,dialect}.md"

# Semantic self-check: compile each program to native and run it. Every program
# returns 0 iff it computed the expected result, so a nonzero exit is a semantic
# bug (wrong lowering, wrong inverse). Uses reversibility as its own oracle
# where it can (forward then inverse must recover the input). No external
# reference interpreter needed; see test/Exec.
diff:
	@ninja -C $(BUILD) >/dev/null
	@fail=0; for f in test/Exec/*.mlir; do \
	  if $(BUILD)/bin/hssa-opt --hssa-resolve-uncalls --canonicalize --convert-hssa-to-arith $$f 2>/dev/null \
	     | $(LLVM_BIN)/mlir-opt --convert-to-llvm 2>/dev/null \
	     | $(LLVM_BIN)/mlir-translate --mlir-to-llvmir 2>/dev/null \
	     | $(LLVM_BIN)/lli; then echo "ok   $$(basename $$f)"; \
	  else echo "FAIL $$(basename $$f) (exit $$?)"; fail=1; fi; done; \
	  exit $$fail

# Differential test vs the REFERENCE's own expected values: the reference
# .test programs assert results via literal-output patterns (reference-authored).
# We run them through our pipeline; exit 0 means our semantics agree.
diff-ref:
	@ninja -C $(BUILD) >/dev/null
	@HSSA_CORPUS="$(HSSA_CORPUS)" LLVM_BIN="$(LLVM_BIN)" ./scripts/difftest-ref.sh

demo:
	@echo "==== round trip: hssa-opt test/Hssa/ops.mlir ===="
	@$(BUILD)/bin/hssa-opt test/Hssa/ops.mlir
	@echo
	@echo "==== --hssa-invert: test/Hssa/Transforms/invert.mlir ===="
	@$(BUILD)/bin/hssa-opt --hssa-invert test/Hssa/Transforms/invert.mlir
	@echo
	@echo "==== --hssa-invert --hssa-invert (involution, should match the original) ===="
	@$(BUILD)/bin/hssa-opt --hssa-invert --hssa-invert test/Hssa/Transforms/invert.mlir
	@echo
	@echo "==== --hssa-resolve-uncalls: examples/pipeline.mlir ===="
	@$(BUILD)/bin/hssa-opt --hssa-resolve-uncalls examples/pipeline.mlir
	@echo
	@echo "==== + --canonicalize --convert-hssa-to-arith (stock MLIR canon, then ready for LLVM) ===="
	@$(BUILD)/bin/hssa-opt --hssa-resolve-uncalls --canonicalize --convert-hssa-to-arith examples/pipeline.mlir
	@echo
	@echo "==== --import-hssa: examples/basic.hssa (textual HSSA in) ===="
	@$(BUILD)/bin/hssa-translate --import-hssa examples/basic.hssa
	@echo
	@echo "==== imported text -> resolve -> arith ===="
	@$(BUILD)/bin/hssa-translate --import-hssa examples/basic.hssa \
	  | $(BUILD)/bin/hssa-opt --hssa-resolve-uncalls --canonicalize --convert-hssa-to-arith

run:
	@echo "==== examples/pipeline.mlir -> native (expect 5: f maps 5 to 14, uncall maps it back) ===="
	@set -o pipefail; $(BUILD)/bin/hssa-opt --hssa-resolve-uncalls --canonicalize --convert-hssa-to-arith examples/pipeline.mlir \
	  | $(LLVM_BIN)/mlir-opt --convert-to-llvm \
	  | $(LLVM_BIN)/mlir-translate --mlir-to-llvmir > $(BUILD)/pipeline.ll
	@$(LLVM_BIN)/lli $(BUILD)/pipeline.ll; echo "result: $$?"

demo-corpus:
	@echo "==== import a real corpus program (fibpair) ===="
	@$(BUILD)/bin/hssa-translate --import-hssa $(HSSA_CORPUS)/fibpair.hssa
	@echo
	@echo "==== and derive its inverse ===="
	@$(BUILD)/bin/hssa-translate --import-hssa $(HSSA_CORPUS)/fibpair.hssa | $(BUILD)/bin/hssa-opt --hssa-invert

# LIVE differential vs the reference interpreter (needs the toolkit jar built).
diff-live:
	@ninja -C $(BUILD) >/dev/null
	@LLVM_BIN="$(LLVM_BIN)" ./scripts/difftest-live.sh

# LIVE differential for OBJECT programs (heap-valued main): assert the native
# final heap matches the interpreter's, slot by slot. Validates that a real OO
# program COMPUTES the right answer, not just that it compiles.
diff-oo:
	@ninja -C $(BUILD) >/dev/null
	@LLVM_BIN="$(LLVM_BIN)" ./scripts/difftest-oo.sh


# Property testing / fuzzing (scripts/fuzz.sh, scripts/fuzz-gen.sh)
fuzz:
	scripts/fuzz.sh metamorphic

fuzz-crash:
	scripts/fuzz.sh crash 2000
