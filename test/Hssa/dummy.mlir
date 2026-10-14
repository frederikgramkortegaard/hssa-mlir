// RUN: hssa-opt %s | hssa-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = hssa.id %{{.*}} : i32
        %res = hssa.id %0 : i32
        return
    }
}
