// RUN: hssa-opt --hssa-verify-reversible --verify-diagnostics %s

// expected-error @+1 {{function is marked hssa.reversible but is classified (D,N)}}
func.func @leaky(%x: i32) attributes {hssa.reversible} {
  hssa.discard %x : i32
  return
}
