// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "llvm-passes.h"

//#include "src/globals.h"
//#include "src/list-inl.h"

namespace v8 {
namespace internal {

bool NormalizePhisPass::runOnFunction(Function &F) {
  llvm::outs() << "HELLO\n";
  return false;
}

void NormalizePhisPass::getAnalysisUsage(llvm::AnalysisUsage& info) {
  info.addRequired<llvm::DominatorTree>();
}

} }  // namespace v8::internal
