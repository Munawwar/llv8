// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "llvm-passes.h"

#include "src/base/macros.h"
//#include "src/globals.h"
//#include "src/list-inl.h"

namespace v8 {
namespace internal {

char NormalizePhisPass::ID = 0;

static llvm::RegisterPass<NormalizePhisPass> register_normalize_phis(
    "normalizePhis", "Normalize phis");

bool NormalizePhisPass::runOnFunction(llvm::Function &F) {
  llvm::outs() << "HELLO\n";
//  llvm::DominatorTree& dom_tree = getAnalysis<llvm::DominatorTree>();
  return false;
}

void NormalizePhisPass::getAnalysisUsage(llvm::AnalysisUsage& info) const {
  info.addRequired<llvm::DominatorTreeWrapperPass>();
//  info.addPreserved<llvm::DominatorTreeWrapperPass>();
}

} }  // namespace v8::internal
