// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_LLVM_PASSES_H_
#define V8_LLVM_PASSES_H_

#include "llvm-headers.h"


namespace v8 {
namespace internal {


//namespace {
// FunctionPasses may overload three virtual methods to do their work.
// All of these methods should return true if they modified the program,
// or false if they didn’t.
class NormalizePhisPass : public llvm::FunctionPass {
 public:
  NormalizePhisPass()
      : llvm::FunctionPass(ID) {}
  bool runOnFunction(llvm::Function &F) override;
  void getAnalysisUsage(llvm::AnalysisUsage& info) const override;

  bool doInitialization(llvm::Module &M) override { return false; };
  static char ID;
};
//}
//llvm::FunctionPass* llvm::createNormalizePhis() { return new Norm(); }

} }  // namespace v8::internal
#endif  // V8_LLVM_PASSES_H_
