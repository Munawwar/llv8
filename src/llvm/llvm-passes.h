// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_LLVM_PASSES_H_
#define V8_LLVM_PASSES_H_

#include "llvm-headers.h"


namespace v8 {
namespace internal {

#define LLV8_CALL_ONCE_INITIALIZATION(function) \
  static volatile llvm::sys::cas_flag initialized = 0; \
  llvm::sys::cas_flag old_val = llvm::sys::CompareAndSwap(&initialized, 1, 0); \
  if (old_val == 0) { \
    function(Registry); \
    llvm::sys::MemoryFence(); \
    TsanIgnoreWritesBegin(); \
    TsanHappensBefore(&initialized); \
    initialized = 2; \
    TsanIgnoreWritesEnd(); \
  } else { \
    llvm::sys::cas_flag tmp = initialized; \
    llvm::sys::MemoryFence(); \
    while (tmp != 2) { \
      tmp = initialized; \
      llvm::sys::MemoryFence(); \
    } \
  } \
  TsanHappensAfter(&initialized);

#define LLV8_INITIALIZE_PASS_BEGIN(passName, arg, name, cfg, analysis) \
  static void* initialize##passName##PassOnce(llvm::PassRegistry &Registry) {

#define LLV8_INITIALIZE_PASS_DEPENDENCY(depName) \
    initialize##depName##Pass(Registry);
#define INITIALIZE_AG_DEPENDENCY(depName) \
    initialize##depName##AnalysisGroup(Registry);

#define LLV8_INITIALIZE_PASS_END(passName, arg, name, cfg, analysis) \
    llvm::PassInfo *PI = new llvm::PassInfo(name, arg, & passName ::ID, \
      llvm::PassInfo::NormalCtor_t(llvm::callDefaultCtor< passName >), cfg, analysis); \
    Registry.registerPass(*PI, true); \
    return PI; \
  } \
  void initialize##passName##Pass(llvm::PassRegistry &Registry) { \
    LLV8_CALL_ONCE_INITIALIZATION(initialize##passName##PassOnce) \
  }

//void initializeNormalizePhisPassPass(llvm::PassRegistry &Registry);

//namespace {
// FunctionPasses may overload three virtual methods to do their work.
// All of these methods should return true if they modified the program,
// or false if they didn’t.
class NormalizePhisPass : public llvm::FunctionPass {
 public:
  NormalizePhisPass();
  bool runOnFunction(llvm::Function& function) override;
  void getAnalysisUsage(llvm::AnalysisUsage& analysis_usage) const override;

  bool doInitialization(llvm::Module& module) override { return false; };
  static char ID;
};
//}
//llvm::FunctionPass* llvm::createNormalizePhis() { return new Norm(); }

} }  // namespace v8::internal
#endif  // V8_LLVM_PASSES_H_
