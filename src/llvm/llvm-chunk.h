// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_LLVM_CHUNK_H_
#define V8_LLVM_CHUNK_H_

#include "../handles.h"
#include "../lithium.h"
#include "llvm-headers.h"

#include <memory>

namespace v8 {
namespace internal {

class LLVMGranularity FINAL {
 public:
  static LLVMGranularity& getInstance() {
    static LLVMGranularity instance;
    return instance;
  }
  LLVMContext& context() const { return context_; }
  llvm::Module* module() { return module_; } // FIXME!!

 private:
  LLVMContext context_;
  llvm::Module* module_;

  LLVMGranularity()
    : context_(),
      module_(new llvm::Module("v8-llvm", context_)) {}


  DISALLOW_COPY_AND_ASSIGN(LLVMGranularity);
};

class LLVMChunk FINAL : public LowChunk {
 public:
  virtual ~LLVMChunk() {}
  LLVMChunk(CompilationInfo* info, HGraph* graph)
    : LowChunk(info, graph) {}

  static LLVMChunk* NewChunk(HGraph *graph);

  Handle<Code> Codegen() override;
};

class LLVMChunkBuilder FINAL : public LowChunkBuilderBase {
 public:
  // TODO(llvm): add LLVMAllocator param to constructor (sibling of LAllocator)
  LLVMChunkBuilder(CompilationInfo* info, HGraph* graph)
      : LowChunkBuilderBase(info, graph),
        current_instruction_(nullptr),
        current_block_(nullptr),
        next_block_(nullptr) {}
  ~LLVMChunkBuilder() {}

  LLVMChunk* chunk() const { return static_cast<LLVMChunk*>(chunk_); };
  LLVMChunk* Build() override;

 private:
  void DoBasicBlock(HBasicBlock* block, HBasicBlock* next_block);
  void VisitInstruction(HInstruction* current);

  HBasicBlock* current_block_;
  HBasicBlock* next_block_;
  // TODO(llvm): probably pull these up to LowChunkBuilderBase
  HInstruction* current_instruction_;
  HBasicBlock* current_block_;
  HBasicBlock* next_block_;
};


}  // namespace internal
}  // namespace v8
#endif  // V8_LLVM_CHUNK_H_
