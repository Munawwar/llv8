// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_LLVM_CHUNK_H_
#define V8_LLVM_CHUNK_H_

#include "../handles.h"
#include "../lithium.h"
#include "llvm-headers.h"

namespace v8 {
namespace internal {


class LLVMChunk FINAL : public LowChunk {
 public:
  virtual ~LLVMChunk() {}
  LLVMChunk(CompilationInfo* info, HGraph* graph)
    : LowChunk(info, graph) {}

  static LLVMChunk* NewChunk(HGraph *graph);

  const ZoneList<LInstruction*>* instructions() const { return &instructions_; }
  Handle<Code> Codegen() override;

 private:
  ZoneList<LLVMInstruction*> instructions_; // TODO(llvm): find out a suitable class for it in LLVM infrastructure
};

class LLVMChunkBuilder FINAL : public LowChunkBuilderBase {
 public:
  // TODO(llvm): add LLVMAllocator param to constructor (sibling of LAllocator)
  LLVMChunkBuilder(CompilationInfo* info, HGraph* graph)
      : LowChunkBuilderBase(info, graph) {}
  ~LLVMChunkBuilder() {}

  LLVMChunk* chunk() const { return static_cast<LLVMChunk*>(chunk_); };
  LLVMChunk* Build() override;

 private:
  void DoBasicBlock(HBasicBlock* block, HBasicBlock* next_block);
  void VisitInstruction(HInstruction* current); // TODO(llvm): implement

  HBasicBlock* current_block_;
  HBasicBlock* next_block_;
};


}  // namespace internal
}  // namespace v8
#endif  // V8_LLVM_CHUNK_H_
