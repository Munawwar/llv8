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

  std::unique_ptr<llvm::Module> CreateModule(std::string name = "") {
    if ("" == name) {
      name = GenerateName();
    }
    return llvm::make_unique<llvm::Module>(name, context_);
  }

  llvm::Function* AddFunction(std::unique_ptr<llvm::Module> module) {
    if (!engine_) {
      engine_ = llvm::EngineBuilder(std::move(module))
        .setEngineKind(llvm::EngineKind::JIT)
        .setOptLevel(llvm::CodeGenOpt::Aggressive)
        .create(); // TODO(llvm): add options
      CHECK(engine_);
    } else {
      engine_->addModule(std::move(module));
    }
  }
 private:
  LLVMContext context_;
  std::unique_ptr<llvm::Module> module_;
  std::unique_ptr<llvm::ExecutionEngine> engine_; // FIXME(llvm): is it unique? //probably it is shared...
  int count_;

  LLVMGranularity()
    : context_(),
      module_(llvm::make_unique<llvm::Module>("v8-llvm", context_)),
      engine_(nullptr),
      count_(0) {}

  std::string GenerateName() {
    return std::to_string(count_++);
  }

  DISALLOW_COPY_AND_ASSIGN(LLVMGranularity);
};

class LLVMChunk FINAL : public LowChunk {
 public:
  virtual ~LLVMChunk() {}
  LLVMChunk(CompilationInfo* info, HGraph* graph)
    : LowChunk(info, graph) {
//    module_ = LLVMGranularity::getInstance().CreateModule();
  }

  static LLVMChunk* NewChunk(HGraph *graph);

  Handle<Code> Codegen() override;
 private:
  // TODO(llvm): make module_ a unique_ptr if possible
  // after the module is constructed, ownership is transfered to the ExecutionEngine
  // (via a call to AddModule())
//  std::shared_ptr<llvm::Module> module_;
  // now all functions will be in one module
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
