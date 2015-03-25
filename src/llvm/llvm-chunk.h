// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_LLVM_CHUNK_H_
#define V8_LLVM_CHUNK_H_

#include "../hydrogen.h"
#include "../hydrogen-instructions.h"
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

  LLVMContext& context() { return context_; }

  std::unique_ptr<llvm::Module> CreateModule(std::string name = "") {
    if ("" == name) {
      name = GenerateName();
    }
    return llvm::make_unique<llvm::Module>(name, context_);
  }

  void AddModule(std::unique_ptr<llvm::Module> module) {
    llvm::outs() << "Adding module " << *(module.get());
    if (!engine_) {
      llvm::ExecutionEngine* raw = llvm::EngineBuilder(std::move(module))
        .setErrorStr(&err_str_)
        .setEngineKind(llvm::EngineKind::JIT)
        .setOptLevel(llvm::CodeGenOpt::Aggressive)
        .create(); // TODO(llvm): add options
      engine_ = std::unique_ptr<llvm::ExecutionEngine>(raw);
      engine_->DisableLazyCompilation(false); // FIXME(llvm): remove
      CHECK(engine_);
    } else {
      engine_->addModule(std::move(module));
    }
      // Finalize each time after adding a new module
      // (assuming the added module is constructed and won't change)
      engine_->finalizeObject();
  }

  uint64_t GetFunctionAddress(int id) {
    DCHECK(engine_);
    return engine_->getFunctionAddress(std::to_string(id));
  }

  void Err() {
    std::cerr << err_str_ << std::endl;
  }
 private:
  LLVMContext context_;
  std::unique_ptr<llvm::ExecutionEngine> engine_; // FIXME(llvm): is it unique? //probably it is shared...
  int count_;
  std::string err_str_;

  LLVMGranularity()
    : context_(),
      engine_(nullptr),
      count_(0),
      err_str_() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  }

  std::string GenerateName() {
    return std::to_string(count_++);
  }

  DISALLOW_COPY_AND_ASSIGN(LLVMGranularity);
};

class LLVMChunk FINAL : public LowChunk {
 public:
  virtual ~LLVMChunk();
  LLVMChunk(CompilationInfo* info, HGraph* graph)
    : LowChunk(info, graph),
      llvm_function_id_(-1) {}

  static LLVMChunk* NewChunk(HGraph *graph);

  Handle<Code> Codegen() override;

  void set_llvm_function_id(int id) { llvm_function_id_ = id; }
  int llvm_function_id() { return llvm_function_id_; }

 private:
  int llvm_function_id_;
};

class LLVMChunkBuilder FINAL : public LowChunkBuilderBase {
 public:
  LLVMChunkBuilder(CompilationInfo* info, HGraph* graph)
      : LowChunkBuilderBase(info, graph),
        current_instruction_(nullptr),
        current_block_(nullptr),
        next_block_(nullptr),
        module_(nullptr),
        function_(nullptr),
        llvm_ir_builder_(nullptr) {}
  ~LLVMChunkBuilder() {}

  LLVMChunk* chunk() const { return static_cast<LLVMChunk*>(chunk_); };
  LLVMChunk* Build() override;

  // Declare methods that deal with the individual node types.
#define DECLARE_DO(type) void Do##type(H##type* node);
  HYDROGEN_CONCRETE_INSTRUCTION_LIST(DECLARE_DO)
#undef DECLARE_DO

 private:
  void DoBasicBlock(HBasicBlock* block, HBasicBlock* next_block);
  void VisitInstruction(HInstruction* current);
  llvm::Value* Use(HValue* value);
  // if the llvm counterpart of the block does not exist, create it
  void CreateBasicBlock(HBasicBlock* block);

  // TODO(llvm): probably pull these up to LowChunkBuilderBase
  HInstruction* current_instruction_;
  HBasicBlock* current_block_;
  HBasicBlock* next_block_;
  std::unique_ptr<llvm::Module> module_;
  std::unique_ptr<llvm::Function> function_; // the essence
  std::unique_ptr<llvm::IRBuilder<>> llvm_ir_builder_;
};


}  // namespace internal
}  // namespace v8
#endif  // V8_LLVM_CHUNK_H_
