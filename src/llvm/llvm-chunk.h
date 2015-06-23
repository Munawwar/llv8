// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_LLVM_CHUNK_H_
#define V8_LLVM_CHUNK_H_

#include "llvm-headers.h"

// TODO(llvm): get rid of the ugly ".."
#include "../hydrogen.h"
#include "../hydrogen-instructions.h"
#include "../handles.h"
#include "../x64/lithium-codegen-x64.h"
#include "../lithium.h"
#include "llvm-stackmaps.h"
#include "mcjit-memory-manager.h"

#include <memory>

namespace v8 {
namespace internal {

// TODO(llvm): move this class to a separate file. Or, better, 2 files
class LLVMGranularity FINAL {
 public:
  static LLVMGranularity& getInstance() {
    static LLVMGranularity instance;
    return instance;
  }

  // TODO(llvm):
//  ~LLVMGranularity() {
//    llvm::llvm_shutdown();
//  }

  LLVMContext& context() { return context_; }
  MCJITMemoryManager* memory_manager_ref() { return memory_manager_ref_; }

  std::unique_ptr<llvm::Module> CreateModule(std::string name = "") {
    if ("" == name) {
      name = GenerateName();
    }
    return llvm::make_unique<llvm::Module>(name, context_);
  }

  void AddModule(std::unique_ptr<llvm::Module> module) {
    if (!engine_) {
      std::unique_ptr<MCJITMemoryManager> manager =
          MCJITMemoryManager::Create();
      memory_manager_ref_ = manager.get(); // non-owning!
      llvm::TargetOptions options;
      // rbp based frame so the runtime can walk the stack as before
      options.NoFramePointerElim = true;
      llvm::ExecutionEngine* raw = llvm::EngineBuilder(std::move(module))
        .setMCJITMemoryManager(std::move(manager))
        .setErrorStr(&err_str_)
        .setEngineKind(llvm::EngineKind::JIT)
        .setTargetOptions(options)
        .setOptLevel(llvm::CodeGenOpt::Aggressive) // backend opt level
        .create();
      engine_ = std::unique_ptr<llvm::ExecutionEngine>(raw);
      CHECK(engine_);
    } else {
      engine_->addModule(std::move(module));
    }
    // Finalize each time after adding a new module
    // (assuming the added module is constructed and won't change)
    engine_->finalizeObject();
  }

  void OptimizeFunciton(llvm::Module* module, llvm::Function* function) {
    // TODO(llvm): 1). Instead of using -O3 optimizations, add the
    // appropriate passes manually
    // TODO(llvm): 2). I didn't manage to make use of new PassManagers.
    // llvm::legacy:: things should probably be removed with time.
    // But for now even the llvm optimizer (llvm/tools/opt/opt.cpp) uses them.
    // TODO(llvm): 3). (Probably could be resolved easily when 2. is done)
    // for now we set up the passes for each module (and each function).
    // It would be much nicer if we could just set the passes once
    // and then in OptimizeFunciton() and OptimizeModule() simply run them.
    llvm::legacy::FunctionPassManager pass_manager(module);
    pass_manager_builder_.populateFunctionPassManager(pass_manager);
    pass_manager.doInitialization();
    pass_manager.run(*function);
    pass_manager.doFinalization();
  }

  void OptimizeModule(llvm::Module* module) {
    // TODO(llvm): see OptimizeFunciton TODOs (ditto)
    llvm::legacy::PassManager pass_manager;
    pass_manager_builder_.populateModulePassManager(pass_manager);
    pass_manager.run(*module);
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
  llvm::PassManagerBuilder pass_manager_builder_;
  std::unique_ptr<llvm::ExecutionEngine> engine_;
  int count_;
  MCJITMemoryManager* memory_manager_ref_; // non-owning ptr
  std::string err_str_;

  LLVMGranularity()
      : context_(),
        pass_manager_builder_(),
        engine_(nullptr),
        count_(0),
        memory_manager_ref_(nullptr),
        err_str_() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    pass_manager_builder_.OptLevel = 3; // -O3
  }

  std::string GenerateName() {
    return std::to_string(count_++);
  }

  DISALLOW_COPY_AND_ASSIGN(LLVMGranularity);
};

class LLVMEnvironment FINAL:  public ZoneObject {
 public:
  LLVMEnvironment(Handle<JSFunction> closure,
               FrameType frame_type,
               BailoutId ast_id,
               int parameter_count,
               int argument_count,
               int value_count,
               LLVMEnvironment* outer,
               HEnterInlined* entry,
               Zone* zone)
      : closure_(closure),
        frame_type_(frame_type),
        arguments_stack_height_(argument_count),
        deoptimization_index_(Safepoint::kNoDeoptimizationIndex),
        ast_id_(ast_id),
        translation_size_(value_count),
        parameter_count_(parameter_count),
        pc_offset_(-1),
        values_(value_count, zone),
        is_tagged_(value_count, zone),
        is_uint32_(value_count, zone),
        object_mapping_(0, zone),
        outer_(outer),
        entry_(entry),
        zone_(zone),
        has_been_used_(false) { }

  Handle<JSFunction> closure() const { return closure_; }
  FrameType frame_type() const { return frame_type_; }
  int arguments_stack_height() const { return arguments_stack_height_; }
  LLVMEnvironment* outer() const { return outer_; }
  const ZoneList<llvm::Value*>* values() const { return &values_; }
  BailoutId ast_id() const { return ast_id_; }
  int translation_size() const { return translation_size_; }
  int parameter_count() const { return parameter_count_; }
  Zone* zone() const { return zone_; }

  // Marker value indicating a de-materialized object.
  static llvm::Value* materialization_marker() { return nullptr; }

  void AddValue(llvm::Value* value,
                Representation representation,
                bool is_uint32) {
    values_.Add(value, zone());
    if (representation.IsSmiOrTagged()) {
      DCHECK(!is_uint32);
      is_tagged_.Add(values_.length() - 1, zone());
    }

    if (is_uint32) {
      is_uint32_.Add(values_.length() - 1, zone());
    }
  }

  bool HasTaggedValueAt(int index) const {
    return is_tagged_.Contains(index);
  }

  bool HasUint32ValueAt(int index) const {
    return is_uint32_.Contains(index);
  }

  ~LLVMEnvironment() { // FIXME(llvm): remove unused fields.
//    USE(closure_);
//    USE(frame_type_);
//    USE(arguments_stack_height_);
    USE(deoptimization_index_);
//    USE(ast_id_);
//    USE(translation_size_);
//    USE(parameter_count_);
    USE(pc_offset_);
    //USE(object_mapping_);
//    USE(outer_);
    USE(entry_);
    USE(has_been_used_);
  }

 private:
  Handle<JSFunction> closure_;
  FrameType frame_type_;
  int arguments_stack_height_;
  int deoptimization_index_;
  BailoutId ast_id_;
  int translation_size_;
  int parameter_count_;
  int pc_offset_;

  // Value array: [parameters] [locals] [expression stack] [de-materialized].
  //              |>--------- translation_size ---------<|
  ZoneList<llvm::Value*> values_;
  GrowableBitVector is_tagged_;
  GrowableBitVector is_uint32_;

  // Map with encoded information about materialization_marker operands.
  ZoneList<uint32_t> object_mapping_;

  LLVMEnvironment* outer_;
  HEnterInlined* entry_;
  Zone* zone_;
  bool has_been_used_;
};

class LLVMDeoptData {
 public:
  LLVMDeoptData(Zone* zone)
     : deoptimizations_(4, zone),
       translations_(zone),
       deoptimization_literals_(8, zone),
       zone_(zone) {}

  void Add(LLVMEnvironment* environment) {
    deoptimizations_.Add(environment, environment->zone());
  }

  ZoneList<LLVMEnvironment*>& deoptimizations() { return deoptimizations_; }
  TranslationBuffer& translations() { return translations_; }
  ZoneList<Handle<Object> >& deoptimization_literals() {
    return deoptimization_literals_;
  }

  int DeoptCount() { return deoptimizations_.length(); }

  int DefineDeoptimizationLiteral(Handle<Object> literal);

 private:
  ZoneList<LLVMEnvironment*> deoptimizations_;
  TranslationBuffer translations_;
  ZoneList<Handle<Object> > deoptimization_literals_;

  Zone* zone_;
};

class LLVMChunk FINAL : public LowChunk {
 public:
  virtual ~LLVMChunk();
  LLVMChunk(CompilationInfo* info, HGraph* graph)
    : LowChunk(info, graph),
      llvm_function_id_(-1),
      deopt_data_(nullptr) {}

  static LLVMChunk* NewChunk(HGraph *graph);

  Handle<Code> Codegen() override;

  void set_llvm_function_id(int id) { llvm_function_id_ = id; }
  int llvm_function_id() { return llvm_function_id_; }

  void set_deopt_data(std::unique_ptr<LLVMDeoptData> deopt_data) {
    deopt_data_ = std::move(deopt_data);
  }
 private:
  static const int kStackSlotSize = kPointerSize;
  static const int kPhonySpillCount = 3; // rbp, rsi, rdi

  void SetUpDeoptimizationData(Handle<Code> code);
  // Returns translation index of the newly generated translation
  int WriteTranslationFor(LLVMEnvironment* env, StackMaps::Record& stackmap);
  void WriteTranslation(LLVMEnvironment* environment,
                        StackMaps::Record& stackmap,
                        Translation* translation);
  void AddToTranslation(LLVMEnvironment* environment,
                        Translation* translation,
                        llvm::Value* op, //change
                        StackMaps::Location& location,
                        bool is_tagged,
                        bool is_uint32,
                        int* object_index_pointer,
                        int* dematerialized_index_pointer);

  int llvm_function_id_;
  // Ownership gets transfered from LLVMChunkBuilder
  std::unique_ptr<LLVMDeoptData> deopt_data_;
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
        llvm_ir_builder_(nullptr),
        deopt_data_(llvm::make_unique<LLVMDeoptData>(info->zone())) {}
  ~LLVMChunkBuilder() {}

  LLVMChunk* chunk() const { return static_cast<LLVMChunk*>(chunk_); };
  LLVMChunkBuilder& Build();
  // LLVM requires that each phi input's label be a basic block
  // immediately preceding the given BB.
  // Hydrogen does not impose such a constraint.
  // For that reason our phis are not LLVM-compliant right after phi resolution.
  LLVMChunkBuilder& NormalizePhis();
  LLVMChunkBuilder& Optimize(); // invoke llvm transformation passes for the function
  LLVMChunk* Create();

  LLVMEnvironment* AssignEnvironment();
  LLVMEnvironment* CreateEnvironment(
      HEnvironment* hydrogen_env, int* argument_index_accumulator,
      ZoneList<HValue*>* objects_to_materialize);

  void DeoptimizeIf(llvm::Value* compare, HBasicBlock* block);

  // Declare methods that deal with the individual node types.
#define DECLARE_DO(type) void Do##type(H##type* node);
  HYDROGEN_CONCRETE_INSTRUCTION_LIST(DECLARE_DO)
#undef DECLARE_DO

 private:
  static const int kSmiShift = kSmiTagSize + kSmiShiftSize;

  static llvm::CmpInst::Predicate TokenToPredicate(Token::Value op,
                                                   bool is_unsigned);

  void DoBasicBlock(HBasicBlock* block, HBasicBlock* next_block);
  void VisitInstruction(HInstruction* current);
  void DoPhi(HPhi* phi);
  void ResolvePhis();
  void ResolvePhis(HBasicBlock* block);
  // if the llvm counterpart of the block does not exist, create it
  llvm::BasicBlock* Use(HBasicBlock* block);
  llvm::Value* Use(HValue* value);
  llvm::Value* SmiToInteger32(HValue* value);
  llvm::Value* Integer32ToSmi(HValue* value);
  llvm::Value* Integer32ToSmi(llvm::Value* value);
  // Is the value (not) a smi?
  llvm::Value* SmiCheck(HValue* value, bool negate = false);
  llvm::Value* CallVoid(Address target);
  llvm::Value* CallForResult(Address target);
  // Allocate a heap number in new space with undefined value. Returns
  // tagged pointer in result register, or jumps to gc_required if new
  // space is full. // FIXME(llvm): the comment
  llvm::Value* AllocateHeapNumber();
  llvm::Value* CallRuntime(Runtime::FunctionId id);
  llvm::Value* CallRuntimeFromDeferred(Runtime::FunctionId id, llvm::Value* context, std::vector<llvm::Value*>);
  llvm::Value* GetContext();
  llvm::Value* CompareRoot(HValue* val, HChange* instr);
  void ChangeTaggedToDouble(HValue* val, HChange* instr);
  void ChangeDoubleToTagged(HValue* val, HChange* instr);

  void Retry(BailoutReason reason);
  void AddStabilityDependency(Handle<Map> map);

  // TODO(llvm): probably pull these up to LowChunkBuilderBase
  HInstruction* current_instruction_;
  HBasicBlock* current_block_;
  HBasicBlock* next_block_;
  // module_ ownership is later passed to the execution engine (MCJIT)
  std::unique_ptr<llvm::Module> module_;
  // Non-owning pointer to the function inside llvm module.
  // Not to be used for fetching the actual native code,
  // since the corresponding methods are deprecated.
  llvm::Function* function_;
  std::unique_ptr<llvm::IRBuilder<>> llvm_ir_builder_;
  std::unique_ptr<LLVMDeoptData> deopt_data_;
};

}  // namespace internal
}  // namespace v8
#endif  // V8_LLVM_CHUNK_H_
