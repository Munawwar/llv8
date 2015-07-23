// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_LLVM_CHUNK_H_
#define V8_LLVM_CHUNK_H_

#include "llvm-headers.h"

#include "src/hydrogen.h"
#include "src/hydrogen-instructions.h"
#include "src/handles.h"
#include "src/x64/lithium-codegen-x64.h"
#include "src/lithium.h"
#include "llvm-stackmaps.h"
#include "mcjit-memory-manager.h"

#include <memory>

namespace v8 {
namespace internal {

// ZoneObject is probably a better approach than the fancy
// C++11 smart pointers which I have been using all over the place.
// So TODO(llvm): more zone objects!
class LLVMRelocationData : public ZoneObject {
 public:
  union ExtendedInfo {
    bool cell_extended;
  };

  using RelocMap = std::map<uint64_t, std::pair<RelocInfo, ExtendedInfo>>;

  LLVMRelocationData()
     : reloc_map_(),
       is_transferred_(false) {}

  void Add(RelocInfo rinfo, ExtendedInfo ex_info) {
    DCHECK(!is_transferred_);
    reloc_map_[rinfo.data()] = std::make_pair(rinfo, ex_info);
  }

  RelocMap& reloc_map() {
    return reloc_map_;
  }

  void transfer() { is_transferred_ = true; }

 private:
  // TODO(llvm): re-think the design and probably use ZoneHashMap
  RelocMap reloc_map_;
  bool is_transferred_;
};

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

  void Disass(Address start, Address end) {
    auto pos = start;
    while (pos < end) {
      llvm::MCInst inst;
      uint64_t size;
      auto address = 0;

      llvm::MCDisassembler::DecodeStatus s = disasm_->getInstruction(
          inst /* out */, size /* out */, llvm::ArrayRef<uint8_t>(pos, end),
          address, llvm::nulls(), llvm::nulls());
      if (s == llvm::MCDisassembler::Fail) {
        std::cerr << "disassembler failed at "
            << reinterpret_cast<void*>(pos) << std::endl;
        break;
      }
      inst_printer_->printInst(&inst, llvm::errs(), "");
      llvm::errs() << "\n";
      pos += size;
    }
  }

  Vector<byte> Patch(Address, Address, LLVMRelocationData::RelocMap&);

  static const char* x64_target_triple;
 private:
  LLVMContext context_;
  llvm::PassManagerBuilder pass_manager_builder_;
  std::unique_ptr<llvm::ExecutionEngine> engine_;
  std::unique_ptr<llvm::MCDisassembler> disasm_;
  std::unique_ptr<llvm::MCInstPrinter> inst_printer_;
  std::unique_ptr<llvm::MCInstrInfo> mii_;
  int count_;
  MCJITMemoryManager* memory_manager_ref_; // non-owning ptr
  std::string err_str_;

  LLVMGranularity()
      : context_(),
        pass_manager_builder_(),
        engine_(nullptr),
        disasm_(nullptr),
        inst_printer_(nullptr),
        mii_(nullptr),
        count_(0),
        memory_manager_ref_(nullptr),
        err_str_() {
    llvm::InitializeNativeTarget();
    LLVMInitializeX86Disassembler();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    pass_manager_builder_.OptLevel = 3; // -O3
    SetUpDisassembler();
  }

  void SetUpDisassembler() {
    auto triple = x64_target_triple;
    std::string err;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple,
                                                                    err);
    DCHECK(target);
    std::unique_ptr<llvm::MCRegisterInfo> mri(target->createMCRegInfo(triple));
    DCHECK(mri);
    std::unique_ptr<llvm::MCAsmInfo> mai(target->createMCAsmInfo(*mri.get(),
                                                                 triple));
    DCHECK(mai);
    mii_ = std::unique_ptr<llvm::MCInstrInfo>(target->createMCInstrInfo());
    DCHECK(mii_);
    std::string feature_str;
    const llvm::StringRef cpu = "";
    std::unique_ptr<llvm::MCSubtargetInfo> sti(
        target->createMCSubtargetInfo(triple, cpu, feature_str));
    DCHECK(sti);
    auto intel_syntax = 1;
    inst_printer_ = std::unique_ptr<llvm::MCInstPrinter>(
        target->createMCInstPrinter(intel_syntax, *mai, *mii_, *mri, *sti));
    inst_printer_->setPrintImmHex(true);
    DCHECK(inst_printer_);
    std::unique_ptr<llvm::MCObjectFileInfo> mofi(new llvm::MCObjectFileInfo());
    DCHECK(mofi);
    llvm::MCContext mc_context(mai.get(), mri.get(), mofi.get());
    disasm_ = std::unique_ptr<llvm::MCDisassembler> (
        target->createMCDisassembler(*sti, mc_context));
    DCHECK(disasm_);
  }

  std::string GenerateName() {
    return std::to_string(count_++);
  }

  DISALLOW_COPY_AND_ASSIGN(LLVMGranularity);
};

struct Types FINAL : public AllStatic {
   static llvm::Type* tagged;
   static llvm::PointerType* ptr_tagged;

   static llvm::Type* i8;
   static llvm::Type* i32;
   static llvm::Type* i64;
   static llvm::Type* float64;

   static llvm::PointerType* ptr_i8;
   static llvm::PointerType* ptr_i32;
   static llvm::PointerType* ptr_i64;
   static llvm::PointerType* ptr_float64;

  static void Init(llvm::IRBuilder<>* ir_builder) {
    i8 = ir_builder->getInt8Ty();
    i32 = ir_builder->getInt32Ty();
    i64 = ir_builder->getInt64Ty();
    float64 = ir_builder->getDoubleTy();

    auto address_space = 0;
    ptr_i8 = ir_builder->getInt8PtrTy();
    ptr_i32 = llvm::PointerType::get(ir_builder->getInt32Ty(), address_space);
    ptr_i64 = llvm::PointerType::get(ir_builder->getInt64Ty(), address_space);
    ptr_float64 = llvm::PointerType::get(ir_builder->getDoubleTy(),
                                         address_space);
    // TODO(llvm): we should probably switch to i8*
    tagged = i64;
    ptr_tagged = ptr_i64;
  }
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
      reloc_data_(nullptr),
      deopt_data_(nullptr) {}

  static LLVMChunk* NewChunk(HGraph *graph);

  Handle<Code> Codegen() override;

  void set_llvm_function_id(int id) { llvm_function_id_ = id; }
  int llvm_function_id() { return llvm_function_id_; }

  void set_deopt_data(std::unique_ptr<LLVMDeoptData> deopt_data) {
    deopt_data_ = std::move(deopt_data);
  }
  void set_reloc_data(LLVMRelocationData* reloc_data) {
    reloc_data_ = reloc_data;
    reloc_data->transfer();
  }
 private:
  static const int kStackSlotSize = kPointerSize;
  static const int kPhonySpillCount = 3; // rbp, rsi, rdi

  void SetUpDeoptimizationData(Handle<Code> code);
  Vector<byte> GetRelocationData(CodeDesc& code_desc);
  // Returns translation index of the newly generated translation
  int WriteTranslationFor(LLVMEnvironment* env,
                          StackMaps::Record& stackmap,
                          const StackMaps& stackmaps);
  void WriteTranslation(LLVMEnvironment* environment,
                        StackMaps::Record& stackmap,
                        Translation* translation,
                        const StackMaps& stackmaps);
  void AddToTranslation(LLVMEnvironment* environment,
                        Translation* translation,
                        llvm::Value* op, //change
                        StackMaps::Location& location,
                        const StackMaps& stackmaps,
                        bool is_tagged,
                        bool is_uint32,
                        int* object_index_pointer,
                        int* dematerialized_index_pointer);

  int llvm_function_id_;
  // Ownership gets transferred from LLVMChunkBuilder
  LLVMRelocationData* reloc_data_;
  // Ownership gets transferred from LLVMChunkBuilder
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
        deopt_data_(llvm::make_unique<LLVMDeoptData>(info->zone())),
        reloc_data_(nullptr),
        pending_pushed_args_(4, info->zone()),
        emit_debug_code_(FLAG_debug_code) {
    reloc_data_ = new(zone()) LLVMRelocationData();
  }
  ~LLVMChunkBuilder() {}

  LLVMChunk* chunk() const { return static_cast<LLVMChunk*>(chunk_); };
  void set_emit_degug_code(bool v) { emit_debug_code_ = v; }
  bool emit_debug_code() { return emit_debug_code_; }
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

  void DeoptimizeIf(llvm::Value* compare,
                    HBasicBlock* block,
                    bool negate = false,
                    llvm::BasicBlock* next_block = nullptr);

  // Declare methods that deal with the individual node types.
#define DECLARE_DO(type) void Do##type(H##type* node);
  HYDROGEN_CONCRETE_INSTRUCTION_LIST(DECLARE_DO)
#undef DECLARE_DO
  static const uintptr_t kExtFillingValue = 0xabcdbeef;

 private:
  static const int kSmiShift = kSmiTagSize + kSmiShiftSize;

  static llvm::CmpInst::Predicate TokenToPredicate(Token::Value op,
                                                   bool is_unsigned,
                                                   bool is_double = false);

  void DoBasicBlock(HBasicBlock* block, HBasicBlock* next_block);
  void VisitInstruction(HInstruction* current);
  void DoPhi(HPhi* phi);
  void ResolvePhis();
  void ResolvePhis(HBasicBlock* block);
  llvm::BasicBlock* NewBlock(const char* name);
  // if the llvm counterpart of the block does not exist, create it
  llvm::BasicBlock* Use(HBasicBlock* block);
  llvm::Value* Use(HValue* value);
  llvm::Value* SmiToInteger32(HValue* value);
  llvm::Value* Integer32ToSmi(HValue* value);
  llvm::Value* Integer32ToSmi(llvm::Value* value);
  // Is the value (not) a smi?
  llvm::Value* SmiCheck(llvm::Value* value, bool negate = false);
  void AssertSmi(llvm::Value* value, bool assert_not_smi = false);
  void AssertNotSmi(llvm::Value* value);
  void Assert(llvm::Value* condition);
  void IncrementCounter(StatsCounter* counter, int value);
  llvm::Value* CallVoid(Address target);
  llvm::Value* CallAddressForMathPow(Address target, llvm::CallingConv::ID calling_conv,
                           std::vector<llvm::Value*>& params);
  // This is intended to be a highly reusable method for calling stuff.
  llvm::Value* CallAddress(Address target, llvm::CallingConv::ID calling_conv,
                           std::vector<llvm::Value*>& params);
  llvm::Value* FieldOperand(llvm::Value* base, int offset);
  llvm::Value* LoadFieldOperand(llvm::Value* base, int offset);
  llvm::Value* ConstructAddress(llvm::Value* base, int offset);
  llvm::Value* MoveHeapObject(Handle<Object> obj);
  llvm::Value* Move(Handle<Object> object, RelocInfo::Mode rmode);
  llvm::Value* Compare(llvm::Value* lhs, llvm::Value* rhs);
  llvm::Value* Compare(llvm::Value* lhs, Handle<Object> rhs);
  llvm::Value* CompareMap(llvm::Value* object, Handle<Map> map);
  llvm::Value* CheckPageFlag(llvm::Value* object, int mask);
  // Allocate a heap number in new space with undefined value. Returns
  // tagged pointer in result register, or jumps to gc_required if new
  // space is full. // FIXME(llvm): the comment
  llvm::Value* AllocateHeapNumber();
  llvm::Value* CallRuntime(const Runtime::Function*);
  llvm::Value* CallRuntimeViaId(Runtime::FunctionId id);
  llvm::Value* CallRuntimeFromDeferred(Runtime::FunctionId id, llvm::Value* context, std::vector<llvm::Value*>);
  llvm::Value* GetContext();
  llvm::Value* LoadRoot(Heap::RootListIndex index);
  llvm::Value* CompareRoot(llvm::Value* val_address, Heap::RootListIndex index);
  llvm::Value* RecordRelocInfo(uint64_t intptr_value, RelocInfo::Mode rmode);
  void RecordWriteForMap(llvm::Value* object, llvm::Value* map);
  void ChangeTaggedToDouble(HValue* val, HChange* instr);
  void ChangeDoubleToTagged(HValue* val, HChange* instr);
  void ChangeTaggedToISlow(HValue* val, HChange* instr);
  void BranchTagged(HBranch* instr,
                    ToBooleanStub::Types expected,
                    llvm::BasicBlock* true_target,
                    llvm::BasicBlock* false_target);

  llvm::Type* GetLLVMType(Representation r);
  void DoDummyUse(HInstruction* instr);
  void DoStoreKeyedFixedArray(HStoreKeyed* value);
  void DoLoadKeyedFixedArray(HLoadKeyed* value);
  void DoLoadKeyedFixedDoubleArray(HLoadKeyed* value);
  void DoStoreKeyedFixedDoubleArray(HStoreKeyed* value); 
  void Retry(BailoutReason reason);
  void AddStabilityDependency(Handle<Map> map);
  void CallStackMap(int stackmap_id, llvm::Value* value);
  void CallStackMap(int stackmap_id, std::vector<llvm::Value*>& values);
  void DoMathAbs(HUnaryMathOperation* instr);
  void DoIntegerMathAbs(HUnaryMathOperation* instr);
  void DoMathPowHalf(HUnaryMathOperation* instr);
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
  LLVMRelocationData* reloc_data_;
  ZoneList<llvm::Value*> pending_pushed_args_;
  bool emit_debug_code_;
  enum ScaleFactor {
    times_1 = 0,
    times_2 = 1,
    times_4 = 2,
    times_8 = 3,
    times_int_size = times_4,
    times_half_pointer_size = times_2,
    times_pointer_size = times_4,
    times_twice_pointer_size = times_8
  };
};

}  // namespace internal
}  // namespace v8
#endif  // V8_LLVM_CHUNK_H_
