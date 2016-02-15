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
#include "pass-rewrite-safepoints.h"
#include "mcjit-memory-manager.h"
#include "src/base/division-by-constant.h"

#include <memory>

namespace v8 {
namespace internal {

// TODO(llvm): Move to a separate file.
// Actually it should be elsewhere. And probably there is.
// So find it and remove this class.
class IntHelper : public AllStatic {
 public:
  // FIXME(llvm): consider int != int32
  static bool IsInt(uint64_t x) { return is_int32(x); }
  static int AsInt(uint64_t x) {
    DCHECK(IsInt(x));
    return static_cast<int>(x);
  }
  static bool IsInt(long x) { return is_int32(x); }
  static int AsInt(long x) {
    DCHECK(IsInt(x));
    return static_cast<int>(x);
  }
  static int AsUInt32(uint64_t x) {
    DCHECK(is_uint32(x));
    return static_cast<uint32_t>(x);
  }
  static int AsInt32(int64_t x) {
    DCHECK(is_int32(x));
    return static_cast<int32_t>(x);
  }
};

// ZoneObject is probably a better approach than the fancy
// C++11 smart pointers which I have been using all over the place.
// So TODO(llvm): more zone objects!
struct DeoptIdMap {
        int32_t patchpoint_id;
        int bailout_id;
};
class LLVMRelocationData : public ZoneObject {
 public:
  union ExtendedInfo {
    bool cell_extended;
  };

  using RelocMap = std::map<uint64_t, std::pair<RelocInfo, ExtendedInfo>>;

  LLVMRelocationData(Zone* zone)
     : reloc_map_(),
       last_patchpoint_id_(-1),
       is_reloc_(8, zone),
       is_reloc_with_nop_(8, zone),
       is_deopt_(8, zone),
       is_safepoint_(8, zone),
       is_transferred_(false),
       zone_(zone) {}

  void Add(RelocInfo rinfo, ExtendedInfo ex_info) {
    DCHECK(!is_transferred_);
    reloc_map_[rinfo.data()] = std::make_pair(rinfo, ex_info);
  }

  RelocMap& reloc_map() {
    return reloc_map_;
  }

  int32_t GetNextUnaccountedPatchpointId();
  // TODO(llvm): all of these methods have the same typo.
  int32_t GetNextDeoptPatchpointId();
  int32_t GetNextSafepointPatchpointId();
  int32_t GetNextRelocPatchpointId(bool is_safepoint = false);
  int32_t GetNextRelocNopPatchpointId(bool is_safepoint = false);
  int32_t GetNextDeoptRelocPatchpointId();
  int GetBailoutId(int32_t patchpoint_id);
  void SetBailoutId(int32_t patchpoint_id, int bailout_id);
  bool IsPatchpointIdDeopt(int32_t patchpoint_id);
  bool IsPatchpointIdSafepoint(int32_t patchpoint_id);
  bool IsPatchpointIdReloc(int32_t patchpoint_id);
  bool IsPatchpointIdRelocNop(int32_t patchpoint_id);

  void transfer() { is_transferred_ = true; }

  void DumpSafepointIds();

 private:
  // TODO(llvm): re-think the design and probably use ZoneHashMap
  RelocMap reloc_map_;
  int32_t last_patchpoint_id_;
  // FIXME(llvm): not totally sure those belong here:
  // Patchpoint ids belong to one (or more) of the following:
  GrowableBitVector is_reloc_;
  GrowableBitVector is_reloc_with_nop_;
  ZoneList<DeoptIdMap> is_deopt_;
  GrowableBitVector is_safepoint_;
  bool is_transferred_;
  Zone* zone_;
};

// TODO(llvm): move this class to a separate file. Or, better, 2 files
class LLVMGranularity final {
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
      std::vector<std::string> machine_attributes;
      SetMachineAttributes(machine_attributes);

      std::unique_ptr<MCJITMemoryManager> manager =
          MCJITMemoryManager::Create();
      memory_manager_ref_ = manager.get(); // non-owning!

      llvm::ExecutionEngine* raw = llvm::EngineBuilder(std::move(module))
        .setMCJITMemoryManager(std::move(manager))
        .setErrorStr(&err_str_)
        .setEngineKind(llvm::EngineKind::JIT)
        .setMAttrs(machine_attributes)
        .setMCPU("x86-64")
        .setRelocationModel(llvm::Reloc::PIC_) // position independent code
        // A good read on code models can be found here:
        // eli.thegreenplace.net/2012/01/03/understanding-the-x64-code-models
        // We use a modified Large code model, which uses rip-relative
        // addressing for jump tables.
        .setCodeModel(llvm::CodeModel::Large)
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

  // TODO(llvm): move to a separate file
  void Disass(Address start, Address end) {
    auto triple = x64_target_triple;
    std::string err;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple,
                                                                    err);
    DCHECK(target);
    std::unique_ptr<llvm::MCRegisterInfo> mri(target->createMCRegInfo(triple));
    DCHECK(mri);
    std::unique_ptr<llvm::MCAsmInfo> mai(target->createMCAsmInfo(*mri, triple));
    DCHECK(mai);
    std::unique_ptr<llvm::MCInstrInfo> mii(target->createMCInstrInfo());
    DCHECK(mii);
    std::string feature_str;
    const llvm::StringRef cpu = "";
    std::unique_ptr<llvm::MCSubtargetInfo> sti(
        target->createMCSubtargetInfo(triple, cpu, feature_str));
    DCHECK(sti);
    auto intel_syntax = 1;
    inst_printer_ = std::unique_ptr<llvm::MCInstPrinter>(
        target->createMCInstPrinter(llvm::Triple(llvm::Triple::normalize(triple)),
                                    intel_syntax, *mai, *mii, *mri));
    inst_printer_->setPrintImmHex(true);
    DCHECK(inst_printer_);
    llvm::MCContext mc_context(mai.get(), mri.get(), nullptr);
    std::unique_ptr<llvm::MCDisassembler> disasm(
        target->createMCDisassembler(*sti, mc_context));
    DCHECK(disasm);


    auto pos = start;
    while (pos < end) {
      llvm::MCInst inst;
      uint64_t size;
      auto address = 0;

      llvm::MCDisassembler::DecodeStatus s = disasm->getInstruction(
          inst /* out */, size /* out */, llvm::ArrayRef<uint8_t>(pos, end),
          address, llvm::nulls(), llvm::nulls());
      if (s == llvm::MCDisassembler::Fail) {
        std::cerr << "disassembler failed at "
            << reinterpret_cast<void*>(pos) << std::endl;
        break;
      }
      llvm::errs() << pos << "\t";
      inst_printer_->printInst(&inst, llvm::errs(), "", *sti);
      llvm::errs() << "\n";
      pos += size;
    }
  }

  int CallInstructionSizeAt(Address pc);
  std::vector<RelocInfo> Patch(Address, Address, LLVMRelocationData::RelocMap&);

  static const char* x64_target_triple;
 private:
  LLVMContext context_;
  llvm::PassManagerBuilder pass_manager_builder_;
  std::unique_ptr<llvm::ExecutionEngine> engine_;
  std::unique_ptr<llvm::MCInstPrinter> inst_printer_;
  int count_;
  MCJITMemoryManager* memory_manager_ref_; // non-owning ptr
  std::string err_str_;

  LLVMGranularity()
      : context_(),
        pass_manager_builder_(),
        engine_(nullptr),
        inst_printer_(nullptr),
        count_(0),
        memory_manager_ref_(nullptr),
        err_str_() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    llvm::InitializeNativeTargetDisassembler();
//    llvm::initializeCodeGen(*llvm::PassRegistry::getPassRegistry());
    pass_manager_builder_.OptLevel = 3; // -O3
  }

  std::string GenerateName() {
    return std::to_string(count_++);
  }

  void SetMachineAttributes(std::vector<std::string>& machine_attributes) {
    // TODO(llvm): add desired machine attributes. See llc -mattr=help
    // FIXME(llvm): for each attribute see, if the corresponding cpu
    // feature is supported.
    for (auto attr : {
      "sse","sse2","sse4.1","sse4.2",
      "sse4a", "ssse3", "aes", "avx", "avx2" }) {
      machine_attributes.push_back(attr);
    }
  }

  DISALLOW_COPY_AND_ASSIGN(LLVMGranularity);
};

struct Types final : public AllStatic {
   static llvm::Type* smi;
   static llvm::Type* ptr_smi;
   static llvm::Type* tagged;
   static llvm::PointerType* ptr_tagged;

   static llvm::Type* i8;
   static llvm::Type* i32;
   static llvm::Type* i64;
   static llvm::Type* float32;
   static llvm::Type* float64;

   static llvm::PointerType* ptr_i8;
   static llvm::PointerType* ptr_i16;
   static llvm::PointerType* ptr_i32;
   static llvm::PointerType* ptr_i64;
   static llvm::PointerType* ptr_float32;
   static llvm::PointerType* ptr_float64;

  static void Init(llvm::IRBuilder<>* ir_builder) {
    i8 = ir_builder->getInt8Ty();
    i32 = ir_builder->getInt32Ty();
    i64 = ir_builder->getInt64Ty();
    float32 = ir_builder->getFloatTy();
    float64 = ir_builder->getDoubleTy();

    auto address_space = 0;
    ptr_i8 = ir_builder->getInt8PtrTy();
    ptr_i16 = llvm::PointerType::get(ir_builder->getHalfTy(), address_space);
    ptr_i32 = llvm::PointerType::get(ir_builder->getInt32Ty(), address_space);
    ptr_i64 = llvm::PointerType::get(ir_builder->getInt64Ty(), address_space);
    ptr_float32 = llvm::PointerType::get(ir_builder->getFloatTy(), address_space);
    ptr_float64 = llvm::PointerType::get(ir_builder->getDoubleTy(),
                                         address_space);
    tagged = ptr_i8;
    ptr_tagged = ptr_i8->getPointerTo();
    smi = i64;
    ptr_smi = smi->getPointerTo();
  }
};

class LLVMEnvironment final : public ZoneObject {
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
        translation_index_(-1),
        ast_id_(ast_id),
        translation_size_(value_count),
        parameter_count_(parameter_count),
        pc_offset_(-1),
        values_(value_count, zone),
        is_tagged_(value_count, zone),
        is_uint32_(value_count, zone),
        is_double_(value_count, zone),
        object_mapping_(0, zone),
        outer_(outer),
        entry_(entry),
        zone_(zone),
        has_been_used_(false) { }

  Handle<JSFunction> closure() const { return closure_; }
  FrameType frame_type() const { return frame_type_; }
  int arguments_stack_height() const { return arguments_stack_height_; }
  LLVMEnvironment* outer() const { return outer_; }
  HEnterInlined* entry() { return entry_; }
  const ZoneList<llvm::Value*>* values() const { return &values_; }
  BailoutId ast_id() const { return ast_id_; }
  int translation_size() const { return translation_size_; }
  int parameter_count() const { return parameter_count_; }
  Zone* zone() const { return zone_; }

  // Marker value indicating a de-materialized object.
  static llvm::Value* materialization_marker() { return nullptr; }

  bool has_been_used() const { return has_been_used_; }
  void set_has_been_used() { has_been_used_ = true; }

  void AddValue(llvm::Value* value,
                Representation representation,
                bool is_uint32);

  bool HasTaggedValueAt(int index) const {
    return is_tagged_.Contains(index);
  }

  bool HasUint32ValueAt(int index) const {
    return is_uint32_.Contains(index);
  }

  bool HasDoubleValueAt(int index) const {
    return is_double_.Contains(index);
  }

  void Register(int deoptimization_index,
                int translation_index,
                int pc_offset) {
    DCHECK(!HasBeenRegistered());
    deoptimization_index_ = deoptimization_index;
    translation_index_ = translation_index;
    pc_offset_ = pc_offset;
  }
  bool HasBeenRegistered() const {
    return deoptimization_index_ != Safepoint::kNoDeoptimizationIndex;
  }

  ~LLVMEnvironment() { // FIXME(llvm): remove unused fields.
    USE(pc_offset_);
  }

 private:
  Handle<JSFunction> closure_;
  FrameType frame_type_;
  int arguments_stack_height_;
  int deoptimization_index_;
  int translation_index_;
  BailoutId ast_id_;
  int translation_size_;
  int parameter_count_;
  int pc_offset_;

  // Value array: [parameters] [locals] [expression stack] [de-materialized].
  //              |>--------- translation_size ---------<|
  ZoneList<llvm::Value*> values_;
  GrowableBitVector is_tagged_;
  GrowableBitVector is_uint32_;
  GrowableBitVector is_double_;

  // Map with encoded information about materialization_marker operands.
  ZoneList<uint32_t> object_mapping_;

  LLVMEnvironment* outer_;
  HEnterInlined* entry_;
  Zone* zone_;
  bool has_been_used_;
};

static bool MatchFunForInts(void* key1, void* key2) {
  return *static_cast<int32_t*>(key1) == *static_cast<int32_t*>(key2);
}

// TODO(llvm): LLVMDeoptData and LLVMRelocationData should probably be merged.
class LLVMDeoptData {
 public:
  LLVMDeoptData(Zone* zone)
     : deoptimizations_(MatchFunForInts,
                        ZoneHashMap::kDefaultHashMapCapacity,
                        ZoneAllocationPolicy(zone)),
       reverse_deoptimizations_(),
       translations_(zone),
       deoptimization_literals_(8, zone),
       zone_(zone) {}

  void Add(LLVMEnvironment* environment, int32_t patchpoint_id);
  LLVMEnvironment* GetEnvironmentByPatchpointId(int32_t patchpoint_id);
  int32_t GetPatchpointIdByEnvironment(LLVMEnvironment* env);

  TranslationBuffer& translations() { return translations_; }
  ZoneList<Handle<Object> >& deoptimization_literals() {
    return deoptimization_literals_;
  }

  int DeoptCount() { return deoptimizations_.occupancy(); }

  int DefineDeoptimizationLiteral(Handle<Object> literal);

 private:
  void* GetKey(int32_t patchpoint_id);
  uint32_t GetHash(int32_t patchpoint_id);
  // Patchpoint_id -> LLVMEnvironment*
  ZoneHashMap deoptimizations_;
  // LLVMEnvironment* -> Patchpoint_id
  // FIXME(llvm): consistency: this one is stdmap and the one above is ZoneHMap.
  std::map<LLVMEnvironment*, int32_t> reverse_deoptimizations_;
  TranslationBuffer translations_;
  ZoneList<Handle<Object> > deoptimization_literals_;

  Zone* zone_;
};

class LLVMChunk final : public LowChunk {
 public:
  virtual ~LLVMChunk();
  LLVMChunk(CompilationInfo* info, HGraph* graph)
    : LowChunk(info, graph),
      llvm_function_id_(-1),
      reloc_data_(nullptr),
      deopt_data_(nullptr),
      masm_(info->isolate(), nullptr, 0),
      target_index_for_ppid_(),
      deopt_target_offset_for_ppid_(),
      inlined_functions_(1, info->zone()) {}

  using PpIdToIndexMap = std::map<int32_t, uint32_t>;
  using PpIdToOffsetMap = std::map<int32_t, std::ptrdiff_t>;

  static LLVMChunk* NewChunk(HGraph *graph);

  Handle<Code> Codegen() override;

  void set_llvm_function_id(int id) { llvm_function_id_ = id; }
  int llvm_function_id() { return llvm_function_id_; }

  const ZoneList<Handle<SharedFunctionInfo>>& inlined_functions() const {
    return inlined_functions_;
  }

  void set_deopt_data(std::unique_ptr<LLVMDeoptData> deopt_data) {
    deopt_data_ = std::move(deopt_data);
  }
  void set_reloc_data(LLVMRelocationData* reloc_data) {
    reloc_data_ = reloc_data;
    reloc_data->DumpSafepointIds();
    reloc_data->transfer();
  }
  Assembler& masm() { return masm_; }
  PpIdToIndexMap& target_index_for_ppid() {
    return target_index_for_ppid_;
  }
  PpIdToOffsetMap& deopt_target_offset_for_ppid() {
    return deopt_target_offset_for_ppid_;
  }

  void AddInlinedFunction(Handle<SharedFunctionInfo> closure) {
    inlined_functions_.Add(closure, zone());
  }
  int GetParameterStackSlot(int index) const;

 private:
  static const int kStackSlotSize = kPointerSize;
  static const int kPhonySpillCount = 3; // rbp, rsi, rdi

  static int SpilledCount(const StackMaps& stackmaps);

  std::vector<RelocInfo> SetUpRelativeCalls(Address start,
                                            const StackMaps& stackmaps);
  StackMaps GetStackMaps();
  void SetUpDeoptimizationData(Handle<Code> code, StackMaps& stackmaps);
  void EmitSafepointTable(Assembler* code_desc,
                          StackMaps& stackmaps,
                          Address instruction_start);
  Vector<byte> GetFullRelocationInfo(
      CodeDesc& code_desc,
      const std::vector<RelocInfo>& reloc_data_from_patchpoints);
  // Returns translation index of the newly generated translation
  int WriteTranslationFor(LLVMEnvironment* env, const StackMaps& stackmaps);
  void WriteTranslation(LLVMEnvironment* environment,
                        Translation* translation,
                        const StackMaps& stackmaps,
                        int32_t patchpoint_id,
                        int start_index);
  void AddToTranslation(LLVMEnvironment* environment,
                        Translation* translation,
                        llvm::Value* op, //change
                        StackMaps::Location& location,
                        const std::vector<StackMaps::Constant> constants,
                        bool is_tagged,
                        bool is_uint32,
                        bool is_double,
                        int* object_index_pointer,
                        int* dematerialized_index_pointer);

  int llvm_function_id_;
  // Ownership gets transferred from LLVMChunkBuilder
  LLVMRelocationData* reloc_data_;
  // Ownership gets transferred from LLVMChunkBuilder
  std::unique_ptr<LLVMDeoptData> deopt_data_;
  // FIXME(llvm): memory leak. Assembler is Malloced and doesn't die either.
  Assembler masm_;
  // FIXME(llvm): memory leak
  // (this map allocates keys on the heap and doesn't die).
  // Map patchpointId -> index in masm_.code_targets_
  PpIdToIndexMap target_index_for_ppid_;
  PpIdToOffsetMap deopt_target_offset_for_ppid_;
  // TODO(llvm): hoist to base class.
  ZoneList<Handle<SharedFunctionInfo>> inlined_functions_;
};

class LLVMChunkBuilder final : public LowChunkBuilderBase {
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
        osr_preserved_values_(4, info->zone()),
        emit_debug_code_(FLAG_debug_code),
        volatile_zero_address_(nullptr),
        global_receiver_(nullptr),
        pointers_(),
        number_of_pointers_(-1) {
    reloc_data_ = new(zone()) LLVMRelocationData(zone());
  }
  ~LLVMChunkBuilder() {}

  static llvm::Type* GetLLVMType(Representation r);

  LLVMChunk* chunk() const { return static_cast<LLVMChunk*>(chunk_); };
  void set_emit_degug_code(bool v) { emit_debug_code_ = v; }
  bool emit_debug_code() { return emit_debug_code_; }
  LLVMChunkBuilder& Build();
  // LLVM requires that each phi input's label be a basic block
  // immediately preceding the given BB.
  // Hydrogen does not impose such a constraint.
  // For that reason our phis are not LLVM-compliant right after phi resolution.
  LLVMChunkBuilder& NormalizePhis();
  LLVMChunkBuilder& GiveNamesToPointerValues();
  LLVMChunkBuilder& PlaceStatePoints();
  LLVMChunkBuilder& RewriteStatePoints();
  LLVMChunkBuilder& Optimize(); // invoke llvm transformation passes for the function
  LLVMChunk* Create();

  LLVMEnvironment* AssignEnvironment();
  LLVMEnvironment* CreateEnvironment(
      HEnvironment* hydrogen_env, int* argument_index_accumulator,
      ZoneList<HValue*>* objects_to_materialize);

  void DeoptimizeIf(llvm::Value* compare,
                    bool negate = false,
                    llvm::BasicBlock* next_block = nullptr);

  void UIntToTag(HChange* instr);
  // Declare methods that deal with the individual node types.
#define DECLARE_DO(type) void Do##type(H##type* node);
  HYDROGEN_CONCRETE_INSTRUCTION_LIST(DECLARE_DO)
#undef DECLARE_DO
  static const uintptr_t kExtFillingValue = 0xabcdbeef;
  static const char* kGcStrategyName;
  static const std::string kPointersPrefix;

 private:
  static const int kSmiShift = kSmiTagSize + kSmiShiftSize;
  static const int kMaxCallSequenceLen = 16; // FIXME(llvm): find out max size.

  static llvm::CmpInst::Predicate TokenToPredicate(Token::Value op,
                                                   bool is_unsigned,
                                                   bool is_double = false);
  static bool HasTaggedValue(HValue* value);

  void GetAllEnvironmentValues(LLVMEnvironment* environment,
                               std::vector<llvm::Value*>& mapped_values);
  void CreateSafepointPollFunction();
  void DoBasicBlock(HBasicBlock* block, HBasicBlock* next_block);
  void VisitInstruction(HInstruction* current);
  void PatchReceiverToGlobalProxy();
  llvm::Value* GetParameter(int index);
  void DoPhi(HPhi* phi);
  void ResolvePhis();
  void ResolvePhis(HBasicBlock* block);
  void CreateVolatileZero();
  llvm::Value* GetVolatileZero();
  llvm::Value* BuildFastArrayOperand(HValue*, llvm::Value*,
                                     ElementsKind, uint32_t);
  llvm::Value* ConstFoldBarrier(llvm::Value* imm);
  llvm::BasicBlock* NewBlock(const std::string& name,
                             llvm::Function* = nullptr);
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
  void Assert(llvm::Value* condition, llvm::BasicBlock* next_block = nullptr);
  void InsertDebugTrap();
  void IncrementCounter(StatsCounter* counter, int value);
  llvm::Value* CallVoid(Address target);
  llvm::Value* CallAddressForMathPow(Address target,
                                     llvm::CallingConv::ID calling_conv,
                                     std::vector<llvm::Value*>& params);
  // These Call functions are intended to be highly reusable.
  // TODO(llvm): default parameters -- not very good.
  // (Especially with different default values for different methods).
  llvm::Value* CallVal(llvm::Value* callable_value,
                       llvm::CallingConv::ID calling_conv,
                       std::vector<llvm::Value*>& params,
                       llvm::Type* return_type = nullptr, // void return type
                       bool record_safepoint = true);
  llvm::Value* CallCode(Handle<Code> code,
                        llvm::CallingConv::ID calling_conv,
                        std::vector<llvm::Value*>& params);
  llvm::Value* CallAddress(Address target,
                           llvm::CallingConv::ID calling_conv,
                           std::vector<llvm::Value*>& params,
                           llvm::Type* return_type = nullptr);
  void CheckEnumCache(llvm::Value* enum_val, llvm::Value* val, llvm::BasicBlock* bb);
  llvm::Value* EnumLength(llvm::Value* map_);
  llvm::Value* FieldOperand(llvm::Value* base, int offset);
  llvm::Value* LoadFieldOperand(llvm::Value* base,
                                int offset,
                                const char* name = "");
  llvm::Value* ValueFromSmi(Smi* smi);
  llvm::Value* CreateConstant(HConstant* instr, HBasicBlock* block = NULL);
  llvm::Value* ConstructAddress(llvm::Value* base, int64_t offset);
  llvm::Value* MoveHeapObject(Handle<Object> obj);
  llvm::Value* Move(Handle<Object> object, RelocInfo::Mode rmode);
  llvm::Value* Compare(llvm::Value* lhs, llvm::Value* rhs);
  llvm::Value* Compare(llvm::Value* lhs, Handle<Object> rhs);
  llvm::Value* CompareMap(llvm::Value* object, Handle<Map> map);
  llvm::Value* CheckPageFlag(llvm::Value* object, int mask);
  // Allocate a heap number in new space with undefined value. Returns
  // tagged pointer in result register, or jumps to gc_required if new
  // space is full. // FIXME(llvm): the comment
  llvm::Value* AllocateHeapNumberSlow(HValue* instr = nullptr);
  llvm::Value* AllocateHeapNumber(MutableMode mode = IMMUTABLE);
  llvm::Value* Allocate(llvm::Value* object_size,
                        llvm::Value* (LLVMChunkBuilder::*fptr)(HValue*),
                        AllocationFlags flag,
                        HValue* instr = nullptr);
  llvm::Value* AllocateSlow(HValue* instr);
  llvm::Value* LoadAllocationTopHelper(AllocationFlags flags);
  void UpdateAllocationTopHelper(llvm::Value* result_end, AllocationFlags flags);
  void DirtyHack(int arg_count);
  llvm::CallingConv::ID GetCallingConv(CallInterfaceDescriptor descriptor);
  llvm::Value* CallRuntime(const Runtime::Function*);
  llvm::Value* CallRuntimeViaId(Runtime::FunctionId id);
  llvm::Value* CallRuntimeFromDeferred(Runtime::FunctionId id, llvm::Value* context, std::vector<llvm::Value*>);
  llvm::Value* GetContext();
  llvm::Value* GetNan();
  llvm::Value* LoadRoot(Heap::RootListIndex index);
  llvm::Value* CompareRoot(llvm::Value* val, Heap::RootListIndex index,
                           llvm::CmpInst::Predicate = llvm::CmpInst::ICMP_EQ);
  llvm::Value* CmpObjectType(llvm::Value* heap_object,
                             InstanceType type,
                             llvm::CmpInst::Predicate = llvm::CmpInst::ICMP_EQ);
  llvm::Value* RecordRelocInfo(uint64_t intptr_value, RelocInfo::Mode rmode);
  void RecordWriteForMap(llvm::Value* object, llvm::Value* map);
  void RecordWriteField(llvm::Value* object,
                        llvm::Value* key_reg,
                        int offset,
                        enum SmiCheck smi_check,
                        PointersToHereCheck ptr_check,
                        RememberedSetAction set);
  void RecordWrite(llvm::Value* object, llvm::Value* map, llvm::Value* value,
                   PointersToHereCheck ptr_check, RememberedSetAction set);
  void ChangeTaggedToDouble(HValue* val, HChange* instr);
  void ChangeDoubleToI(HValue* val, HChange* instr);
  void ChangeDoubleToTagged(HValue* val, HChange* instr);
  void ChangeTaggedToISlow(HValue* val, HChange* instr);
  void BranchTagged(HBranch* instr,
                    ToBooleanStub::Types expected,
                    llvm::BasicBlock* true_target,
                    llvm::BasicBlock* false_target);

  std::vector<llvm::Value*> GetSafepointValues(HInstruction* instr);
  void DoDummyUse(HInstruction* instr);
  void DoStoreKeyedFixedArray(HStoreKeyed* value);
  void DoLoadKeyedFixedArray(HLoadKeyed* value);
  void DoLoadKeyedExternalArray(HLoadKeyed* value);
  void DoStoreKeyedExternalArray(HStoreKeyed* value);
  void DoLoadKeyedFixedDoubleArray(HLoadKeyed* value);
  void DoStoreKeyedFixedDoubleArray(HStoreKeyed* value); 
  void Retry(BailoutReason reason);
  void AddStabilityDependency(Handle<Map> map);
  void AddDeprecationDependency(Handle<Map> map);
  void CallStackMap(int stackmap_id, llvm::Value* value);
  void CallStackMap(int stackmap_id, std::vector<llvm::Value*>& values);
  llvm::CallInst* CallPatchPoint(int32_t stackmap_id,
                                 llvm::Value* target_function,
                                 std::vector<llvm::Value*>& function_args,
                                 std::vector<llvm::Value*>& live_values,
                                 int covering_nop_size = kMaxCallSequenceLen);
  llvm::Value* CallStatePoint(int32_t stackmap_id,
                              llvm::Value* target_function,
                              llvm::CallingConv::ID calling_conv,
                              std::vector<llvm::Value*>& function_args,
                              int covering_nop_size);
  void DoMathAbs(HUnaryMathOperation* instr);
  void DoIntegerMathAbs(HUnaryMathOperation* instr);
  void DoSmiMathAbs(HUnaryMathOperation* instr);
  void DoMathPowHalf(HUnaryMathOperation* instr);
  void DoMathSqrt(HUnaryMathOperation* instr);
  void DoMathRound(HUnaryMathOperation* instr);
  void DoModByPowerOf2I(HMod* instr);
  void DoModByConstI(HMod* instr);
  void DoModI(HMod* instr);
  void DoMathFloor(HUnaryMathOperation* instr);
  void DoMathLog(HUnaryMathOperation* instr);
  void DoMathExp(HUnaryMathOperation* instr);
  llvm::Value* ExternalOperand(ExternalReference offset);
  int64_t RootRegisterDelta(ExternalReference offset);
  void PrepareCallCFunction(int num_arguments);
  int ArgumentStackSlotsForCFunctionCall(int num_arguments);
  llvm::Value* CallCFunction(ExternalReference function, std::vector<llvm::Value*>, int num_arguments);
  llvm::Value* LoadAddress(ExternalReference);
  void DumpPointerValues();
  llvm::Value* CmpInstanceType(llvm::Value*, InstanceType, llvm::CmpInst::Predicate);
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
  llvm::Function* safepoint_poll_;
  std::unique_ptr<llvm::IRBuilder<>> llvm_ir_builder_;
  std::unique_ptr<LLVMDeoptData> deopt_data_;
  LLVMRelocationData* reloc_data_;
  ZoneList<llvm::Value*> pending_pushed_args_;
  ZoneList<llvm::Value*> osr_preserved_values_;
  bool emit_debug_code_;
  llvm::Value* volatile_zero_address_;
  llvm::Value* global_receiver_;
  // TODO(llvm): choose more appropriate data structure (maybe in the zone).
  // Or even some fancy lambda to pass to createAppendLivePointersToSafepoints.
  std::set<llvm::Value*> pointers_;
  int number_of_pointers_;
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
