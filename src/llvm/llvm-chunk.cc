// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <cstdio>
#include "src/disassembler.h"
#include "llvm-chunk.h"
#include "llvm-passes.h"
#include <llvm/IR/InlineAsm.h>
#include "llvm-stackmaps.h"

namespace v8 {
namespace internal {

auto LLVMGranularity::x64_target_triple = "x86_64-unknown-linux-gnu";

LLVMChunk::~LLVMChunk() {}

Handle<Code> LLVMChunk::Codegen() {
  uint64_t address = LLVMGranularity::getInstance().GetFunctionAddress(
      llvm_function_id_);
#ifdef DEBUG
  std::cerr << "\taddress == " <<  reinterpret_cast<void*>(address) << std::endl;
  std::cerr << "\tlast code allocated == "
      << reinterpret_cast<void*>(
          LLVMGranularity::getInstance()
            .memory_manager_ref()
            ->LastAllocatedCode()
            .buffer)
      << std::endl;
  LLVMGranularity::getInstance().Err();
#else
  USE(address);
#endif

  Isolate* isolate = info()->isolate();
  CodeDesc code_desc =
      LLVMGranularity::getInstance().memory_manager_ref()->LastAllocatedCode();

#ifdef DEBUG
  LLVMGranularity::getInstance().Disass(
      code_desc.buffer, code_desc.buffer + code_desc.instr_size);
#endif

  Vector<byte> reloc_data = LLVMGranularity::getInstance().Patch(
      code_desc.buffer, code_desc.buffer + code_desc.instr_size,
      reloc_data_->reloc_map());

  // Allocate and install the code.
  Handle<Code> code = isolate->factory()->NewLLVMCode(code_desc, &reloc_data,
                                                      info()->flags());
  isolate->counters()->total_compiled_code_size()->Increment(
      code->instruction_size());

  SetUpDeoptimizationData(code);
#ifdef DEBUG
  std::cout << "Instruction start: "
      << reinterpret_cast<void*>(code->instruction_start()) << std::endl;
#endif

#ifdef DEBUG
  LLVMGranularity::getInstance().Disass(
      code->instruction_start(),
      code->instruction_start() + code->instruction_size());
#endif
  return code;
}

void LLVMChunk::WriteTranslation(LLVMEnvironment* environment,
                                 StackMaps::Record& stackmap,
                                 Translation* translation) {
  if (environment == nullptr) return;

  // The translation includes one command per value in the environment.
  int translation_size = environment->translation_size();
  // The output frame height does not include the parameters.
  int height = translation_size - environment->parameter_count();

  WriteTranslation(environment->outer(), stackmap, translation);
  bool has_closure_id = !info()->closure().is_null() &&
      !info()->closure().is_identical_to(environment->closure());

  int closure_id;
  if (has_closure_id)
    UNIMPLEMENTED();
  else
    closure_id = Translation::kSelfLiteralId;

//  int closure_id = has_closure_id
//      ? DefineDeoptimizationLiteral(environment->closure())
//      : Translation::kSelfLiteralId;

  switch (environment->frame_type()) {
    case JS_FUNCTION:
      translation->BeginJSFrame(environment->ast_id(), closure_id, height);
      break;
    case JS_CONSTRUCT:
      translation->BeginConstructStubFrame(closure_id, translation_size);
      break;
    case JS_GETTER:
      DCHECK(translation_size == 1);
      DCHECK(height == 0);
      translation->BeginGetterStubFrame(closure_id);
      break;
    case JS_SETTER:
      DCHECK(translation_size == 2);
      DCHECK(height == 0);
      translation->BeginSetterStubFrame(closure_id);
      break;
    case ARGUMENTS_ADAPTOR:
      translation->BeginArgumentsAdaptorFrame(closure_id, translation_size);
      break;
    case STUB:
      translation->BeginCompiledStubFrame();
      break;
  }

  int object_index = 0;
  int dematerialized_index = 0;

  if (translation_size != stackmap.locations.size()) {
    // To support inlining (environment -> outer != NULL)
    // we probably should introduce some mapping between llvm::Value* and
    // Location number in a StackMap.
    UNIMPLEMENTED();
  }
  for (int i = 0; i < translation_size; ++i) {
    // FIXME(llvm): (probably when adding inlining support)
    // Here we assume the i-th stackmap's Location corresponds
    // to the i-th llvm::Value which is not a very safe assumption in general.

    // Also, it seems we cannot really use llvm::Value* here, because
    // since we generated them optimization has happened
    // (therefore those values are now invalid).

    llvm::Value* value = environment->values()->at(i);
    StackMaps::Location location = stackmap.locations[i];
    AddToTranslation(environment,
                     translation,
                     value,
                     location,
                     environment->HasTaggedValueAt(i),
                     environment->HasUint32ValueAt(i),
                     &object_index,
                     &dematerialized_index);
  }
}

void LLVMChunk::AddToTranslation(LLVMEnvironment* environment,
                                 Translation* translation,
                                 llvm::Value* op,
                                 StackMaps::Location& location,
                                 bool is_tagged,
                                 bool is_uint32,
                                 int* object_index_pointer,
                                 int* dematerialized_index_pointer) {
  if (op == LLVMEnvironment::materialization_marker()) {
    UNIMPLEMENTED();
  }

  // TODO(llvm): What about StoreDouble..()?
  // It's an unimplemented case which might be hidden
  if (location.kind == StackMaps::Location::kDirect) {
    UNIMPLEMENTED();
  } else if (location.kind == StackMaps::Location::kIndirect) {
    Register reg = location.dwarf_reg.reg();
    if (!reg.is(rbp)) UNIMPLEMENTED();
    auto offset = location.offset;
    DCHECK(offset % kPointerSize == 0);
    auto index = offset / kPointerSize;
    CHECK(index != 1 && index != 0); // rbp and return address
    if (index >= 0)
      index = 1 - index;
    else {
      index = -index -
        (StandardFrameConstants::kFixedFrameSize / kPointerSize - 1);
    }
    if (is_tagged) {
      translation->StoreStackSlot(index);
    } else if (is_uint32) {
      translation->StoreUint32StackSlot(index);
    } else {
      translation->StoreInt32StackSlot(index);
    }
  } else if (location.kind == StackMaps::Location::kRegister) {
    Register reg = location.dwarf_reg.reg();
    if (is_tagged) {
      translation->StoreRegister(reg);
    } else if (is_uint32) {
      translation->StoreUint32Register(reg);
    } else {
      translation->StoreInt32Register(reg);
    }
  } else if (location.kind == StackMaps::Location::kConstantIndex) { // FIXME(llvm): it's wrong
    UNIMPLEMENTED();
//    XMMRegister reg = ToDoubleRegister(op);
//    translation->StoreDoubleRegister(reg);
  } else if (location.kind == StackMaps::Location::kConstant) {
    int src_index = deopt_data_->DefineDeoptimizationLiteral(
        isolate()->factory()->NewNumberFromInt(location.offset, TENURED));
    translation->StoreLiteral(src_index);
  } else {
    UNREACHABLE();
  }
}

int LLVMChunk::WriteTranslationFor(LLVMEnvironment* env,
                                    StackMaps::Record& stackmap) {
  int frame_count = 0;
  int jsframe_count = 0;
  for (LLVMEnvironment* e = env; e != NULL; e = e->outer()) {
    ++frame_count;
    if (e->frame_type() == JS_FUNCTION) {
      ++jsframe_count;
    }
  }
  Translation translation(&deopt_data_->translations(), frame_count,
                          jsframe_count, zone());
  WriteTranslation(env, stackmap, &translation);
  return translation.index();
}

int LLVMDeoptData::DefineDeoptimizationLiteral(Handle<Object> literal) {
  int result = deoptimization_literals_.length();
  for (int i = 0; i < deoptimization_literals_.length(); ++i) {
    if (deoptimization_literals_[i].is_identical_to(literal)) return i;
  }
  deoptimization_literals_.Add(literal, zone_);
  return result;
}

void LLVMChunk::SetUpDeoptimizationData(Handle<Code> code) {
  List<byte*>& stackmap_list =
      LLVMGranularity::getInstance().memory_manager_ref()->stackmaps();
  if (stackmap_list.length() == 0) return;
  DCHECK(stackmap_list.length() == 1);

  StackMaps stackmaps;
  DataView view(stackmap_list[0]);
  stackmaps.parse(&view);
  stackmaps.dumpMultiline(std::cerr, "  ");
  auto length = deopt_data_->DeoptCount();

  uint64_t address = LLVMGranularity::getInstance().GetFunctionAddress(
      llvm_function_id_);
  auto it = std::find_if(std::begin(stackmaps.stack_sizes),
                         std::end(stackmaps.stack_sizes),
                         [address](const StackMaps::StackSize& s) {
                           return s.functionOffset ==  address;
                         });
  DCHECK(it != std::end(stackmaps.stack_sizes));
  DCHECK(it->size / kStackSlotSize - kPhonySpillCount >= 0);
  code->set_stack_slots(it->size / kStackSlotSize - kPhonySpillCount);

  Handle<DeoptimizationInputData> data =
      DeoptimizationInputData::New(isolate(), length, TENURED);

  // FIXME(llvm): separate deopt data's stackmaps from reloc data's
  // Also, it's a sum.
  DCHECK(length == stackmaps.records.size());
  if (!length) return;

  for (auto id = 0; id < length; id++) {
    StackMaps::Record stackmap_record = stackmaps.computeRecordMap()[id];
    // Time to make a Translation from Stackmaps and Environments.
    LLVMEnvironment* env = deopt_data_->deoptimizations()[id];

    int translation_index = WriteTranslationFor(env, stackmap_record);

    data->SetAstId(id, env->ast_id());
    data->SetTranslationIndex(id, Smi::FromInt(translation_index));
    data->SetArgumentsStackHeight(id,
                                  Smi::FromInt(env->arguments_stack_height()));
    // pc offset can be obtained from the stackmap TODO(llvm):
    // but we do not support lazy deopt yet (and for eager it should be -1)
    data->SetPc(id, Smi::FromInt(-1));
  }

  auto literals_len = deopt_data_->deoptimization_literals().length();
  Handle<FixedArray> literals = isolate()->factory()->NewFixedArray(
      literals_len, TENURED);
  {
    AllowDeferredHandleDereference copy_handles;
    for (int i = 0; i < literals_len; i++) {
      literals->set(i, *(deopt_data_->deoptimization_literals()[i]));
    }
    data->SetLiteralArray(*literals);
  }

  Handle<ByteArray> translations =
      deopt_data_->translations().CreateByteArray(isolate()->factory());
  data->SetTranslationByteArray(*translations);
  // FIXME(llvm):  inlined function count
  data->SetInlinedFunctionCount(Smi::FromInt(0));
  data->SetOptimizationId(Smi::FromInt(info()->optimization_id()));
  if (info()->IsOptimizing()) {
    // Reference to shared function info does not change between phases.
    AllowDeferredHandleDereference allow_handle_dereference;
    data->SetSharedFunctionInfo(*info()->shared_info());
  } else {
    data->SetSharedFunctionInfo(Smi::FromInt(0));
  }
  data->SetWeakCellCache(Smi::FromInt(0)); // I don't know what this is.

  data->SetOsrAstId(Smi::FromInt(info()->osr_ast_id().ToInt()));
  // TODO(llvm): OSR entry point
  data->SetOsrPcOffset(Smi::FromInt(-1));

  code->set_deoptimization_data(*data);
  // TODO(llvm): it is not thread-safe. It's not anything-safe.
  // We assume a new function gets attention after the previous one
  // has been fully processed by llv8.
  LLVMGranularity::getInstance().memory_manager_ref()->DropStackmaps();
}

Vector<byte> LLVMGranularity::Patch(Address start, Address end,
                                    LLVMRelocationData::RelocMap& reloc_map) {
  RelocInfoBuffer buffer_writer(4, start);

  auto pos = start;
  while (pos < end) {
    llvm::MCInst inst;
    uint64_t size;
    auto address = 0;

    llvm::MCDisassembler::DecodeStatus s = disasm_->getInstruction(
        inst /* out */, size /* out */, llvm::ArrayRef<uint8_t>(pos, end),
        address, llvm::nulls(), llvm::nulls());
    DCHECK(s == llvm::MCDisassembler::Success);

    // const llvm::MCInstrDesc& desc = mii_->get(inst.getOpcode());
    // and testing desc.isMoveImmediate() did't work :(

    if (inst.getNumOperands() == 2 && inst.getOperand(1).isImm()) {
      auto imm = static_cast<uint64_t>(inst.getOperand(1).getImm());
      if (!is_uint32(imm) && reloc_map.count(imm)) {
        DCHECK(size == 10); // size of mov imm64
        auto pair = reloc_map[imm];
        RelocInfo rinfo = pair.first;
        LLVMRelocationData::ExtendedInfo minfo = pair.second;
        if (rinfo.rmode() == RelocInfo::CELL ||
            rinfo.rmode() == RelocInfo::EMBEDDED_OBJECT) {
          if (minfo.cell_extended) { // immediate was extended from 32 bit to 64.
            DCHECK((imm & 0xffffffff) == LLVMChunkBuilder::kExtFillingValue);
            Memory::uintptr_at(pos + 2) = imm >> 32;
          }
          rinfo.set_pc(pos + 2);
          buffer_writer.Write(&rinfo);
        } else {
          UNIMPLEMENTED();
        }
      }
    }
    pos += size;
  }
  // Here we just (rightfully) hope for RVO
  return buffer_writer.GetResult();
}

LLVMChunk* LLVMChunk::NewChunk(HGraph *graph) {
  DisallowHandleAllocation no_handles;
  DisallowHeapAllocation no_gc;
  graph->DisallowAddingNewValues();
//  int values = graph->GetMaximumValueID();
  CompilationInfo* info = graph->info();

  LLVMChunkBuilder builder(info, graph);
  LLVMChunk* chunk = builder.Build().NormalizePhis().Optimize().Create();
  if (chunk == NULL) return NULL;

  return chunk;
}

LLVMChunkBuilder& LLVMChunkBuilder::Build() {
  chunk_ = new(zone()) LLVMChunk(info(), graph());
  module_ = LLVMGranularity::getInstance().CreateModule();
  module_->setTargetTriple(LLVMGranularity::x64_target_triple);
  std::cerr << module_->getTargetTriple() << std::endl;
  status_ = BUILDING;

//  // If compiling for OSR, reserve space for the unoptimized frame,
//  // which will be subsumed into this frame.
//  if (graph()->has_osr()) {
//    for (int i = graph()->osr()->UnoptimizedFrameSlots(); i > 0; i--) {
//      chunk()->GetNextSpillIndex(GENERAL_REGISTERS);
//    }
//  }

  LLVMContext& context = LLVMGranularity::getInstance().context();

  // First param is context (v8, js context) which goes to rsi,
  // second param is the callee's JSFunction object (rdi),
  // third param is Parameter 0 which is I am not sure what
  int num_parameters = info()->num_parameters() + 3;

  std::vector<llvm::Type*> params(num_parameters, nullptr);
  for (auto i = 0; i < num_parameters; i++) {
    // For now everything is Int64. Probably it is even right for x64.
    // So in that case we are going to do come casts AFAIK
    params[i] = llvm::Type::getInt64Ty(context);
  }
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(context), params, false);

  // TODO(llvm): return type for void JS functions?
  // I think it's all right, because undefined is a tagged value
  function_ = llvm::cast<llvm::Function>(
      module_->getOrInsertFunction(module_->getModuleIdentifier(),
                                   function_type));

  function_->setCallingConv(llvm::CallingConv::X86_64_V8);

  const ZoneList<HBasicBlock*>* blocks = graph()->blocks();
  for (int i = 0; i < blocks->length(); i++) {
    HBasicBlock* next = NULL;
    if (i < blocks->length() - 1) next = blocks->at(i + 1);
    DoBasicBlock(blocks->at(i), next);
    DCHECK(!is_aborted());
  }

  ResolvePhis();

  DCHECK(module_);
  chunk()->set_llvm_function_id(std::stoi(module_->getModuleIdentifier()));
  chunk()->set_deopt_data(std::move(deopt_data_));
  chunk()->set_reloc_data(reloc_data_);
  status_ = DONE;
  return *this;
}

LLVMChunk* LLVMChunkBuilder::Create() {
  LLVMGranularity::getInstance().AddModule(std::move(module_));
  return chunk();
}

void LLVMChunkBuilder::ResolvePhis() {
  // Process the blocks in reverse order.
  const ZoneList<HBasicBlock*>* blocks = graph_->blocks();
  for (int block_id = blocks->length() - 1; block_id >= 0; --block_id) {
    HBasicBlock* block = blocks->at(block_id);
    ResolvePhis(block);
  }
}

void LLVMChunkBuilder::ResolvePhis(HBasicBlock* block) {
  for (int i = 0; i < block->phis()->length(); ++i) {
    HPhi* phi = block->phis()->at(i);
    for (int j = 0; j < phi->OperandCount(); ++j) {
      HValue* operand = phi->OperandAt(j);
      auto llvm_phi = static_cast<llvm::PHINode*>(phi->llvm_value());
      llvm_phi->addIncoming(Use(operand),
                            operand->block()->llvm_end_basic_block());
    }
  }
}

void LLVMChunkBuilder::VisitInstruction(HInstruction* current) {
  HInstruction* old_current = current_instruction_;
  current_instruction_ = current;

//  LInstruction* instr = NULL;
  if (current->CanReplaceWithDummyUses()) {
    if (current->OperandCount() == 0) {
//      instr = DefineAsRegister(new(zone()) LDummy());
      UNIMPLEMENTED();
    } else {
      DCHECK(!current->OperandAt(0)->IsControlInstruction());
      UNIMPLEMENTED();
//      instr = DefineAsRegister(new(zone())
//          LDummyUse(UseAny(current->OperandAt(0))));
    }
    for (int i = 1; i < current->OperandCount(); ++i) {
      if (current->OperandAt(i)->IsControlInstruction()) continue;
        UNIMPLEMENTED();
//      LInstruction* dummy =
//          new(zone()) LDummyUse(UseAny(current->OperandAt(i)));
//      dummy->set_hydrogen_value(current);
//      chunk()->AddInstruction(dummy, current_block_);
    }
  } else {
    HBasicBlock* successor;
    if (current->IsControlInstruction() &&
        HControlInstruction::cast(current)->KnownSuccessorBlock(&successor) &&
        successor != NULL) {
      // Goto(successor)
      llvm_ir_builder_->CreateBr(Use(successor));
    } else {
      current->CompileToLLVM(this); // the meat
    }
  }

//  argument_count_ += current->argument_delta();
//  DCHECK(argument_count_ >= 0);
//
//  if (instr != NULL) {
//    AddInstruction(instr, current);
//  }
//
  current_instruction_ = old_current;
}

llvm::BasicBlock* LLVMChunkBuilder::Use(HBasicBlock* block) {
  if (!block->llvm_start_basic_block()) {
    llvm::BasicBlock* llvm_block = llvm::BasicBlock::Create(
        LLVMGranularity::getInstance().context(),
        "BlockEntry", function_);
    block->set_llvm_start_basic_block(llvm_block);
  }
  DCHECK(block->llvm_start_basic_block());
  return block->llvm_start_basic_block();
}

llvm::Value* LLVMChunkBuilder::Use(HValue* value) {
  if (value->EmitAtUses() && !value->llvm_value()) {
    HInstruction* instr = HInstruction::cast(value);
    VisitInstruction(instr);
  }
  DCHECK(value->llvm_value());
  return value->llvm_value();
}

llvm::Value* LLVMChunkBuilder::SmiToInteger32(HValue* value) {
  llvm::Value* res = nullptr;
  if (SmiValuesAre32Bits()) {
    res = llvm_ir_builder_->CreateLShr(Use(value), kSmiShift);
    res = llvm_ir_builder_->CreateTrunc(res, llvm_ir_builder_->getInt32Ty());
  } else {
    DCHECK(SmiValuesAre31Bits());
    UNIMPLEMENTED();
    // TODO(llvm): just implement sarl(dst, Immediate(kSmiShift));
  }
  return res;
}

llvm::Value* LLVMChunkBuilder::SmiCheck(HValue* value, bool negate) {
  llvm::Value* res = llvm_ir_builder_->CreateAnd(Use(value),
                                                 llvm_ir_builder_->getInt64(1));
  return llvm_ir_builder_->CreateICmp(
      negate ? llvm::CmpInst::ICMP_NE : llvm::CmpInst::ICMP_EQ,
      res, llvm_ir_builder_->getInt64(0));
}

llvm::Value* LLVMChunkBuilder::Integer32ToSmi(HValue* value) {
  llvm::Value* int32_val = Use(value);
  llvm::Value* extended_width_val = llvm_ir_builder_->CreateZExt(
      int32_val, llvm_ir_builder_->getInt64Ty());
  return llvm_ir_builder_->CreateShl(extended_width_val, kSmiShift);
}

llvm::Value* LLVMChunkBuilder::Integer32ToSmi(llvm::Value* value) {
  llvm::Value* extended_width_val = llvm_ir_builder_->CreateZExt(
      value, llvm_ir_builder_->getInt64Ty());
  return llvm_ir_builder_->CreateShl(extended_width_val, kSmiShift);
}


llvm::Value* LLVMChunkBuilder::CallVoid(Address target) {
  llvm::Value* target_adderss = llvm_ir_builder_->getInt64(
      reinterpret_cast<uint64_t>(target));
  bool is_var_arg = false;
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      llvm_ir_builder_->getVoidTy(), is_var_arg);
  llvm::PointerType* ptr_to_function = function_type->getPointerTo();
  llvm::Value* casted = llvm_ir_builder_->CreateIntToPtr(target_adderss,
                                                         ptr_to_function);
  return llvm_ir_builder_->CreateCall(casted,  llvm::ArrayRef<llvm::Value*>());
}

llvm::Value* LLVMChunkBuilder::CallAddress(Address target,
                                           llvm::CallingConv::ID calling_conv,
                                           std::vector<llvm::Value*>& params) {
  llvm::Value* target_adderss = llvm_ir_builder_->getInt64(
      reinterpret_cast<uint64_t>(target));
  bool is_var_arg = false;

  // Tagged return type won't hurt even if in fact it's void
  auto return_type = llvm_ir_builder_->getInt8PtrTy();
  // TODO(llvm): tagged type
  auto param_type = llvm_ir_builder_->getInt64Ty();
  std::vector<llvm::Type*> param_types(params.size(), param_type);
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      return_type, param_types, is_var_arg);
  llvm::PointerType* ptr_to_function = function_type->getPointerTo();

  llvm::Value* casted = llvm_ir_builder_->CreateIntToPtr(target_adderss,
                                                         ptr_to_function);
  llvm::CallInst* call_inst = llvm_ir_builder_->CreateCall(casted, params);
  call_inst->setCallingConv(calling_conv);

  return call_inst;
}

llvm::Value* LLVMChunkBuilder::AllocateHeapNumber() {
  // FIXME(llvm): if FLAG_inline_new is set (which is the default)
  // fast inline allocation should be used
  // (otherwise runtime stub call should be performed).

  CHECK(!FLAG_inline_new);

  // return an i8*
  llvm::Value* allocated = CallRuntime(Runtime::kAllocateHeapNumber);
  // RecordSafepointWithRegisters...
  return allocated;
}

llvm::Value* LLVMChunkBuilder::CallRuntime(Runtime::FunctionId id) {
  const Runtime::Function* function = Runtime::FunctionForId(id);
  auto arg_count = function->nargs;
  // if (arg_count != 0) UNIMPLEMENTED();

  llvm::Type* int8_ptr_type =  llvm_ir_builder_->getInt8PtrTy();
  llvm::Type* int64_type =  llvm_ir_builder_->getInt64Ty();

  Address rt_target = ExternalReference(function, isolate()).address();
  // TODO(llvm): we shouldn't always save FP regs
  // moreover, we should find a way to offload such decisions to LLVM.
  // TODO(llvm): With a proper calling convention implemented in LLVM
  // we could call the runtime functions directly.
  // For now we call the CEntryStub which calls the function
  // (just as CrankShaft does).

  // Don't save FP regs because llvm will [try to] take care of that
  CEntryStub ces(isolate(), function->result_size, kDontSaveFPRegs);
  Handle<Code> code = Handle<Code>::null();
  {
    AllowHandleAllocation allow_handles;
    code = ces.GetCode();
    // FIXME(llvm,gc): respect reloc info mode...
  }

  llvm::Value* target_address = llvm_ir_builder_->getInt64(
      reinterpret_cast<uint64_t>(code->instruction_start()));
  bool is_var_arg = false;
  llvm::Type* param_types[] = { int64_type, int8_ptr_type, int64_type };
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      int8_ptr_type, param_types, is_var_arg);
  llvm::PointerType* ptr_to_function = function_type->getPointerTo();
  llvm::Value* casted = llvm_ir_builder_->CreateIntToPtr(target_address,
                                                         ptr_to_function);

  auto llvm_nargs = llvm_ir_builder_->getInt64(arg_count);
  auto target_temp = llvm_ir_builder_->getInt64(
      reinterpret_cast<uint64_t>(rt_target));
  auto llvm_rt_target = llvm_ir_builder_->CreateIntToPtr(
      target_temp, int8_ptr_type);
  auto context = GetContext();
  llvm::CallInst* call_inst = llvm_ir_builder_->CreateCall3(
      casted, llvm_nargs, llvm_rt_target, context);
  call_inst->setCallingConv(llvm::CallingConv::X86_64_V8_CES);
  // return value has type i8*
  return call_inst;
}

llvm::Value* LLVMChunkBuilder::CallRuntimeFromDeferred(Runtime::FunctionId id,
    llvm::Value* context, std::vector<llvm::Value*> params) {
  const Runtime::Function* function = Runtime::FunctionForId(id);
  auto arg_count = function->nargs + 1;
//  if (arg_count != 0) UNIMPLEMENTED();

  llvm::Type* int8_ptr_type =  llvm_ir_builder_->getInt8PtrTy();
  llvm::Type* int64_type =  llvm_ir_builder_->getInt64Ty();

  Address rt_target = ExternalReference(function, isolate()).address();
  // TODO(llvm): we shouldn't always save FP regs
  // moreover, we should find a way to offload such decisions to LLVM.
  // TODO(llvm): With a proper calling convention implemented in LLVM
  // we could call the runtime functions directly.
  // For now we call the CEntryStub which calls the function
  // (just as CrankShaft does).

  // Don't save FP regs because llvm will [try to] take care of that
  CEntryStub ces(isolate(), function->result_size, kDontSaveFPRegs);
  Handle<Code> code = Handle<Code>::null();
  {
    AllowHandleAllocation allow_handles;
    code = ces.GetCode();
    // FIXME(llvm,gc): respect reloc info mode...
  }

  llvm::Value* target_address = llvm_ir_builder_->getInt64(
      reinterpret_cast<uint64_t>(code->instruction_start()));
  bool is_var_arg = false;
  // llvm::Type* param_types [] = { int64_type ,int8_ptr_type, int64_type };
  std::vector<llvm::Type*> pTypes;
//  for (auto i = 0; i < params.size(); ++i)
//     pTypes.push_back(params[i]->getType());
  pTypes.push_back(int64_type); //, int8_ptr_type, int64_type, params[0]->getType());
  pTypes.push_back(int8_ptr_type);
  pTypes.push_back(int64_type);
  for (auto i = 0; i < params.size(); ++i) 
     pTypes.push_back(params[i]->getType());
  llvm::ArrayRef<llvm::Type*> pRef (pTypes);
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      int8_ptr_type, pRef, is_var_arg);
  llvm::PointerType* ptr_to_function = function_type->getPointerTo();
  llvm::Value* casted = llvm_ir_builder_->CreateIntToPtr(target_address,
                                                         ptr_to_function);

  auto llvm_nargs = llvm_ir_builder_->getInt64(arg_count);
  auto target_temp = llvm_ir_builder_->getInt64(
      reinterpret_cast<uint64_t>(rt_target));
  auto llvm_rt_target = llvm_ir_builder_->CreateIntToPtr(
      target_temp, int8_ptr_type);
  std::vector<llvm::Value*> actualParams;
//  for (auto i = 0; i < params.size(); ++i)
//     actualParams.push_back(params[i]);
  actualParams.push_back(llvm_nargs);
  actualParams.push_back(llvm_rt_target);
  actualParams.push_back(context);
  for (auto i = 0; i < params.size(); ++i)
     actualParams.push_back(params[i]);
  llvm::ArrayRef<llvm::Value*> args (actualParams);
  llvm::CallInst* call_inst = llvm_ir_builder_->CreateCall(
      casted, args );
  call_inst->setCallingConv(llvm::CallingConv::X86_64_V8_CES);
  // return value has type i8*
  return call_inst;

}

llvm::Value* LLVMChunkBuilder::GetContext() {
  // First parameter is our context (rsi).
  return function_->arg_begin();
}

LLVMEnvironment* LLVMChunkBuilder::AssignEnvironment() {
  HEnvironment* hydrogen_env = current_block_->last_environment();
  int argument_index_accumulator = 0;
  ZoneList<HValue*> objects_to_materialize(0, zone());
  return CreateEnvironment(
      hydrogen_env, &argument_index_accumulator, &objects_to_materialize);
}

void LLVMChunkBuilder::DeoptimizeIf(llvm::Value* compare, HBasicBlock* block) {
  LLVMEnvironment* environment = AssignEnvironment();
  deopt_data_->Add(environment);

  if (FLAG_deopt_every_n_times != 0 && !info()->IsStub()) UNIMPLEMENTED();
  if (info()->ShouldTrapOnDeopt()) UNIMPLEMENTED();

  Deoptimizer::BailoutType bailout_type = info()->IsStub()
      ? Deoptimizer::LAZY
      : Deoptimizer::EAGER;

  Address entry;
  {
    AllowHandleAllocation allow;
    entry = Deoptimizer::GetDeoptimizationEntry(isolate(),
        deopt_data_->DeoptCount() - 1, bailout_type);
  }
  if (entry == NULL) {
    Abort(kBailoutWasNotPrepared);
    return;
  }

  // TODO(llvm): create Deoptimizer::DeoptInfo & Deoptimizer::JumpTableEntry (?)

  LLVMContext& llvm_context = LLVMGranularity::getInstance().context();
  llvm::BasicBlock* saved_insert_point = llvm_ir_builder_->GetInsertBlock();
  llvm::BasicBlock* next_block = llvm::BasicBlock::Create(llvm_context,
      "BlockContinue", function_);
  llvm::BasicBlock* deopt_block = llvm::BasicBlock::Create(
      LLVMGranularity::getInstance().context(), "DeoptBlock", function_);
  llvm_ir_builder_->SetInsertPoint(deopt_block);

  // StackMap (id = #deopts, shadow_bytes=0, ...)
  llvm::Function* stackmap = llvm::Intrinsic::getDeclaration(module_.get(),
      llvm::Intrinsic::experimental_stackmap);
  std::vector<llvm::Value*> mapped_values;
  int stackmap_id = deopt_data_->DeoptCount() - 1;
  mapped_values.push_back(llvm_ir_builder_->getInt64(stackmap_id));
  int shadow_bytes = 0;
  mapped_values.push_back(llvm_ir_builder_->getInt32(shadow_bytes));
  for (auto val : *environment->values()) {
    mapped_values.push_back(val);
  }
  llvm_ir_builder_->CreateCall(stackmap, mapped_values);
  CallVoid(entry);
  llvm_ir_builder_->CreateUnreachable();

  llvm_ir_builder_->SetInsertPoint(saved_insert_point);
  llvm_ir_builder_->CreateCondBr(compare, deopt_block, next_block);
  llvm_ir_builder_->SetInsertPoint(next_block);
  block->set_llvm_end_basic_block(next_block);
}

llvm::CmpInst::Predicate LLVMChunkBuilder::TokenToPredicate(Token::Value op,
                                                            bool is_unsigned) {
  llvm::CmpInst::Predicate pred = llvm::CmpInst::BAD_FCMP_PREDICATE;
  switch (op) {
    case Token::EQ:
    case Token::EQ_STRICT:
      pred = llvm::CmpInst::ICMP_EQ;
      break;
    case Token::NE:
    case Token::NE_STRICT:
      pred = llvm::CmpInst::ICMP_NE;
      break;
    case Token::LT:
      pred = is_unsigned ? llvm::CmpInst::ICMP_ULT : llvm::CmpInst::ICMP_SLT;
      break;
    case Token::GT:
      pred = is_unsigned ? llvm::CmpInst::ICMP_UGT : llvm::CmpInst::ICMP_SGT;
      break;
    case Token::LTE:
      pred = is_unsigned ? llvm::CmpInst::ICMP_ULE : llvm::CmpInst::ICMP_SLE;
      break;
    case Token::GTE:
      pred = is_unsigned ? llvm::CmpInst::ICMP_UGE : llvm::CmpInst::ICMP_SGE;
      break;
    case Token::IN:
    case Token::INSTANCEOF:
    default:
      UNREACHABLE();
  }
  return pred;
}

LLVMChunkBuilder& LLVMChunkBuilder::NormalizePhis() {
#ifdef DEBUG
  std::cerr << "===========vvv Module BEFORE normalizationvvv===========" << std::endl;
  llvm::errs() << *(module_.get());
  std::cerr << "===========^^^ Module BEFORE normalization^^^===========" << std::endl;
#endif
  llvm::legacy::FunctionPassManager pass_manager(module_.get());
  //pass_manager.add(new NormalizePhisPass());
  pass_manager.doInitialization();
  pass_manager.run(*function_);
  pass_manager.doFinalization();
  return *this;
}

LLVMChunkBuilder& LLVMChunkBuilder::Optimize() {
  DCHECK(module_);
#ifdef DEBUG
  llvm::verifyFunction(*function_, &llvm::errs());

  std::cerr << "===========vvv Module BEFORE optimization vvv===========" << std::endl;
  llvm::errs() << *(module_.get());
  std::cerr << "===========^^^ Module BEFORE optimization ^^^===========" << std::endl;
#endif
  LLVMGranularity::getInstance().OptimizeFunciton(module_.get(), function_);
  LLVMGranularity::getInstance().OptimizeModule(module_.get());
#ifdef DEBUG
  std::cerr << "===========vvv Module AFTER optimization vvv============" << std::endl;
  llvm::errs() << *(module_.get());
  std::cerr << "===========^^^ Module AFTER optimization ^^^============" << std::endl;
#endif
  return *this;
}

void LLVMChunkBuilder::DoBasicBlock(HBasicBlock* block,
                                    HBasicBlock* next_block) {
#ifdef DEBUG
  std::cerr << __FUNCTION__ << std::endl;
#endif
  DCHECK(is_building());
  Use(block);
  // TODO(llvm): is it OK to create a new builder each time?
  // we could just set the insertion point for the irbuilder.
  llvm_ir_builder_ = llvm::make_unique<llvm::IRBuilder<>>(Use(block));
  current_block_ = block;
  next_block_ = next_block;
  if (block->IsStartBlock()) {
    block->UpdateEnvironment(graph_->start_environment());
    argument_count_ = 0;
  } else if (block->predecessors()->length() == 1) {
    // We have a single predecessor => copy environment and outgoing
    // argument count from the predecessor.
    DCHECK(block->phis()->length() == 0);
    HBasicBlock* pred = block->predecessors()->at(0);
    HEnvironment* last_environment = pred->last_environment();
    DCHECK(last_environment != NULL);
    // Only copy the environment, if it is later used again.
    if (pred->end()->SecondSuccessor() == NULL) {
      DCHECK(pred->end()->FirstSuccessor() == block);
    } else {
      if (pred->end()->FirstSuccessor()->block_id() > block->block_id() ||
          pred->end()->SecondSuccessor()->block_id() > block->block_id()) {
        last_environment = last_environment->Copy();
      }
    }
    block->UpdateEnvironment(last_environment);
    DCHECK(pred->argument_count() >= 0);
    argument_count_ = pred->argument_count();
  } else {
    // We are at a state join => process phis.
    HBasicBlock* pred = block->predecessors()->at(0);
    // No need to copy the environment, it cannot be used later.
    HEnvironment* last_environment = pred->last_environment();
    for (int i = 0; i < block->phis()->length(); ++i) {
      HPhi* phi = block->phis()->at(i);
      DoPhi(phi);
      if (phi->HasMergedIndex()) {
        last_environment->SetValueAt(phi->merged_index(), phi);
      }
    }
    for (int i = 0; i < block->deleted_phis()->length(); ++i) {
      if (block->deleted_phis()->at(i) < last_environment->length()) {
        last_environment->SetValueAt(block->deleted_phis()->at(i),
                                     graph_->GetConstantUndefined());
      }
    }
    block->UpdateEnvironment(last_environment);
    // Pick up the outgoing argument count of one of the predecessors.
    argument_count_ = pred->argument_count();
  }
  HInstruction* current = block->first();
//  int start = chunk()->instructions()->length();
  while (current != NULL && !is_aborted()) {
    // Code for constants in registers is generated lazily.
    if (!current->EmitAtUses()) {
      VisitInstruction(current);
    }
    current = current->next();
  }
//  int end = chunk()->instructions()->length() - 1;
//  if (end >= start) {
//    block->set_first_instruction_index(start);
//    block->set_last_instruction_index(end);
//  }
  block->set_argument_count(argument_count_);
  next_block_ = NULL;
  current_block_ = NULL;
}

LLVMEnvironment* LLVMChunkBuilder::CreateEnvironment(
    HEnvironment* hydrogen_env, int* argument_index_accumulator,
    ZoneList<HValue*>* objects_to_materialize) {
  if (hydrogen_env == NULL) return NULL;

  LLVMEnvironment* outer =
      CreateEnvironment(hydrogen_env->outer(), argument_index_accumulator,
                        objects_to_materialize);
  BailoutId ast_id = hydrogen_env->ast_id();
  DCHECK(!ast_id.IsNone() ||
         hydrogen_env->frame_type() != JS_FUNCTION);

  int omitted_count = (hydrogen_env->frame_type() == JS_FUNCTION)
                          ? 0
                          : hydrogen_env->specials_count();

  int value_count = hydrogen_env->length() - omitted_count;
  LLVMEnvironment* result =
      new(zone()) LLVMEnvironment(hydrogen_env->closure(),
                                  hydrogen_env->frame_type(),
                                  ast_id,
                                  hydrogen_env->parameter_count(),
                                  argument_count_,
                                  value_count,
                                  outer,
                                  hydrogen_env->entry(),
                                  zone());
  int argument_index = *argument_index_accumulator;

  // Store the environment description into the environment
  // (with holes for nested objects)
  for (int i = 0; i < hydrogen_env->length(); ++i) {
    if (hydrogen_env->is_special_index(i) &&
        hydrogen_env->frame_type() != JS_FUNCTION) {
      continue;
    }
    llvm::Value* op;
    HValue* value = hydrogen_env->values()->at(i);
    CHECK(!value->IsPushArguments());  // Do not deopt outgoing arguments
    if (value->IsArgumentsObject() || value->IsCapturedObject()) {
      op = LLVMEnvironment::materialization_marker();
    } else {
      op = Use(value);
    }
    // Well, we can add a corresponding llvm value here.
    // Though it seems redundant...
    result->AddValue(op,
                     value->representation(),
                     value->CheckFlag(HInstruction::kUint32));
  }

  // Recursively store the nested objects into the environment
  for (int i = 0; i < hydrogen_env->length(); ++i) {
    if (hydrogen_env->is_special_index(i)) continue;

    HValue* value = hydrogen_env->values()->at(i);
    if (value->IsArgumentsObject() || value->IsCapturedObject()) {
//      AddObjectToMaterialize(value, objects_to_materialize, result);
      UNIMPLEMENTED();
    }
  }

  if (hydrogen_env->frame_type() == JS_FUNCTION) {
    *argument_index_accumulator = argument_index;
  }

  return result;
}

void LLVMChunkBuilder::DoPhi(HPhi* phi) {
  Representation r = phi->RepresentationFromInputs();
  llvm::Type* phi_type;
  switch (r.kind()) {
    case Representation::Kind::kInteger32:
      phi_type = llvm_ir_builder_->getInt32Ty();
      break;
    case Representation::Kind::kTagged:
    case Representation::Kind::kSmi:
      phi_type = llvm_ir_builder_->getInt64Ty();
      break;
    default:
      UNIMPLEMENTED();
      phi_type = nullptr;
  }

  llvm::PHINode* llvm_phi = llvm_ir_builder_->CreatePHI(phi_type,
                                                        phi->OperandCount());
  phi->set_llvm_value(llvm_phi);
}

void LLVMChunkBuilder::DoBlockEntry(HBlockEntry* instr) {
  Use(instr->block());
  // TODO(llvm): LGap & parallel moves (OSR support)
  // return new(zone()) LLabel(instr->block());
}

void LLVMChunkBuilder::DoContext(HContext* instr) {
  if (instr->HasNoUses()) return;
  if (info()->IsStub()) {
    UNIMPLEMENTED();
  }
  instr->set_llvm_value(GetContext());
}

void LLVMChunkBuilder::DoParameter(HParameter* instr) {
  int index = instr->index();
#ifdef DEBUG
  std::cerr << "Parameter #" << index << std::endl;
#endif

  int num_parameters = info()->num_parameters() + 3;
  llvm::Function::arg_iterator it = function_->arg_begin();
  // First off, skip first 2 parameters: context (rsi)
  // and callee's JSFunction object (rdi).
  // Now, I couldn't find a way to tweak the calling convention through LLVM
  // in a way that parameters are passed left-to-right on the stack.
  // So for now they are passed right-to-left, as in cdecl.
  // And therefore we do the magic here.
  index = -index;
  while (--index + num_parameters > 0) ++it;
  instr->set_llvm_value(it);
}

void LLVMChunkBuilder::DoArgumentsObject(HArgumentsObject* instr) {
  // There are no real uses of the arguments object.
  // arguments.length and element access are supported directly on
  // stack arguments, and any real arguments object use causes a bailout.
  // So this value is never used.
  return;
}

void LLVMChunkBuilder::DoGoto(HGoto* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoSimulate(HSimulate* instr) {
  //  The “Simulate” instructions are for keeping track of what the stack
  //  machine state would be, in case we need to bail out and start using
  //  unoptimized code. They don’t generate any actual machine instructions.

  // Seems to be the right implementation (same as for Lithium)
  instr->ReplayEnvironment(current_block_->last_environment());
}

void LLVMChunkBuilder::DoStackCheck(HStackCheck* instr) {
#ifdef DEBUG
  std::cerr << __FUNCTION__ << std::endl;
#endif
//  LLVMContext& llvm_context = LLVMGranularity::getInstance().context();
//  std::vector<llvm::Type*> types(1, llvm_ir_builder_->getInt64Ty());
//
//  llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
//      llvm::Intrinsic::read_register, types);
//
//  auto metadata =
//    llvm::MDNode::get(llvm_context, llvm::MDString::get(llvm_context, "rsp"));
//  llvm::MetadataAsValue* val = llvm::MetadataAsValue::get(
//      llvm_context, metadata);
//
//  llvm::Value* rsp_value = llvm_ir_builder_->CreateCall(intrinsic, val);
//
//  llvm::Value* rsp_ptr = llvm_ir_builder_->CreateIntToPtr(rsp_value,
//      llvm_ir_builder_->getInt64Ty()->getPointerTo());
//  llvm::Value* r13_value = llvm_ir_builder_->CreateLoad(rsp_ptr);
//
//  llvm::Value* compare = llvm_ir_builder_->CreateICmp(llvm::CmpInst::ICMP_ULT,
//                                                      rsp_value,
//                                                      r13_value);
//
////  byte* target = isolate()->builtins()->StackCheck()->instruction_start();
//  Address target = reinterpret_cast<Address>(0);
//  instr->block()->set_llvm_end_basic_block(DoBadThing(compare, target));
//  instr->set_llvm_value(sum);
//  UNIMPLEMENTED();
}

// TODO(llvm): this version of stackmap call is most often
// used only for program counter (pc) and should be replaced in the
// future by less optimization-constraining intrinsic
// (which should be added to LLVM).
void LLVMChunkBuilder::CallStackMap(int stackmap_id, llvm::Value* value) {
  auto vector = std::vector<llvm::Value*>(1, value);
  CallStackMap(stackmap_id, vector);
}

void LLVMChunkBuilder::CallStackMap(int stackmap_id,
                                    std::vector<llvm::Value*>& values) {
  llvm::Function* stackmap = llvm::Intrinsic::getDeclaration(
      module_.get(), llvm::Intrinsic::experimental_stackmap);
  std::vector<llvm::Value*> mapped_values;
  mapped_values.push_back(llvm_ir_builder_->getInt64(stackmap_id));
  int shadow_bytes = 0;
  mapped_values.push_back(llvm_ir_builder_->getInt32(shadow_bytes));
  mapped_values.insert(mapped_values.end(), values.begin(), values.end());
  llvm_ir_builder_->CreateCall(stackmap, mapped_values);
}

llvm::Value* LLVMChunkBuilder::RecordRelocInfo(uint64_t intptr_value,
                                               RelocInfo::Mode rmode) {
  bool extended = false;
  if (is_uint32(intptr_value)) {
    intptr_value = (intptr_value << 32) | kExtFillingValue;
    extended = true;
  }
  llvm::Value* value = llvm_ir_builder_->getInt64(intptr_value);

  // Here we use the intptr_value (data) only to identify the entry in the map
  RelocInfo rinfo(rmode, intptr_value);
  LLVMRelocationData::ExtendedInfo meta_info;
  meta_info.cell_extended = extended;
  reloc_data_->Add(rinfo, meta_info);

//  int stackmap_id = deopt_data_->DeoptCount() + reloc_data_->RelocCount() - 1;
//  CallStackMap(stackmap_id, value);

  return value;
}

void LLVMChunkBuilder::DoConstant(HConstant* instr) {
  // Note: constants might have EmitAtUses() == true
  Representation r = instr->representation();
  if (r.IsSmi()) {
    // TODO(llvm): use/write a function for that
    // FIXME(llvm): this block was not tested
    int64_t int32_value = instr->Integer32Value();
    llvm::Value* value = llvm_ir_builder_->getInt64(int32_value << (kSmiShift));
    instr->set_llvm_value(value);
  } else if (r.IsInteger32()) {
    int64_t int32_value = instr->Integer32Value();
    llvm::Value* value = llvm_ir_builder_->getInt32(int32_value);
    instr->set_llvm_value(value);
  } else if (r.IsDouble()) {
    llvm::Value* value = llvm::ConstantFP::get(llvm_ir_builder_->getDoubleTy(),
                                               instr->DoubleValue());
    instr->set_llvm_value(value);
  } else if (r.IsExternal()) {
    Address external_address = instr->ExternalReferenceValue().address();
    // TODO(llvm): tagged type
    // TODO(llvm): RelocInfo::EXTERNAL_REFERENCE
    llvm::Value* value = llvm_ir_builder_->getInt64(
        reinterpret_cast<uint64_t>(external_address));
    instr->set_llvm_value(value);
  } else if (r.IsTagged()) {
    Handle<Object> object = instr->handle(isolate());
    if (object->IsSmi()) {
      // TODO(llvm): use/write a function for that
      Smi* smi = Smi::cast(*object);
      intptr_t intptr_value = reinterpret_cast<intptr_t>(smi);
      llvm::Value* value = llvm_ir_builder_->getInt64(intptr_value);
      instr->set_llvm_value(value);
    } else { // Heap object
      // MacroAssembler::MoveHeapObject
      // TODO(llvm): is allowing all these things for a block here OK?
      AllowDeferredHandleDereference using_raw_address;
      AllowHeapAllocation allow_allocation;
      AllowHandleAllocation allow_handles;
      DCHECK(object->IsHeapObject());
      if (isolate()->heap()->InNewSpace(*object)) {
        Handle<Cell> new_cell = isolate()->factory()->NewCell(object);
        DCHECK(new_cell->IsHeapObject());
        DCHECK(!isolate()->heap()->InNewSpace(*new_cell));

        auto intptr_value = reinterpret_cast<uint64_t>(new_cell.location());
        llvm::Value* value = RecordRelocInfo(intptr_value, RelocInfo::CELL);

        llvm::Value* ptr = llvm_ir_builder_->CreateIntToPtr(value,
            llvm_ir_builder_->getInt64Ty()->getPointerTo());
        llvm::Value* deref = llvm_ir_builder_->CreateLoad(ptr);
        instr->set_llvm_value(deref);
      } else {
        uint64_t intptr_value = reinterpret_cast<uint64_t>(object.location());
        llvm::Value* value = RecordRelocInfo(intptr_value,
                                             RelocInfo::EMBEDDED_OBJECT);
        instr->set_llvm_value(value);
      }
    }
  } else {
    UNREACHABLE();
  }
}

void LLVMChunkBuilder::DoReturn(HReturn* instr) {
  if (info()->IsStub()) {
    UNIMPLEMENTED();
  }
  if (info()->saves_caller_doubles()) {
    UNIMPLEMENTED();
  }
  // see NeedsEagerFrame() in lithium-codegen. For now here it's always true.
  DCHECK(!info()->IsStub());
  // I don't know what the absence (= 0) of this field means
  DCHECK(instr->parameter_count());
  if (instr->parameter_count()->IsConstant()) {
    llvm::Value* ret_val = Use(instr->value());
    llvm_ir_builder_->CreateRet(ret_val);
  } else {
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoAbnormalExit(HAbnormalExit* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoAccessArgumentsAt(HAccessArgumentsAt* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoAdd(HAdd* instr) {
  if(instr->representation().IsSmiOrInteger32()) {
    DCHECK(instr->left()->representation().Equals(instr->representation()));
    DCHECK(instr->right()->representation().Equals(instr->representation()));
    bool can_overflow = instr->CheckFlag(HValue::kCanOverflow);
    HValue* left = instr->left();
    HValue* right = instr->right();
    if (!can_overflow) {
      llvm::Value* add = llvm_ir_builder_->CreateAdd(Use(left), Use(right),"");
      instr->set_llvm_value(add);
    } else {
      auto type = instr->representation().IsSmi()
          ? llvm_ir_builder_->getInt64Ty() : llvm_ir_builder_->getInt32Ty();
      llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
          llvm::Intrinsic::sadd_with_overflow, type);

      llvm::Value* params[] = { Use(left), Use(right) };
      llvm::Value* call = llvm_ir_builder_->CreateCall(intrinsic, params);

      llvm::Value* sum = llvm_ir_builder_->CreateExtractValue(call, 0);
      llvm::Value* overflow = llvm_ir_builder_->CreateExtractValue(call, 1);
      instr->set_llvm_value(sum);
      DeoptimizeIf(overflow, instr->block());
    }
  } else if (instr->representation().IsDouble()) {
    DCHECK(instr->left()->representation().IsDouble());
    DCHECK(instr->right()->representation().IsDouble());
    HValue* left = instr->BetterLeftOperand();
    HValue* right = instr->BetterRightOperand();
    llvm::Value* fadd = llvm_ir_builder_->CreateFAdd(Use(left), Use(right));
    instr->set_llvm_value(fadd);
  } else {
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoAllocateBlockContext(HAllocateBlockContext* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoAllocate(HAllocate* instr) {
  std::vector<llvm::Value*> args;
  llvm::Value* arg1 = Integer32ToSmi(instr->size());
  int flags = 0;
  if (instr->IsOldPointerSpaceAllocation()) {
   // DCHECK(!instr->IsOldDataSpaceAllocation());
   // DCHECK(!instr->->IsNewSpaceAllocation());
    flags = AllocateTargetSpace::update(flags, OLD_POINTER_SPACE);
  } else if (instr->IsOldDataSpaceAllocation()) {
   // DCHECK(!instr->->IsNewSpaceAllocation());
    flags = AllocateTargetSpace::update(flags, OLD_DATA_SPACE);
  } else {
    flags = AllocateTargetSpace::update(flags, NEW_SPACE);
  }
  llvm::Value* value = llvm_ir_builder_->getInt32(flags);
  llvm::Value* arg2 = Integer32ToSmi(value);
  args.push_back(arg2);
  args.push_back(arg1);
  llvm::Value* alloc =  CallRuntimeFromDeferred(Runtime::kAllocateInTargetSpace, Use(instr->context()), args);
  if (instr->MustPrefillWithFiller()) {
    UNIMPLEMENTED();
  }
  instr->set_llvm_value(alloc);
}

void LLVMChunkBuilder::DoApplyArguments(HApplyArguments* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoArgumentsElements(HArgumentsElements* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoArgumentsLength(HArgumentsLength* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoBitwise(HBitwise* instr) {
  DCHECK(instr->left()->representation().Equals(instr->representation()));
  DCHECK(instr->right()->representation().Equals(instr->representation()));
  HValue* left = instr->left();
  HValue* right = instr->right();
  int32_t right_operand;
  if (right->IsConstant())
    right_operand = right->GetInteger32Constant();
  switch (instr->op()) {
    case Token::BIT_AND: {
      llvm::Value* And = llvm_ir_builder_->CreateAnd(Use(left), Use(right),"");
      instr->set_llvm_value(And);
      break;
    }
    case Token::BIT_OR: {
      llvm::Value* Or = llvm_ir_builder_->CreateOr(Use(left), Use(right),"");
      instr->set_llvm_value(Or);
      break;
    }
    case Token::BIT_XOR: {
      if(right->IsConstant() && right_operand == int32_t(~0)) {
        llvm::Value* Not = llvm_ir_builder_->CreateNot(Use(left), "");
        instr->set_llvm_value(Not);
      } else {
        llvm::Value* Xor = llvm_ir_builder_->CreateXor(Use(left), Use(right), "");
        instr->set_llvm_value(Xor);
      }
      break;
    }
    default:
      UNREACHABLE();
      break;
  }
}

void LLVMChunkBuilder::DoBoundsCheck(HBoundsCheck* instr) {
  Representation representation = instr->length()->representation();
  DCHECK(representation.Equals(instr->index()->representation()));
  DCHECK(representation.IsSmiOrInteger32());
  llvm::Type* type = llvm_ir_builder_->getInt64Ty();
  llvm::Value* left = Use(instr->length());
  llvm::Value* right = Use(instr->index());
  if (instr->index()->representation().IsInteger32()) {
     right = llvm_ir_builder_->CreateSExt(right, type);
  }  
  if (instr->length()->representation().IsInteger32()) {
     left = llvm_ir_builder_->CreateSExt(right, type);
  }
  llvm::Value* compare = llvm_ir_builder_->CreateICmpEQ(left, right);
  instr->set_llvm_value(compare);
  if (FLAG_debug_code && instr->skip_check()) {
    UNIMPLEMENTED();
  } else {
    DeoptimizeIf(Use(instr), instr->block());
  }
}

void LLVMChunkBuilder::DoBoundsCheckBaseIndexInformation(HBoundsCheckBaseIndexInformation* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoBranch(HBranch* instr) {
  HValue* value = instr->value();
  Representation r = value->representation();
  if(r.IsInteger32()) {
    llvm::Value* zero = llvm_ir_builder_->getInt32(0);
    llvm::Value* compare = llvm_ir_builder_->CreateICmpNE(Use(value), zero);
    llvm::BranchInst* branch = llvm_ir_builder_->CreateCondBr(compare,
        Use(instr->SuccessorAt(0)), Use(instr->SuccessorAt(1)));
    instr->set_llvm_value(branch);
//    llvm::outs() << "Adding module " << *(module_.get());
  } else {
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoCallWithDescriptor(HCallWithDescriptor* instr) {
  CallInterfaceDescriptor descriptor = instr->descriptor();

  if (descriptor.GetRegisterParameterCount() != 5) UNIMPLEMENTED();
  if (!descriptor.GetParameterRegister(0).is(rsi) ||
      !descriptor.GetParameterRegister(1).is(rdi) ||
      !descriptor.GetParameterRegister(2).is(rbx) ||
      !descriptor.GetParameterRegister(3).is(rcx) ||
      !descriptor.GetParameterRegister(4).is(rdx)) UNIMPLEMENTED();

  HValue* target = instr->target();
  // TODO(llvm): how  about a zone list?
  std::vector<llvm::Value*> params;
  for (int i = 1; i < instr->OperandCount(); i++)
    params.push_back(Use(instr->OperandAt(i)));

  for (int i = pending_pushed_args_.length() - 1; i >= 0; i--)
    params.push_back(pending_pushed_args_[i]);
  pending_pushed_args_.Clear();

  if (instr->IsTailCall()) {
    // Well, may be llvm can grok it's a tail call.
    // This branch just needs a test.
    UNIMPLEMENTED();
  } else {
    // TODO(llvm):
    // LPointerMap* pointers = instr->pointer_map();
    // SafepointGenerator generator(this, pointers, Safepoint::kLazyDeopt);

    if (target->IsConstant()) {
      Handle<Object> handle = HConstant::cast(target)->handle(isolate());
      Handle<Code> code = Handle<Code>::cast(handle);
      // TODO(llvm, gc): reloc info mode of the code (CODE_TARGET)...
      llvm::Value* call = CallAddress(code->instruction_start(),
                                      llvm::CallingConv::X86_64_V8_S1, params);
      instr->set_llvm_value(call);
    } else {
      UNIMPLEMENTED();
    }
    // codegen_->RecordSafepoint(pointers_, deopt_mode_); (AfterCall)
  }

  // TODO(llvm): MarkAsCall(DefineFixed(result, rax), instr);
}

void LLVMChunkBuilder::DoPushArguments(HPushArguments* instr) {
  // Every push must be followed with a call.
  CHECK(pending_pushed_args_.is_empty());
  for (int i = 0; i < instr->OperandCount(); i++)
    pending_pushed_args_.Add(Use(instr->argument(i)), info()->zone());
}

void LLVMChunkBuilder::DoCallJSFunction(HCallJSFunction* instr) {
  LLVMContext& llvm_context = LLVMGranularity::getInstance().context();
  // Don't know what this is yet.
  if (instr->pass_argument_count()) UNIMPLEMENTED();
  // Code that follows relies on this assumption
  if (!instr->function()->IsConstant()) UNIMPLEMENTED();

  Use(instr->function()); // It's an int constant (a ptr)

  Handle<JSFunction> js_function = Handle<JSFunction>::null();
  HConstant* fun_const = HConstant::cast(instr->function());
  js_function = Handle<JSFunction>::cast(fun_const->handle(isolate()));
  Address js_function_addr = reinterpret_cast<Address>(*js_function);
  Address js_context_ptr =
      js_function_addr + JSFunction::kContextOffset - kHeapObjectTag;
  Address target_entry_ptr =
      js_function_addr + JSFunction::kCodeEntryOffset - kHeapObjectTag;

  llvm::Value* js_function_val = llvm_ir_builder_->getInt64(
      reinterpret_cast<uint64_t>(js_function_addr));
  llvm::Value* js_context_ptr_val = llvm_ir_builder_->getInt64(
      reinterpret_cast<uint64_t>(js_context_ptr));
  js_context_ptr_val = llvm_ir_builder_->CreateIntToPtr(
      js_context_ptr_val,
      llvm::Type::getInt64PtrTy(llvm_context));

  auto argument_count = instr->argument_count() + 2; // rsi, rdi

  // Construct the function type (signature)
  std::vector<llvm::Type*> params(argument_count, nullptr);
  for (auto i = 0; i < argument_count; i++)
    params[i] = llvm::Type::getInt64Ty(llvm_context);
  bool is_var_arg = false;
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(llvm_context), params, is_var_arg);

  // Get the callee's address
  // TODO(llvm): it is a pointer, not an int64
  llvm::PointerType* ptr_to_function = function_type->getPointerTo();
  llvm::PointerType* ptr_to_ptr_to_function = ptr_to_function->getPointerTo();

  llvm::Value* target_entry_ptr_val = llvm_ir_builder_->CreateIntToPtr(
     llvm_ir_builder_->getInt64(reinterpret_cast<int64_t>(target_entry_ptr)),
     ptr_to_ptr_to_function);
  llvm::Value* target_entry_val =  llvm_ir_builder_->CreateAlignedLoad(
      target_entry_ptr_val, 1);

  // Set up the actual arguments
  std::vector<llvm::Value*> args(argument_count, nullptr);
  // FIXME(llvm): pointers, not int64
  args[0] = llvm_ir_builder_->CreateAlignedLoad(js_context_ptr_val, 1);
  args[1] = js_function_val;

  DCHECK(pending_pushed_args_.length() + 2 == argument_count);
  // The order is reverse because X86_64_V8 is not implemented quite right.
  for (int i = 0; i < pending_pushed_args_.length(); i++) {
    args[argument_count - 1 - i] = pending_pushed_args_[i];
  }
  pending_pushed_args_.Clear();

  llvm::CallInst* call = llvm_ir_builder_->CreateCall(target_entry_val, args);
  call->setCallingConv(llvm::CallingConv::X86_64_V8);
  instr->set_llvm_value(call);
}

void LLVMChunkBuilder::DoCallFunction(HCallFunction* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCallNew(HCallNew* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCallNewArray(HCallNewArray* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCallRuntime(HCallRuntime* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCallStub(HCallStub* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCapturedObject(HCapturedObject* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::ChangeDoubleToTagged(HValue* val, HChange* instr) {
  // TODO(llvm): this case in Crankshaft utilizes deferred calling.

  DCHECK(Use(val)->getType()->isDoubleTy());
  if (FLAG_inline_new) UNIMPLEMENTED();

  // TODO(llvm): tagged value will be i8* in the future
  llvm::Type* tagged_type = llvm_ir_builder_->getInt64Ty();
  llvm::PointerType* ptr_to_tagged = llvm::PointerType::get(tagged_type, 0);

  llvm::Value* new_heap_number = AllocateHeapNumber(); // i8*

  auto offset = HeapNumber::kValueOffset - kHeapObjectTag;
  auto llvm_offset = llvm_ir_builder_->getInt64(offset);
  llvm::Value* store_address = llvm_ir_builder_->CreateGEP(new_heap_number,
                                                           llvm_offset);
  llvm::Value* casted_adderss = llvm_ir_builder_->CreateBitCast(store_address,
                                                                ptr_to_tagged);
  llvm::Value* casted_val = llvm_ir_builder_->CreateBitCast(Use(val),
                                                            tagged_type);
  // [(i8*)new_heap_number + offset] = val;
  llvm_ir_builder_->CreateStore(casted_val, casted_adderss);

  auto new_heap_number_casted = llvm_ir_builder_->CreatePtrToInt(
      new_heap_number, tagged_type);
  instr->set_llvm_value(new_heap_number_casted); // no offset

  //  TODO(llvm): AssignPointerMap(Define(result, result_temp));
}

llvm::Value* LLVMChunkBuilder::CompareRoot(llvm::Value* val, HChange* instr)
{
  LLVMContext& context = LLVMGranularity::getInstance().context();
  ExternalReference roots_array_start =
        ExternalReference::roots_array_start(isolate());
  Address target_address = roots_array_start.address();
  auto address_val = llvm_ir_builder_->getInt64(
      reinterpret_cast<uint64_t>(target_address));
 
  llvm::Value* int8_ptr_2 = llvm_ir_builder_->CreateIntToPtr(
        address_val, llvm::Type::getInt64PtrTy(context)); 
 
  llvm::Value* gep = llvm_ir_builder_->CreateLoad(int8_ptr_2);
  auto value = llvm_ir_builder_->getInt64(kRootRegisterBias);
  llvm::Value* r13_val = llvm_ir_builder_->CreateAdd(gep, value);
  
  auto offset = llvm_ir_builder_->getInt64((Heap::kHeapNumberMapRootIndex
        << kPointerSizeLog2) - kRootRegisterBias);
  llvm::Value* int8_ptr = llvm_ir_builder_->CreateIntToPtr(
        r13_val, llvm_ir_builder_->getInt8PtrTy());
  llvm::Value* gep_2 = llvm_ir_builder_->CreateGEP(int8_ptr, offset);
 
  llvm::Type* int_type = llvm_ir_builder_->getInt64Ty();
  llvm::PointerType* ptr_to_int = llvm::PointerType::get(int_type, 0);
 
  llvm::Value* bitcast_1 = llvm_ir_builder_->CreateBitCast(gep_2, ptr_to_int);
  llvm::Value* bitcast_2 = llvm_ir_builder_->CreateBitCast(val, ptr_to_int);
  llvm::Value* load_first = llvm_ir_builder_->CreateLoad(bitcast_2);
  llvm::Value* load_second = llvm_ir_builder_->CreateLoad(bitcast_1);
 
  llvm::Value* cmp_result = llvm_ir_builder_->CreateICmpSGT(load_first, load_second);
 
  return cmp_result;
  
}

void LLVMChunkBuilder::ChangeTaggedToDouble(HValue* val, HChange* instr) {
  LLVMContext& context = LLVMGranularity::getInstance().context();
  bool can_convert_undefined_to_nan =
      instr->can_convert_undefined_to_nan();
  
  bool deoptimize_on_minus_zero = instr->deoptimize_on_minus_zero();
  
  llvm::BasicBlock* cond_true = llvm::BasicBlock::Create(context,
                                                         "Cond True Block",
                                                         function_);
  llvm::BasicBlock* cond_false = llvm::BasicBlock::Create(context,
                                                          "Cond False Block",
                                                          function_);
  llvm::BasicBlock* continue_block = llvm::BasicBlock::Create(context,
                                                              "Continue Block",
                                                              function_);
  llvm::Type* double_type = llvm_ir_builder_->getDoubleTy();
  llvm::PointerType* ptr_to_double = llvm::PointerType::get(double_type, 0);

  llvm::Value* cmp_val = nullptr;

  llvm::LoadInst* load_d = nullptr;
  bool not_smi = false;
  llvm::Value* cond = SmiCheck(val, not_smi);
  if (!val->representation().IsSmi()) {
    llvm_ir_builder_->CreateCondBr(cond, cond_true, cond_false);
    llvm_ir_builder_->SetInsertPoint(cond_false);

    auto offset_1 = llvm_ir_builder_->getInt64(HeapObject::kMapOffset - kHeapObjectTag);
    llvm::Value* int8_ptr_1 = llvm_ir_builder_->CreateIntToPtr(
        Use(val), llvm_ir_builder_->getInt8PtrTy());
    llvm::Value* cmp_first = llvm_ir_builder_->CreateGEP(int8_ptr_1, offset_1);
    cmp_val = CompareRoot(cmp_first, instr);
    auto offset = llvm_ir_builder_->getInt64(
        HeapNumber::kValueOffset - kHeapObjectTag);
    llvm::Value* int8_ptr = llvm_ir_builder_->CreateIntToPtr(
        Use(val), llvm_ir_builder_->getInt8PtrTy());
    llvm::Value* gep = llvm_ir_builder_->CreateGEP(int8_ptr, offset);
    llvm::Value* bitcast = llvm_ir_builder_->CreateBitCast(gep, ptr_to_double);
    load_d = llvm_ir_builder_->CreateLoad(bitcast);
    // TODO(llvm): deopt
    // AssignEnvironment(DefineSameAsFirst(new(zone()) LCheckSmi(value)));

    //FIXME: false must me deleted after full implementation
    if (can_convert_undefined_to_nan && false) {
      UNIMPLEMENTED();
    } else {
      DeoptimizeIf(cmp_val, instr->block());
    }

    if (deoptimize_on_minus_zero) {
      UNIMPLEMENTED();
    }
    
    llvm_ir_builder_->CreateBr(continue_block);
  }
  
  llvm_ir_builder_->SetInsertPoint(cond_true);
  llvm::Value* int32_val = SmiToInteger32(val);
  llvm::Value* double_val = llvm_ir_builder_->CreateSIToFP(int32_val,
                                                           double_type);
  llvm_ir_builder_->CreateBr(continue_block);
  llvm_ir_builder_->SetInsertPoint(continue_block);
  llvm::PHINode* phi = llvm_ir_builder_->CreatePHI(double_type, 2);
  phi->addIncoming(load_d, cond_false);
  phi->addIncoming(double_val, cond_true);
  instr->set_llvm_value(phi);
}

void LLVMChunkBuilder::DoChange(HChange* instr) {
  Representation from = instr->from();
  Representation to = instr->to();
  HValue* val = instr->value();
  if (from.IsSmi()) {
    if (to.IsTagged()) {
      instr->set_llvm_value(Use(val));
      return;
    }
    from = Representation::Tagged();
  }
  if (from.IsTagged()) {
    if (to.IsDouble()) {
      ChangeTaggedToDouble(val, instr);
    } else if (to.IsSmi()) {
      if (!val->type().IsSmi()) {
        bool not_smi = true;
        llvm::Value* cond = SmiCheck(val, not_smi);
        DeoptimizeIf(cond, instr->block());
      }
      instr->set_llvm_value(Use(val));
    } else {
      DCHECK(to.IsInteger32());
      if (val->type().IsSmi() || val->representation().IsSmi()) {
        // convert smi to int32, no need to perform smi check
        // lithium codegen does __ AssertSmi(input)
        instr->set_llvm_value(SmiToInteger32(val));
      } else {
#ifdef DEBUG
      std::cerr << "SECOND " << instr->from().IsSmi()
          << " " << instr->from().IsTagged() << std::endl;
#endif
        bool truncating = instr->CanTruncateToInt32();
        USE(truncating);
        // TODO(llvm): perform smi check, bailout if not a smi
        // see LCodeGen::DoTaggedToI
        if (!val->representation().IsSmi()) {
          bool not_smi = true;
          llvm::Value* cond = SmiCheck(val, not_smi);
          DeoptimizeIf(cond, instr->block());
        }
        instr->set_llvm_value(SmiToInteger32(val));
//        if (!val->representation().IsSmi()) result = AssignEnvironment(result);
      }
    }
  } else if (from.IsDouble()) {
      if (to.IsInteger32()) {
        llvm::Type* type = llvm_ir_builder_->getInt32Ty();
        llvm::Value* casted_int =  llvm_ir_builder_->CreateFPToSI(Use(val),
                                                                  type);
        instr->set_llvm_value(casted_int);
      } else if (to.IsTagged()) {
        ChangeDoubleToTagged(val, instr);
      } else {
        UNIMPLEMENTED();
      }
  } else if (from.IsInteger32()) {
    if (to.IsTagged()) {
      if (!instr->CheckFlag(HValue::kCanOverflow)) {
        instr->set_llvm_value(Integer32ToSmi(val));
      } else {
        UNIMPLEMENTED();
      }
    } else if (to.IsSmi()) {
      UNIMPLEMENTED();
    } else {
      DCHECK(to.IsDouble());
      UNIMPLEMENTED();
    }
  }
}

void LLVMChunkBuilder::DoCheckHeapObject(HCheckHeapObject* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCheckInstanceType(HCheckInstanceType* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::Retry(BailoutReason reason) {
  info()->RetryOptimization(reason);
  status_ = ABORTED;
}

void LLVMChunkBuilder::AddStabilityDependency(Handle<Map> map) {
  if (!map->is_stable()) return Retry(kMapBecameUnstable);
  chunk()->AddStabilityDependency(map);
  // TODO(llvm): stability_dependencies_ unused yet
}

void LLVMChunkBuilder::DoCheckMaps(HCheckMaps* instr) {
  if (instr->IsStabilityCheck()) {
    const UniqueSet<Map>* maps = instr->maps();
    for (int i = 0; i < maps->size(); ++i) {
      AddStabilityDependency(maps->at(i).handle());
    }
    return;
  }
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCheckMapValue(HCheckMapValue* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCheckSmi(HCheckSmi* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCheckValue(HCheckValue* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoClampToUint8(HClampToUint8* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoClassOfTestAndBranch(HClassOfTestAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCompareNumericAndBranch(HCompareNumericAndBranch* instr) {
  Representation r = instr->representation();
  HValue* left = instr->left();
  HValue* right = instr->right();
  DCHECK(left->representation().Equals(r));
  DCHECK(right->representation().Equals(r));
  bool is_unsigned = r.IsDouble()
      || left->CheckFlag(HInstruction::kUint32)
      || right->CheckFlag(HInstruction::kUint32);
  llvm::CmpInst::Predicate pred = TokenToPredicate(instr->token(), is_unsigned);

  if (r.IsSmi()) {
    UNIMPLEMENTED();
  } else if (r.IsInteger32()) {
    llvm::Value* compare = llvm_ir_builder_->CreateICmp(pred, Use(left),
                                                        Use(right));
    llvm::Value* branch = llvm_ir_builder_->CreateCondBr(compare,
        Use(instr->SuccessorAt(0)), Use(instr->SuccessorAt(1)));
    instr->set_llvm_value(branch);
  } else {
    DCHECK(r.IsDouble());
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoCompareHoleAndBranch(HCompareHoleAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCompareGeneric(HCompareGeneric* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCompareMinusZeroAndBranch(HCompareMinusZeroAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCompareObjectEqAndBranch(HCompareObjectEqAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCompareMap(HCompareMap* instr) {
   LLVMContext& context = LLVMGranularity::getInstance().context();
   auto offset = llvm_ir_builder_->getInt64(HeapObject::kMapOffset - kHeapObjectTag);
   Handle<Object> handle_value = instr->map().handle();
   int64_t value = reinterpret_cast<int64_t>((handle_value.location()));
   auto address_val = llvm_ir_builder_->getInt64(value);
   //llvm::Value* int8_ptr = llvm_ir_builder_->CreateIntToPtr(
     //    address_val, llvm::Type::getInt64PtrTy(context));
   llvm::Value* int8_ptr_1 = llvm_ir_builder_->CreateIntToPtr(
         Use(instr->value()), llvm::Type::getInt64PtrTy(context));
   llvm::Value* gep = llvm_ir_builder_->CreateGEP(int8_ptr_1, offset);
   llvm::Value* load = llvm_ir_builder_->CreateLoad(gep); 
   llvm::Value* compare = llvm_ir_builder_->CreateICmpNE(load, address_val);
   llvm::BranchInst* branch = llvm_ir_builder_->CreateCondBr(compare,
         Use(instr->SuccessorAt(0)), Use(instr->SuccessorAt(1)));
   instr->set_llvm_value(branch);
  //UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoConstructDouble(HConstructDouble* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoDateField(HDateField* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoDebugBreak(HDebugBreak* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoDeclareGlobals(HDeclareGlobals* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoDeoptimize(HDeoptimize* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoDiv(HDiv* instr) {
  if(instr->representation().IsInteger32() || instr->representation().IsSmi()) {
    DCHECK(instr->left()->representation().Equals(instr->representation()));
    DCHECK(instr->right()->representation().Equals(instr->representation()));
    HValue* dividend = instr->left();
    HValue* divisor = instr->right();
    llvm::Value* Div = llvm_ir_builder_->CreateUDiv(Use(dividend), Use(divisor),"");
    instr->set_llvm_value(Div);
  } else if (instr->representation().IsDouble()) {
    DCHECK(instr->representation().IsDouble());
    DCHECK(instr->left()->representation().IsDouble());
    DCHECK(instr->right()->representation().IsDouble());
    HValue* left = instr->left();
    HValue* right = instr->right();
    llvm::Value* fDiv =  llvm_ir_builder_->CreateFDiv(Use(left), Use(right), "");
    instr->set_llvm_value(fDiv);
   }
  else {
    UNIMPLEMENTED();
  } 
}

void LLVMChunkBuilder::DoDoubleBits(HDoubleBits* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoDummyUse(HDummyUse* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoEnterInlined(HEnterInlined* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoEnvironmentMarker(HEnvironmentMarker* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoForceRepresentation(HForceRepresentation* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoForInCacheArray(HForInCacheArray* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoForInPrepareMap(HForInPrepareMap* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoFunctionLiteral(HFunctionLiteral* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoGetCachedArrayIndex(HGetCachedArrayIndex* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoHasCachedArrayIndexAndBranch(HHasCachedArrayIndexAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoHasInstanceTypeAndBranch(HHasInstanceTypeAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoInnerAllocatedObject(HInnerAllocatedObject* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoInstanceOf(HInstanceOf* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoInstanceOfKnownGlobal(HInstanceOfKnownGlobal* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoInvokeFunction(HInvokeFunction* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoIsConstructCallAndBranch(HIsConstructCallAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoIsObjectAndBranch(HIsObjectAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoIsStringAndBranch(HIsStringAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoIsSmiAndBranch(HIsSmiAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoIsUndetectableAndBranch(HIsUndetectableAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoLeaveInlined(HLeaveInlined* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoLoadContextSlot(HLoadContextSlot* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoLoadFieldByIndex(HLoadFieldByIndex* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoLoadFunctionPrototype(HLoadFunctionPrototype* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoLoadGlobalCell(HLoadGlobalCell* instr) {
  Handle<Object> handle_value = instr->cell().handle(); 
  llvm::Type* type = llvm_ir_builder_->getInt64Ty();
  llvm::PointerType* ptr_to_type = llvm::PointerType::get(type, 0);
  int64_t value = reinterpret_cast<int64_t>((handle_value.location()));
  auto address_val = llvm_ir_builder_->getInt64(value);
  llvm::Value* int8_ptr = llvm_ir_builder_->CreateIntToPtr(
        address_val, llvm_ir_builder_->getInt8PtrTy());
  llvm::Value* gep = llvm_ir_builder_->CreateGEP(int8_ptr, llvm_ir_builder_->getInt64(7));
  llvm::Value* casted_address = llvm_ir_builder_->CreateBitCast(gep, ptr_to_type);
  llvm::Value* load_cell = llvm_ir_builder_->CreateLoad(casted_address);
  instr->set_llvm_value(load_cell); 
  if(instr->RequiresHoleCheck()){
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoLoadGlobalGeneric(HLoadGlobalGeneric* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoLoadKeyed(HLoadKeyed* instr) {
  HValue* key = instr->key();
  llvm::Type* type = llvm_ir_builder_->getInt64Ty();
  Representation representation = instr->representation();
  bool requires_hole_check = instr->RequiresHoleCheck();
  int shift_size = ElementsKindToShiftSize(FAST_ELEMENTS);
  uint32_t inst_offset = instr->base_offset();
  llvm::Value* gep_0 = nullptr;
  if (kPointerSize == kInt32Size && !key->IsConstant() &&
      instr->IsDehoisted()) {
    UNIMPLEMENTED();
  }
  if (representation.IsInteger32() && SmiValuesAre32Bits() &&
      instr->elements_kind() == FAST_SMI_ELEMENTS) {
    DCHECK(!requires_hole_check);
    if (FLAG_debug_code) {
      UNIMPLEMENTED();
    }
    inst_offset += kPointerSize / 2;
    
  }
  if (key->IsConstant()) {
    uint32_t const_val = (HConstant::cast(key))->Integer32Value();
    auto offset = llvm_ir_builder_->getInt64((const_val << shift_size) +
          inst_offset);
    llvm::Value* int_ptr = llvm_ir_builder_->CreateIntToPtr(
          Use(instr->elements()), llvm_ir_builder_->getInt8PtrTy());
    gep_0 = llvm_ir_builder_->CreateGEP(int_ptr, offset);
  } else {
     llvm::Value* lkey = Use(key);
     if (key->representation().IsInteger32()) {
        lkey = llvm_ir_builder_->CreateSExt(lkey, type);
     }
     // ScaleFactor scale_factor = static_cast<ScaleFactor>(shift_size);
     llvm::Value* scale = llvm_ir_builder_->getInt64(8); //FIXME: //find a way to pass by ScaleFactor
     llvm::Value* mul = llvm_ir_builder_->CreateMul(lkey, scale);
     auto offset = llvm_ir_builder_->getInt64(inst_offset);
     llvm::Value* add = llvm_ir_builder_->CreateAdd(mul, offset);
     llvm::Value* int_ptr = llvm_ir_builder_->CreateIntToPtr(
          Use(instr->elements()), llvm_ir_builder_->getInt8PtrTy());
     gep_0 = llvm_ir_builder_->CreateGEP(int_ptr, add);
   
  }
  llvm::Value* load = llvm_ir_builder_->CreateLoad(gep_0);
  if (requires_hole_check) {
    if (IsFastSmiElementsKind(instr->elements_kind())) {
      UNIMPLEMENTED();
    } else {
      UNIMPLEMENTED();
    }
  }
  instr->set_llvm_value(load);
}

void LLVMChunkBuilder::DoLoadKeyedGeneric(HLoadKeyedGeneric* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoLoadNamedField(HLoadNamedField* instr) {

  HObjectAccess access = instr->access();
  int offset = access.offset() - 1;
  if (access.IsExternalMemory()) {
    UNIMPLEMENTED();
  }

  if (instr->representation().IsDouble()){
    UNIMPLEMENTED();
  }

  if(!access.IsInobject()) {
    UNIMPLEMENTED();
  }

  Representation representation = access.representation();
  if (representation.IsSmi() && SmiValuesAre32Bits() &&
    instr->representation().IsInteger32()) {
    if(FLAG_debug_code) {
      UNIMPLEMENTED();
    }
    STATIC_ASSERT(kSmiTag == 0);
    DCHECK(kSmiTagSize + kSmiShiftSize == 32);
    offset += kPointerSize / 2;
    representation = Representation::Integer32();
  }
 
  llvm::Type* type = llvm_ir_builder_->getInt64Ty();
  llvm::PointerType* ptr_to_type = llvm::PointerType::get(type, 0); 
  auto offset_1 = llvm_ir_builder_->getInt64(offset);
  llvm::Value* int8_ptr = llvm_ir_builder_->CreateIntToPtr(Use(instr->object()), llvm_ir_builder_->getInt8PtrTy());
  llvm::Value* obj = llvm_ir_builder_->CreateGEP(int8_ptr, offset_1);
  llvm::Value* casted_address = llvm_ir_builder_->CreateBitCast(obj, ptr_to_type);
  llvm::Value* res = llvm_ir_builder_->CreateLoad(casted_address);
  instr->set_llvm_value(res);
 
}

void LLVMChunkBuilder::DoLoadNamedGeneric(HLoadNamedGeneric* instr) {
    UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoLoadRoot(HLoadRoot* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoMapEnumLength(HMapEnumLength* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoMathFloorOfDiv(HMathFloorOfDiv* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoMathMinMax(HMathMinMax* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoMod(HMod* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoMul(HMul* instr) {
  if(instr->representation().IsInteger32() || instr->representation().IsSmi()) {
    DCHECK(instr->left()->representation().Equals(instr->representation()));
    DCHECK(instr->right()->representation().Equals(instr->representation()));
    HValue* left = instr->left();
    HValue* right = instr->right();
    if (instr->representation().IsSmi()) {
      llvm::Value* shift = llvm_ir_builder_->CreateAShr(Use(left), 32);
      llvm::Value* Mul = llvm_ir_builder_->CreateNSWMul(shift, Use(right), "");
      instr->set_llvm_value(Mul);
    } else {
      llvm::Value* Mul = llvm_ir_builder_->CreateNSWMul(Use(left), Use(right), "");
      instr->set_llvm_value(Mul);
    }
  } else if (instr->representation().IsDouble()) {
    DCHECK(instr->representation().IsDouble());
    DCHECK(instr->left()->representation().IsDouble());
    DCHECK(instr->right()->representation().IsDouble());
    HValue* left = instr->left();
    HValue* right = instr->right();
    llvm::Value* fMul =  llvm_ir_builder_->CreateFMul(Use(left), Use(right), "");
    instr->set_llvm_value(fMul);
   }
  else {
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoOsrEntry(HOsrEntry* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoPower(HPower* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoRegExpLiteral(HRegExpLiteral* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoRor(HRor* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoSar(HSar* instr) {
  if(instr->representation().IsInteger32() || instr->representation().IsSmi()) {
    DCHECK(instr->left()->representation().Equals(instr->representation()));
    DCHECK(instr->right()->representation().Equals(instr->representation()));
    HValue* left = instr->left();
    HValue* right = instr->right();
    llvm::Value* AShr = llvm_ir_builder_->CreateAShr(Use(left), Use(right),"");
    instr->set_llvm_value(AShr);
  }
  else {
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoSeqStringGetChar(HSeqStringGetChar* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoSeqStringSetChar(HSeqStringSetChar* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoShl(HShl* instr) {
  if(instr->representation().IsInteger32() || instr->representation().IsSmi()) {
    DCHECK(instr->left()->representation().Equals(instr->representation()));
    DCHECK(instr->right()->representation().Equals(instr->representation()));
    HValue* left = instr->left();
    HValue* right = instr->right();
    llvm::Value* Shl = llvm_ir_builder_->CreateShl(Use(left), Use(right),"");
    instr->set_llvm_value(Shl);
  }
  else {
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoShr(HShr* instr) {
  if(instr->representation().IsInteger32() || instr->representation().IsSmi()) {
    DCHECK(instr->left()->representation().Equals(instr->representation()));
    DCHECK(instr->right()->representation().Equals(instr->representation()));
    HValue* left = instr->left();
    HValue* right = instr->right();
    llvm::Value* LShr = llvm_ir_builder_->CreateLShr(Use(left), Use(right),"");
    instr->set_llvm_value(LShr);
  }
  else {
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoStoreCodeEntry(HStoreCodeEntry* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoStoreContextSlot(HStoreContextSlot* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoStoreFrameContext(HStoreFrameContext* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoStoreGlobalCell(HStoreGlobalCell* instr) {
  if (instr->RequiresHoleCheck()) {
    UNIMPLEMENTED();
  } else {
    Handle<Object> handle_value = instr->cell().handle();
    llvm::Type* type = llvm_ir_builder_->getInt64Ty();
    llvm::PointerType* ptr_to_type = llvm::PointerType::get(type, 0);
    int64_t value = reinterpret_cast<int64_t>(*(handle_value.location()));
    auto address_val = llvm_ir_builder_->getInt64(value);
    llvm::Value* int8_ptr = llvm_ir_builder_->CreateIntToPtr(
          address_val, llvm_ir_builder_->getInt8PtrTy());
    llvm::Value* gep = llvm_ir_builder_->CreateGEP(int8_ptr, llvm_ir_builder_->getInt64(7));
    llvm::Value* casted_address = llvm_ir_builder_->CreateBitCast(gep, ptr_to_type);
    llvm::Value* store_cell = llvm_ir_builder_->CreateStore(Use(instr->value()), casted_address);
    instr->set_llvm_value(store_cell);
   }
}

void LLVMChunkBuilder::DoStoreKeyed(HStoreKeyed* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoStoreKeyedGeneric(HStoreKeyedGeneric* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoStoreNamedField(HStoreNamedField* instr) {
  Representation representation = instr->representation();

   HObjectAccess access = instr->access();
   int offset = access.offset() - 1;

  if (access.IsExternalMemory()) { 
    UNIMPLEMENTED();
  }

  // Register object = ToRegister(instr->object());
  // __ AssertNotSmi(object);

  if (!FLAG_unbox_double_fields && representation.IsDouble()) {
    UNIMPLEMENTED();
  }

  if (instr->has_transition()) {
    UNIMPLEMENTED();
  }

  // Do the store.
  // Register write_register = object;
  if (!access.IsInobject()) {
    UNIMPLEMENTED();
  }

  if (representation.IsSmi() && SmiValuesAre32Bits() &&
      instr->value()->representation().IsInteger32()) {
    UNIMPLEMENTED();
  }

  //Operand operand = FieldOperand(write_register, offset);

  if (FLAG_unbox_double_fields && representation.IsDouble()) {
    UNIMPLEMENTED();
  } else if (!instr->value()->IsConstant()) {
    UNIMPLEMENTED();
  } else {
    HConstant* constant = HConstant::cast(instr->value());
    if (constant->representation().IsInteger32()) {
      auto llvm_offset = llvm_ir_builder_->getInt64(offset);
      llvm::Value* store_address = llvm_ir_builder_->CreateGEP(Use(instr->object()),
                                                              llvm_offset);
      llvm::Type* type = llvm_ir_builder_->getInt32Ty();
      llvm::PointerType* ptr_to_type = llvm::PointerType::get(type, 0);
      llvm::Value* casted_adderss = llvm_ir_builder_->CreateBitCast(store_address,
                                                                ptr_to_type);
      llvm::Value* casted_value = llvm_ir_builder_->CreateBitCast(Use(constant),
                                                                  type);
      llvm_ir_builder_->CreateStore(casted_value, casted_adderss);
    } else if (constant->representation().IsSmi()){
      auto llvm_offset = llvm_ir_builder_->getInt64(offset);
      llvm::Value* store_address = llvm_ir_builder_->CreateGEP(Use(instr->object()),
                                                              llvm_offset);
      llvm::Type* type = llvm_ir_builder_->getInt64Ty();
      llvm::PointerType* ptr_to_type = llvm::PointerType::get(type, 0);
      llvm::Value* casted_adderss = llvm_ir_builder_->CreateBitCast(store_address,
                                                                ptr_to_type);
      llvm::Value* casted_value = llvm_ir_builder_->CreateBitCast(Use(constant),
                                                                  type);
      llvm_ir_builder_->CreateStore(casted_value, casted_adderss);
    } else {
      llvm::Type* type = llvm_ir_builder_->getInt64Ty();
      llvm::PointerType* ptr_to_type = llvm::PointerType::get(type, 0);
      Handle<Object> handle_value = constant->handle(isolate()); 
      int64_t value = reinterpret_cast<int64_t>(*(handle_value.location()));
      auto llvm_offset = llvm_ir_builder_->getInt64(offset);
      llvm::Value* store_address = llvm_ir_builder_->CreateGEP(Use(instr->object()),
                                                           llvm_offset);
      llvm::Value* casted_adderss = llvm_ir_builder_->CreateBitCast(store_address,
                                                                ptr_to_type);
      auto llvm_val = llvm_ir_builder_->getInt64(value);
      llvm_ir_builder_->CreateStore(llvm_val, casted_adderss);

    }
  }

  if (instr->NeedsWriteBarrier()) {
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoStoreNamedGeneric(HStoreNamedGeneric* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoStringAdd(HStringAdd* instr) {
//  llvm::Value* context = llvm_ir_builder_->CreateLoad(instr->context()->llvm_value(), "RSI");
  // see GetContext()!
  StringAddStub stub(isolate(),
                     instr->flags(),
                     instr->pretenure_flag());

  //llvm::Function* callStrAdd = llvm::Function::Create(&LCodeGen::CallCode, llvm::Function::ExternalLinkage );
  //LCodeGen(NULL, NULL, NULL).CallCode(stub.GetCode(), RelocInfo::CODE_TARGET, NULL);
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoStringCharCodeAt(HStringCharCodeAt* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoStringCharFromCode(HStringCharFromCode* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoStringCompareAndBranch(HStringCompareAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoSub(HSub* instr) {
  if(instr->representation().IsInteger32() || instr->representation().IsSmi()) {
    DCHECK(instr->left()->representation().Equals(instr->representation()));
    DCHECK(instr->right()->representation().Equals(instr->representation()));
    HValue* left = instr->left();
    HValue* right = instr->right();
    llvm::Value* Sub = llvm_ir_builder_->CreateSub(Use(left), Use(right), "");
    instr->set_llvm_value(Sub);
  } else if (instr->representation().IsDouble()) {
    DCHECK(instr->representation().IsDouble());
    DCHECK(instr->left()->representation().IsDouble());
    DCHECK(instr->right()->representation().IsDouble());
    HValue* left = instr->left();
    HValue* right = instr->right();
    llvm::Value* fSub =  llvm_ir_builder_->CreateFSub(Use(left), Use(right), "");
    instr->set_llvm_value(fSub);  
   }
  else {
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoTailCallThroughMegamorphicCache(HTailCallThroughMegamorphicCache* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoThisFunction(HThisFunction* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoToFastProperties(HToFastProperties* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoTransitionElementsKind(HTransitionElementsKind* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoTrapAllocationMemento(HTrapAllocationMemento* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoTypeof(HTypeof* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoTypeofIsAndBranch(HTypeofIsAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoUnaryMathOperation(HUnaryMathOperation* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoUnknownOSRValue(HUnknownOSRValue* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoUseConst(HUseConst* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoWrapReceiver(HWrapReceiver* instr) {
  UNIMPLEMENTED();
}

}  // namespace internal
}  // namespace v8
