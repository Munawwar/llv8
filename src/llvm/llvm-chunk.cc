// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <cstdio>
#include "src/code-factory.h"
#include "src/disassembler.h"
#include "llvm-chunk.h"
#include "llvm-passes.h"
#include <llvm/IR/InlineAsm.h>
#include "llvm-stackmaps.h"

namespace v8 {
namespace internal {

#define __ llvm_ir_builder_->

auto LLVMGranularity::x64_target_triple = "x86_64-unknown-linux-gnu";
llvm::Type* Types::i8 = nullptr;
llvm::Type* Types::i32 = nullptr;
llvm::Type* Types::i64 = nullptr;
llvm::Type* Types::float64 = nullptr;
llvm::PointerType* Types::ptr_i8 = nullptr;
llvm::PointerType* Types::ptr_i32 = nullptr;
llvm::PointerType* Types::ptr_i64 = nullptr;
llvm::PointerType* Types::ptr_float64 = nullptr;
llvm::Type* Types::tagged = nullptr;
llvm::PointerType* Types::ptr_tagged = nullptr;

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
                                 Translation* translation,
                                 const StackMaps& stackmaps) {
  if (environment == nullptr) return;

  // The translation includes one command per value in the environment.
  int translation_size = environment->translation_size();
  // The output frame height does not include the parameters.
  int height = translation_size - environment->parameter_count();

  WriteTranslation(environment->outer(), stackmap, translation, stackmaps);
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
                     stackmaps,
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
                                 const StackMaps& stackmaps,
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
    Register reg = location.dwarf_reg.reg().IntReg();
    if (!reg.is(rbp)) UNIMPLEMENTED();
    auto offset = location.offset;
    DCHECK(offset % kInt32Size == 0);
    // TODO(llvm): check for off-by-one error in int32 case on real deopts.
    auto index = offset / kPointerSize;
    CHECK(index != 1 && index != 0); // rbp and return address
    if (index >= 0)
      index = 1 - index;
    else {
      index = -index -
        (StandardFrameConstants::kFixedFrameSize / kPointerSize - 1);
    }
    if (is_tagged) {
      DCHECK(location.size == kPointerSize);
      translation->StoreStackSlot(index);
    } else if (is_uint32) {
      DCHECK(location.size == kInt32Size);
      translation->StoreUint32StackSlot(index);
    } else {
      DCHECK(location.size == kInt32Size);
      translation->StoreInt32StackSlot(index);
    }
  } else if (location.kind == StackMaps::Location::kRegister) {
    StackMapReg stack_reg = location.dwarf_reg.reg();
    if (stack_reg.IsIntReg()) {
      Register reg = stack_reg.IntReg();
      if (is_tagged) {
        translation->StoreRegister(reg);
      } else if (is_uint32) {
        translation->StoreUint32Register(reg);
      } else {
        translation->StoreInt32Register(reg);
      }
    } else if (stack_reg.IsDoubleReg()) {
      XMMRegister reg = stack_reg.XMMReg();
      translation->StoreDoubleRegister(reg);
    } else {
      UNIMPLEMENTED();
    }
  } else if (location.kind == StackMaps::Location::kConstantIndex) {
    // FIXME(llvm): We assume large constant is a heap object address
    // this block has not really been thoroughly tested
    auto value = stackmaps.constants[location.offset].integer;

    if (reloc_data_->reloc_map().count(value)) {
      auto pair = reloc_data_->reloc_map()[value];
      LLVMRelocationData::ExtendedInfo minfo = pair.second;
      if (minfo.cell_extended) {
        DCHECK((value & 0xffffffff) == LLVMChunkBuilder::kExtFillingValue);
        value >>= 32;
      }
    }
    Handle<Object> const_obj =  bit_cast<Handle<HeapObject> >(
        static_cast<intptr_t>(value));
    int literal_id = deopt_data_->DefineDeoptimizationLiteral(const_obj);
    translation->StoreLiteral(literal_id);
  } else if (location.kind == StackMaps::Location::kConstant) {
    int literal_id = deopt_data_->DefineDeoptimizationLiteral(
        isolate()->factory()->NewNumberFromInt(location.offset, TENURED));
    translation->StoreLiteral(literal_id);
  } else {
    UNREACHABLE();
  }
}

int LLVMChunk::WriteTranslationFor(LLVMEnvironment* env,
                                   StackMaps::Record& stackmap,
                                   const StackMaps& stackmaps) {
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
  WriteTranslation(env, stackmap, &translation, stackmaps);
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
#ifdef DEBUG
  stackmaps.dumpMultiline(std::cerr, "  ");
#endif

  uint64_t address = LLVMGranularity::getInstance().GetFunctionAddress(
      llvm_function_id_);
  auto it = std::find_if(stackmaps.stack_sizes.begin(),
                         stackmaps.stack_sizes.end(),
                         [address](const StackMaps::StackSize& s) {
                           return s.functionOffset ==  address;
                         });
  DCHECK(it != std::end(stackmaps.stack_sizes));
  DCHECK(it->size / kStackSlotSize - kPhonySpillCount >= 0);
  code->set_stack_slots(it->size / kStackSlotSize - kPhonySpillCount);

  auto true_deopt_count = stackmaps.records.size();
  Handle<DeoptimizationInputData> data =
      DeoptimizationInputData::New(isolate(), true_deopt_count, TENURED);

  if (true_deopt_count == 0) return;

  std::vector<uint32_t> sorted_ids;
  for (auto i = 0; i < true_deopt_count; i++)
    sorted_ids.push_back(stackmaps.records[i].patchpointID);
  std::sort(sorted_ids.begin(), sorted_ids.end());

  for (auto i = 0; i < true_deopt_count; i++) {
    auto stackmap_record = stackmaps.records[i];
    auto stackmap_id = stackmap_record.patchpointID;

    // stackmap_id s are unique so we'll find exactly one.
    auto it = std::lower_bound(sorted_ids.begin(),
                               sorted_ids.end(),
                               stackmap_id);

    // It's important. It seems something expects deopt entries to be stored
    // is the same order they were added.
    int deopt_entry_number = it - sorted_ids.begin();
    // The corresponding Environment is stored in the array by index = id.
    LLVMEnvironment* env = deopt_data_->deoptimizations()[stackmap_id];
    int translation_index = WriteTranslationFor(env,
                                                stackmap_record,
                                                stackmaps);
    data->SetAstId(deopt_entry_number, env->ast_id());
    data->SetTranslationIndex(deopt_entry_number,
                              Smi::FromInt(translation_index));
    data->SetArgumentsStackHeight(deopt_entry_number,
                                  Smi::FromInt(env->arguments_stack_height()));
    // pc offset can be obtained from the stackmap TODO(llvm):
    // but we do not support lazy deopt yet (and for eager it should be -1)
    data->SetPc(deopt_entry_number, Smi::FromInt(-1));
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

    if (s == llvm::MCDisassembler::Fail) break;

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
  llvm_ir_builder_ = llvm::make_unique<llvm::IRBuilder<>>(
      LLVMGranularity::getInstance().context());
  Types::Init(llvm_ir_builder_.get());
  status_ = BUILDING;

//  // If compiling for OSR, reserve space for the unoptimized frame,
//  // which will be subsumed into this frame.
//  if (graph()->has_osr()) {
//    for (int i = graph()->osr()->UnoptimizedFrameSlots(); i > 0; i--) {
//      chunk()->GetNextSpillIndex(GENERAL_REGISTERS);
//    }
//  }

  // First param is context (v8, js context) which goes to rsi,
  // second param is the callee's JSFunction object (rdi),
  // third param is Parameter 0 which is I am not sure what
  int num_parameters = info()->num_parameters() + 3;

  std::vector<llvm::Type*> params(num_parameters, nullptr);
  for (auto i = 0; i < num_parameters; i++) {
    // For now everything is Int64. Probably it is even right for x64.
    // So in that case we are going to do come casts AFAIK
    params[i] = Types::i64;
  }
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      Types::i64, params, false);

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
  CHECK(pending_pushed_args_.is_empty());
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

llvm::Type* LLVMChunkBuilder::GetLLVMType(Representation r) {
  switch (r.kind()) {
    case Representation::Kind::kInteger32:
      return Types::i32;
    case Representation::Kind::kTagged:
    case Representation::Kind::kExternal: // For now.
      return Types::tagged;
    case Representation::Kind::kSmi:
      return Types::i64;
    case Representation::Kind::kDouble:
      return Types::float64;
    default:
      UNIMPLEMENTED();
      return nullptr;
  }
}

void LLVMChunkBuilder::DoDummyUse(HInstruction* instr) {
  Representation r = instr->representation();
  llvm::Type* type = GetLLVMType(r);
  auto dummy_constant = __ getInt64(0xdead);
  auto casted_dummy_constant = __ CreateBitOrPointerCast(dummy_constant, type);
  for (int i = 1; i < instr->OperandCount(); ++i) {
    if (instr->OperandAt(i)->IsControlInstruction()) continue;
    Use(instr->OperandAt(i)); // Visit all operands and dummy-use them as well.
  }
  instr->set_llvm_value(casted_dummy_constant);
}

void LLVMChunkBuilder::VisitInstruction(HInstruction* current) {
  HInstruction* old_current = current_instruction_;
  current_instruction_ = current;

  if (current->CanReplaceWithDummyUses()) {
    DoDummyUse(current);
  } else {
    HBasicBlock* successor;
    if (current->IsControlInstruction() &&
        HControlInstruction::cast(current)->KnownSuccessorBlock(&successor) &&
        successor != NULL) {
      __ CreateBr(Use(successor)); // Goto(successor)
    } else {
      current->CompileToLLVM(this); // the meat
    }
  }

//  argument_count_ += current->argument_delta();
//  DCHECK(argument_count_ >= 0);

  current_instruction_ = old_current;
}

llvm::BasicBlock* LLVMChunkBuilder::NewBlock(const char* name) {
  LLVMContext& llvm_context = LLVMGranularity::getInstance().context();
  return llvm::BasicBlock::Create(llvm_context, name, function_);
}

llvm::BasicBlock* LLVMChunkBuilder::Use(HBasicBlock* block) {
  if (!block->llvm_start_basic_block()) {
    llvm::BasicBlock* llvm_block = NewBlock("BlockEntry");
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
  DCHECK_EQ(value->llvm_value()->getType(),
            GetLLVMType(value->representation()));
  return value->llvm_value();
}

llvm::Value* LLVMChunkBuilder::SmiToInteger32(HValue* value) {
  llvm::Value* res = nullptr;
  if (SmiValuesAre32Bits()) {
    res = __ CreateLShr(Use(value), kSmiShift);
    res = __ CreateTrunc(res, Types::i32);
  } else {
    DCHECK(SmiValuesAre31Bits());
    UNIMPLEMENTED();
    // TODO(llvm): just implement sarl(dst, Immediate(kSmiShift));
  }
  return res;
}

llvm::Value* LLVMChunkBuilder::SmiCheck(llvm::Value* value, bool negate) {
  llvm::Value* res = __ CreateAnd(value, __ getInt64(1));
  return __ CreateICmp(negate ? llvm::CmpInst::ICMP_NE : llvm::CmpInst::ICMP_EQ,
      res, __ getInt64(0));
}

void LLVMChunkBuilder::Assert(llvm::Value* condition) {
  auto cont = NewBlock("After assertion");
  auto fail = NewBlock("Fail assertion");
  __ CreateCondBr(condition, cont, fail);
  __ SetInsertPoint(fail);
  llvm::Function* debug_trap = llvm::Intrinsic::getDeclaration(
      module_.get(), llvm::Intrinsic::debugtrap);
  __ CreateCall(debug_trap);
  __ CreateUnreachable();
  __ SetInsertPoint(cont);
}

void LLVMChunkBuilder::IncrementCounter(StatsCounter* counter, int value) {
  DCHECK(value != 0);
  if (!FLAG_native_code_counters || !counter->Enabled()) return;
  Address conter_addr = ExternalReference(counter).address();
  auto llvm_counter_addr = __ getInt64(reinterpret_cast<uint64_t>(conter_addr));
  auto casted_address = __ CreateIntToPtr(llvm_counter_addr, Types::ptr_i32);
  auto llvm_conunter = __ CreateLoad(casted_address);
  auto llvm_value = __ getInt32(value);
  auto updated_value = __ CreateAdd(llvm_conunter, llvm_value);
  __ CreateStore(updated_value, casted_address);
}

void LLVMChunkBuilder::AssertSmi(llvm::Value* value, bool assert_not_smi) {
  if (!emit_debug_code()) return;

  auto check = SmiCheck(value, assert_not_smi);
  Assert(check);
}

void LLVMChunkBuilder::AssertNotSmi(llvm::Value* value) {
  bool assert_not_smi = true;
  return AssertSmi(value, assert_not_smi);
}

llvm::Value* LLVMChunkBuilder::Integer32ToSmi(HValue* value) {
  llvm::Value* int32_val = Use(value);
  llvm::Value* extended_width_val = __ CreateZExt(int32_val, Types::i64);
  return __ CreateShl(extended_width_val, kSmiShift);
}

llvm::Value* LLVMChunkBuilder::Integer32ToSmi(llvm::Value* value) {
  llvm::Value* extended_width_val = __ CreateZExt(value, Types::i64);
  return __ CreateShl(extended_width_val, kSmiShift);
}


llvm::Value* LLVMChunkBuilder::CallVoid(Address target) {
  llvm::Value* target_adderss = __ getInt64(reinterpret_cast<uint64_t>(target));
  bool is_var_arg = false;
  llvm::FunctionType* function_type = llvm::FunctionType::get(__ getVoidTy(),
                                                              is_var_arg);
  llvm::PointerType* ptr_to_function = function_type->getPointerTo();
  llvm::Value* casted = __ CreateIntToPtr(target_adderss, ptr_to_function);
  return __ CreateCall(casted,  llvm::ArrayRef<llvm::Value*>());
}

llvm::Value* LLVMChunkBuilder::CallAddressForMathPow(Address target,
                                           llvm::CallingConv::ID calling_conv,
                                           std::vector<llvm::Value*>& params) {
  llvm::Value* target_adderss = __ getInt64(reinterpret_cast<uint64_t>(target));
  bool is_var_arg = false;

  auto return_type = Types::float64;
  std::vector<llvm::Type*> param_types;
  param_types.push_back(params[0]->getType());
  param_types.push_back(params[1]->getType());
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      return_type, param_types, is_var_arg);
  llvm::PointerType* ptr_to_function = function_type->getPointerTo();

  llvm::Value* casted = __ CreateIntToPtr(target_adderss, ptr_to_function);
  llvm::CallInst* call_inst = __ CreateCall(casted, params);
  call_inst->setCallingConv(calling_conv);

  return call_inst;
}

llvm::Value* LLVMChunkBuilder::CallAddress(Address target,
                                           llvm::CallingConv::ID calling_conv,
                                           std::vector<llvm::Value*>& params) {
  llvm::Value* target_adderss = __ getInt64(reinterpret_cast<uint64_t>(target));
  bool is_var_arg = false;

  // Tagged return type won't hurt even if in fact it's void
  auto return_type = Types::ptr_i8; // TODO(llvm): set tagged, check tests
  auto param_type = Types::tagged;
  std::vector<llvm::Type*> param_types(params.size(), param_type);
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      return_type, param_types, is_var_arg);
  llvm::PointerType* ptr_to_function = function_type->getPointerTo();

  llvm::Value* casted = __ CreateIntToPtr(target_adderss, ptr_to_function);
  llvm::CallInst* call_inst = __ CreateCall(casted, params);
  call_inst->setCallingConv(calling_conv);

  return call_inst;
}

llvm::Value* LLVMChunkBuilder::FieldOperand(llvm::Value* base, int offset) {
  llvm::Value* offset_val = __ getInt64(offset - kHeapObjectTag);
  // I don't know why, but it works OK even if base was already an i8*
  llvm::Value* base_casted = __ CreateIntToPtr(base, Types::ptr_i8);
  return __ CreateGEP(base_casted, offset_val);
}

// TODO(llvm): It should probably become 'load field operand as type'
// with tagged as default.
llvm::Value* LLVMChunkBuilder::LoadFieldOperand(llvm::Value* base, int offset) {
  llvm::Value* address = FieldOperand(base, offset);
  llvm::Value* casted_address = __ CreatePointerCast(address,
                                                     Types::ptr_tagged);
  return __ CreateLoad(casted_address);
}

llvm::Value* LLVMChunkBuilder::ConstructAddress(llvm::Value* base, int offset) {
    llvm::Value* offset_val = __ getInt64(offset);
    llvm::Value* base_casted = __ CreateIntToPtr(base, Types::ptr_i8);
    return __ CreateGEP(base_casted, offset_val);
}

llvm::Value* LLVMChunkBuilder::MoveHeapObject(Handle<Object> object) {
    if (object->IsSmi()) {
      // TODO(llvm): use/write a function for that
      Smi* smi = Smi::cast(*object);
      intptr_t intptr_value = reinterpret_cast<intptr_t>(smi);
      llvm::Value* value = __ getInt64(intptr_value);
      return value;
    } else { // Heap object
      // MacroAssembler::MoveHeapObject
      AllowHeapAllocation allow_allocation;
      AllowHandleAllocation allow_handles;
      DCHECK(object->IsHeapObject());
      if (isolate()->heap()->InNewSpace(*object)) {
        Handle<Cell> new_cell = isolate()->factory()->NewCell(object);
        llvm::Value* value = Move(new_cell, RelocInfo::CELL);
        llvm::Value* ptr = __ CreateIntToPtr(value, Types::ptr_i64);
        return  __ CreateLoad(ptr);
      } else {
        return Move(object, RelocInfo::EMBEDDED_OBJECT);
      }
    }
}

llvm::Value* LLVMChunkBuilder::Move(Handle<Object> object,
                                    RelocInfo::Mode rmode) {
  AllowDeferredHandleDereference using_raw_address;
  DCHECK(!RelocInfo::IsNone(rmode));
  DCHECK(object->IsHeapObject());
  DCHECK(!isolate()->heap()->InNewSpace(*object));

  uint64_t intptr_value = reinterpret_cast<uint64_t>(object.location());
  return RecordRelocInfo(intptr_value, rmode);
}

llvm::Value* LLVMChunkBuilder::Compare(llvm::Value* lhs, llvm::Value* rhs) {
  llvm::Value* casted_lhs = __ CreateBitOrPointerCast(lhs, Types::ptr_i8);
  llvm::Value* casted_rhs = __ CreateBitOrPointerCast(rhs, Types::ptr_i8);
  return __ CreateICmpEQ(casted_lhs, casted_rhs);
}

llvm::Value* LLVMChunkBuilder::Compare(llvm::Value* lhs, Handle<Object> rhs) {
  AllowDeferredHandleDereference smi_check;
  if (rhs->IsSmi()) {
    UNIMPLEMENTED();
    //    Cmp(dst, Smi::cast(*rhs));
    return nullptr;
  } else {
    auto type = Types::tagged;
    auto llvm_rhs = __ CreateBitOrPointerCast(MoveHeapObject(rhs), type);
    auto casted_lhs = __ CreateBitOrPointerCast(lhs, type);
    return __ CreateICmpEQ(casted_lhs, llvm_rhs);
  }
}

llvm::Value* LLVMChunkBuilder::CompareMap(llvm::Value* object,
                                          Handle<Map> map) {
  return Compare(LoadFieldOperand(object, HeapObject::kMapOffset), map);
}

llvm::Value* LLVMChunkBuilder::CheckPageFlag(llvm::Value* object, int mask) {
  auto page_align_mask = __ getInt64(~Page::kPageAlignmentMask);
  // TODO(llvm): do the types match?
  auto masked_object = __ CreateAnd(object, page_align_mask, "CheckPageFlag1");
  auto flags_address = ConstructAddress(masked_object,
                                        MemoryChunk::kFlagsOffset);
  auto i32_ptr_flags_address = __ CreateBitCast(flags_address, Types::ptr_i32);
  auto flags = __ CreateLoad(i32_ptr_flags_address);
  auto llvm_mask = __ getInt32(mask);
  auto and_result = __ CreateAnd(flags, llvm_mask);
  return __ CreateICmpEQ(and_result, __ getInt32(0), "CheckPageFlag");
}

llvm::Value* LLVMChunkBuilder::AllocateHeapNumber() {
  // FIXME(llvm): if FLAG_inline_new is set (which is the default)
  // fast inline allocation should be used
  // (otherwise runtime stub call should be performed).

  CHECK(!FLAG_inline_new);

  // return an i8*
  llvm::Value* allocated = CallRuntimeViaId(Runtime::kAllocateHeapNumber);
  // RecordSafepointWithRegisters...
  return allocated;
}

llvm::Value* LLVMChunkBuilder::CallRuntimeViaId(Runtime::FunctionId id) {
  return CallRuntime(Runtime::FunctionForId(id));
}

llvm::Value* LLVMChunkBuilder::CallRuntime(const Runtime::Function* function) {
  auto arg_count = function->nargs;
  // if (arg_count != 0) UNIMPLEMENTED();

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

  llvm::Value* target_address = __ getInt64(
      reinterpret_cast<uint64_t>(code->instruction_start()));

  std::vector<llvm::Type*> param_types(arg_count + 3, nullptr);
  // First 3 types are Types::i64, Types::ptr_i8, Types::i64. The rest is tagged
  for (auto i = 0; i < arg_count + 3; i++)
    param_types[i] = Types::i64;
  param_types[1] = Types::ptr_i8; // Do not change order.
  bool is_var_arg = false;

  llvm::FunctionType* function_type = llvm::FunctionType::get(
      Types::ptr_i8, param_types, is_var_arg);
  llvm::PointerType* ptr_to_function = function_type->getPointerTo();
  llvm::Value* casted = __ CreateIntToPtr(target_address, ptr_to_function);

  // FIXME Dirty hack. We need to find way to push arguments in stack instead of moving them
  // It will also fix arguments offset mismatch problem in runtime functions
  std::string arg_offset = std::to_string(arg_count * kPointerSize);
  std::string asm_string1 = "sub $$";
  std::string asm_string2 = ", %rsp";
  std::string final_strig = asm_string1 + arg_offset + asm_string2;
  llvm::FunctionType* inl_asm_f_type = llvm::FunctionType::get(__ getVoidTy(),
                                                               false);
  llvm::InlineAsm* inline_asm = llvm::InlineAsm::get(
      inl_asm_f_type, final_strig, "~{dirflag},~{fpsr},~{flags}", true);
  __ CreateCall(inline_asm, "");

  auto llvm_nargs = __ getInt64(arg_count);
  auto target_temp = __ getInt64(reinterpret_cast<uint64_t>(rt_target));
  auto llvm_rt_target = __ CreateIntToPtr(target_temp, Types::ptr_i8);
  auto context = GetContext();
  std::vector<llvm::Value*> args(arg_count + 3, nullptr);
  args[0] = llvm_nargs;
  args[1] = llvm_rt_target;
  args[2] = context;

  for (int i = 0; i < pending_pushed_args_.length(); i++) {
    args[arg_count + 3 - 1 - i] = pending_pushed_args_[i];
  }
  pending_pushed_args_.Clear();

  llvm::CallInst* call_inst = __ CreateCall(casted, args);
  call_inst->setCallingConv(llvm::CallingConv::X86_64_V8_CES);
  // return value has type i8*
  return call_inst;
}

llvm::Value* LLVMChunkBuilder::CallRuntimeFromDeferred(Runtime::FunctionId id,
    llvm::Value* context, std::vector<llvm::Value*> params) {
  const Runtime::Function* function = Runtime::FunctionForId(id);
  auto arg_count = function->nargs;

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

  llvm::Value* target_address = __ getInt64(
      reinterpret_cast<uint64_t>(code->instruction_start()));
  bool is_var_arg = false;
  std::vector<llvm::Type*> pTypes;
  LLVMContext& lcontext = LLVMGranularity::getInstance().context();
  std::vector<llvm::Type*>FuncTy_3_args;
  llvm::FunctionType* FuncTy_3 = llvm::FunctionType::get(
       llvm::Type::getVoidTy(lcontext), FuncTy_3_args, false);
  pTypes.push_back(Types::i64);
  pTypes.push_back(Types::ptr_i8);
  pTypes.push_back(Types::i64);
  for (auto i = 0; i < params.size(); ++i) 
     pTypes.push_back(params[i]->getType());
  llvm::ArrayRef<llvm::Type*> pRef (pTypes);
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      Types::ptr_i8, pRef, is_var_arg);
  llvm::PointerType* ptr_to_function = function_type->getPointerTo();
  llvm::Value* casted = __ CreateIntToPtr(target_address, ptr_to_function);
  // FIXME Dirty hack. We need to find way to push arguments in stack instead of moving them
  // It will also fix arguments offset mismatch problem in runtime functions
  std::string argOffset = std::to_string(arg_count * 8);
  std::string asm_string1 = "sub $$";
  std::string asm_string2 = ", %rsp";
  std::string final_strig = asm_string1 + argOffset+asm_string2;
  llvm::InlineAsm* ptr_121 = llvm::InlineAsm::get(FuncTy_3, final_strig, "~{dirflag},~{fpsr},~{flags}",true);
  llvm::CallInst* void_111 = __ CreateCall(ptr_121, "");
  void_111->setCallingConv(llvm::CallingConv::C);
  auto llvm_nargs = __ getInt64(arg_count);
  auto target_temp = __ getInt64(reinterpret_cast<uint64_t>(rt_target));
  auto llvm_rt_target = __ CreateIntToPtr(target_temp, Types::ptr_i8);
  std::vector<llvm::Value*> actualParams;
  actualParams.push_back(llvm_nargs);
  actualParams.push_back(llvm_rt_target);
  actualParams.push_back(context);
  for (auto i = 0; i < params.size(); ++i)
     actualParams.push_back(params[i]);
  llvm::ArrayRef<llvm::Value*> args (actualParams);
  llvm::CallInst* call_inst = __ CreateCall(casted, args );
  call_inst->setCallingConv(llvm::CallingConv::X86_64_V8_CES);
  // FIXME Dirty hack. We need to find way to push arguments in stack instead of moving them
  // It will also fix arguments offset mismatch problem in runtime functions
  // return value has type i8*
  return call_inst;

}

llvm::Value* LLVMChunkBuilder::GetContext() {
  // First parameter is our context (rsi).
  return function_->arg_begin();
}

llvm::Value* LLVMChunkBuilder::GetNan() {
  // Is this NaN OK? :) I see it might be slow...
  // TODO(llvm): use 0/0 or llvm NaN for better performance
  return __ CreateBitCast(LoadRoot(Heap::kNanValueRootIndex), Types::float64);
  //  return __ CreateSIToFP(__ getInt64(0), Types::float64); // 0 for debug
}

LLVMEnvironment* LLVMChunkBuilder::AssignEnvironment() {
  HEnvironment* hydrogen_env = current_block_->last_environment();
  int argument_index_accumulator = 0;
  ZoneList<HValue*> objects_to_materialize(0, zone());
  return CreateEnvironment(
      hydrogen_env, &argument_index_accumulator, &objects_to_materialize);
}

void LLVMChunkBuilder::DeoptimizeIf(llvm::Value* compare,
                                    bool negate,
                                    llvm::BasicBlock* next_block) {
  LLVMEnvironment* environment = AssignEnvironment();
  deopt_data_->Add(environment);

  if (FLAG_deopt_every_n_times != 0 && !info()->IsStub()) UNIMPLEMENTED();
  if (info()->ShouldTrapOnDeopt()) {
    // Our trap on deopt does not allow to proceed to the actual deopt
    // because it gets DCE'd.
    // It could be avoided if we ever need this though.
    auto one = true;
    auto negated_condition = __ CreateXor(__ getInt1(one), compare);
    Assert(negated_condition);
  }

  Deoptimizer::BailoutType bailout_type = info()->IsStub()
      ? Deoptimizer::LAZY
      : Deoptimizer::EAGER;
  DCHECK_EQ(bailout_type, Deoptimizer::EAGER); // We don't support lazy yet.

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

  llvm::BasicBlock* saved_insert_point = __ GetInsertBlock();
  if (!next_block)
    next_block = NewBlock("BlockCont");
  llvm::BasicBlock* deopt_block = NewBlock("DeoptBlock");
  __ SetInsertPoint(deopt_block);

  std::vector<llvm::Value*> mapped_values;
  for (auto val : *environment->values())
    mapped_values.push_back(val);
  CallStackMap(deopt_data_->DeoptCount() - 1, mapped_values);

  CallVoid(entry);
  __ CreateUnreachable();

  __ SetInsertPoint(saved_insert_point);
  if (!negate)
    __ CreateCondBr(compare, deopt_block, next_block);
  else
    __ CreateCondBr(compare, next_block, deopt_block);
  __ SetInsertPoint(next_block);
}

llvm::CmpInst::Predicate LLVMChunkBuilder::TokenToPredicate(Token::Value op,
                                                            bool is_unsigned,
                                                            bool is_double) {
  llvm::CmpInst::Predicate pred = llvm::CmpInst::BAD_FCMP_PREDICATE;
  switch (op) {
    case Token::EQ:
    case Token::EQ_STRICT:
      if (is_double)
        pred = llvm::CmpInst::FCMP_OEQ;
      else
        pred = llvm::CmpInst::ICMP_EQ;
      break;
    case Token::NE:
    case Token::NE_STRICT:
      if (is_double)
        pred = llvm::CmpInst::FCMP_ONE;
      else
        pred = llvm::CmpInst::ICMP_NE;
      break;
    case Token::LT:
      if (is_double)
        pred = llvm::CmpInst::FCMP_OLT;
      else
        pred = is_unsigned ? llvm::CmpInst::ICMP_ULT : llvm::CmpInst::ICMP_SLT;
      break;
    case Token::GT:
      if (is_double)
        pred =  llvm::CmpInst::FCMP_OGT;
      else
        pred = is_unsigned ? llvm::CmpInst::ICMP_UGT : llvm::CmpInst::ICMP_SGT;
      break;
    case Token::LTE:
      if (is_double)
        pred = llvm::CmpInst::FCMP_OLE;
      else
        pred = is_unsigned ? llvm::CmpInst::ICMP_ULE : llvm::CmpInst::ICMP_SLE;
      break;
    case Token::GTE:
      if (is_double)
        pred = llvm::CmpInst::FCMP_OGE;
      else
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
  if (FLAG_phi_normalize) pass_manager.add(new NormalizePhisPass());
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

void LLVMChunkBuilder::CreateVolatileZero() {
  volatile_zero_address_ = __ CreateAlloca(Types::i64);
  bool is_volatile = true;
  __ CreateStore(__ getInt64(0), volatile_zero_address_, is_volatile);
}

llvm::Value* LLVMChunkBuilder::GetVolatileZero() {
  bool is_volatile = true;
  return __ CreateLoad(volatile_zero_address_, is_volatile, "volatile_zero");
}

void LLVMChunkBuilder::DoBasicBlock(HBasicBlock* block,
                                    HBasicBlock* next_block) {
#ifdef DEBUG
  std::cerr << __FUNCTION__ << std::endl;
#endif
  DCHECK(is_building());
  __ SetInsertPoint(Use(block));
  current_block_ = block;
  next_block_ = next_block;
  if (block->IsStartBlock()) {
    CreateVolatileZero();
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
    } else if (current->IsConstant() && current->representation().IsTagged()) {
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
  block->set_llvm_end_basic_block(__ GetInsertBlock());
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
      UNIMPLEMENTED();
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
  llvm::Type* phi_type = GetLLVMType(r);
  llvm::PHINode* llvm_phi = __ CreatePHI(phi_type, phi->OperandCount());
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
//  std::vector<llvm::Type*> types(1, __ getInt64Ty());
//
//  llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
//      llvm::Intrinsic::read_register, types);
//
//  auto metadata =
//    llvm::MDNode::get(llvm_context, llvm::MDString::get(llvm_context, "rsp"));
//  llvm::MetadataAsValue* val = llvm::MetadataAsValue::get(
//      llvm_context, metadata);
//
//  llvm::Value* rsp_value = __ CreateCall(intrinsic, val);
//
//  llvm::Value* rsp_ptr = __ CreateIntToPtr(rsp_value,
//      __ getInt64Ty()->getPointerTo());
//  llvm::Value* r13_value = __ CreateLoad(rsp_ptr);
//
//  llvm::Value* compare = __ CreateICmp(llvm::CmpInst::ICMP_ULT,
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
  mapped_values.push_back(__ getInt64(stackmap_id));
  int shadow_bytes = 0;
  mapped_values.push_back(__ getInt32(shadow_bytes));
  mapped_values.insert(mapped_values.end(), values.begin(), values.end());
  __ CreateCall(stackmap, mapped_values);
}

llvm::Value* LLVMChunkBuilder::RecordRelocInfo(uint64_t intptr_value,
                                               RelocInfo::Mode rmode) {
  bool extended = false;
  if (is_uint32(intptr_value)) {
    intptr_value = (intptr_value << 32) | kExtFillingValue;
    extended = true;
  }

  auto value = __ CreateAdd(GetVolatileZero(), __ getInt64(intptr_value));

  // Here we use the intptr_value (data) only to identify the entry in the map
  RelocInfo rinfo(rmode, intptr_value);
  LLVMRelocationData::ExtendedInfo meta_info;
  meta_info.cell_extended = extended;
  reloc_data_->Add(rinfo, meta_info);

  return value;
}

void LLVMChunkBuilder::DoConstant(HConstant* instr) {
  // Note: constants might have EmitAtUses() == true
  Representation r = instr->representation();
  if (r.IsSmi()) {
    // TODO(llvm): use/write a function for that
    // FIXME(llvm): this block was not tested
    int64_t int32_value = instr->Integer32Value();
    llvm::Value* value = __ getInt64(int32_value << (kSmiShift));
    instr->set_llvm_value(value);
  } else if (r.IsInteger32()) {
    int64_t int32_value = instr->Integer32Value();
    llvm::Value* value = __ getInt32(int32_value);
    instr->set_llvm_value(value);
  } else if (r.IsDouble()) {
    llvm::Value* value = llvm::ConstantFP::get(Types::float64,
                                               instr->DoubleValue());
    instr->set_llvm_value(value);
  } else if (r.IsExternal()) {
    Address external_address = instr->ExternalReferenceValue().address();
    // TODO(llvm): tagged type
    // TODO(llvm): RelocInfo::EXTERNAL_REFERENCE
    llvm::Value* value = __ getInt64(
        reinterpret_cast<uint64_t>(external_address));
    instr->set_llvm_value(value);
  } else if (r.IsTagged()) {
    AllowHandleAllocation allow_handle_allocation;
    AllowHeapAllocation allow_heap_allocation;
    Handle<Object> object = instr->handle(isolate());
    auto value = MoveHeapObject(object);
    instr->set_llvm_value(value);
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
    __ CreateRet(ret_val);
  } else {
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoAbnormalExit(HAbnormalExit* instr) {
  __ CreateUnreachable();
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
    llvm::Value* llvm_left = Use(left);
    llvm::Value* llvm_right = Use(right);
    if (!can_overflow) {
      llvm::Value* Add = __ CreateAdd(llvm_left, llvm_right, "");
      instr->set_llvm_value(Add);
    } else {
      auto type = instr->representation().IsSmi() ? Types::i64 : Types::i32;
      llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
          llvm::Intrinsic::sadd_with_overflow, type);

      llvm::Value* params[] = { llvm_left, llvm_right };
      llvm::Value* call = __ CreateCall(intrinsic, params);

      llvm::Value* sum = __ CreateExtractValue(call, 0);
      llvm::Value* overflow = __ CreateExtractValue(call, 1);
      instr->set_llvm_value(sum);
      DeoptimizeIf(overflow);
    }
  } else if (instr->representation().IsDouble()) {
      DCHECK(instr->left()->representation().IsDouble());
      DCHECK(instr->right()->representation().IsDouble());
      HValue* left = instr->BetterLeftOperand();
      HValue* right = instr->BetterRightOperand();
      llvm::Value* fadd = __ CreateFAdd(Use(left), Use(right));
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
  llvm::Value* value = __ getInt32(flags);
  llvm::Value* arg2 = Integer32ToSmi(value);
  args.push_back(arg2);
  args.push_back(arg1);
  llvm::Value* alloc =  CallRuntimeFromDeferred(Runtime::kAllocateInTargetSpace, Use(instr->context()), args);
  auto alloc_casted = __ CreatePtrToInt(alloc, Types::i64);
  if (instr->MustPrefillWithFiller()) {
    UNIMPLEMENTED();
  }
  instr->set_llvm_value(alloc_casted);
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
      llvm::Value* And = __ CreateAnd(Use(left), Use(right),"");
      instr->set_llvm_value(And);
      break;
    }
    case Token::BIT_OR: {
      llvm::Value* Or = __ CreateOr(Use(left), Use(right),"");
      instr->set_llvm_value(Or);
      break;
    }
    case Token::BIT_XOR: {
      if(right->IsConstant() && right_operand == int32_t(~0)) {
        llvm::Value* Not = __ CreateNot(Use(left), "");
        instr->set_llvm_value(Not);
      } else {
        llvm::Value* Xor = __ CreateXor(Use(left), Use(right), "");
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
  DCHECK(instr->HasNoUses()); // if it fails, see what llvm_value is appropriate
  Representation representation = instr->length()->representation();
  DCHECK(representation.Equals(instr->index()->representation()));
  DCHECK(representation.IsSmiOrInteger32());
  USE(representation);

  if (instr->length()->IsConstant() && instr->index()->IsConstant()) {
    auto length = instr->length()->GetInteger32Constant();
    auto index = instr->index()->GetInteger32Constant();
    // Avoid stackmap creation (happens upon DeoptimizeIf call).
    if (index < length || (instr->allow_equality() && index == length)) {
      instr->set_llvm_value(nullptr); // TODO(llvm): incorrect if instr has uses
      return;
    }
  }

  llvm::Value* length = Use(instr->length());
  llvm::Value* index = Use(instr->index());

  // FIXME(llvm): signed comparison makes sense. Or does it?
  auto cc = instr->allow_equality()
      ? llvm::CmpInst::ICMP_SLE : llvm::CmpInst::ICMP_SLT;

  llvm::Value* compare = __ CreateICmp(cc, index, length);
  if (FLAG_debug_code && instr->skip_check()) {
    UNIMPLEMENTED();
  } else {
    bool negate = true;
    DeoptimizeIf(compare, negate); // kOutOfBounds
  }
  instr->set_llvm_value(compare);
}

void LLVMChunkBuilder::DoBoundsCheckBaseIndexInformation(
    HBoundsCheckBaseIndexInformation* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::BranchTagged(HBranch* instr,
                                    ToBooleanStub::Types expected,
                                    llvm::BasicBlock* true_target,
                                    llvm::BasicBlock* false_target) {
  llvm::Value* value = Use(instr->value());

  if (expected.IsEmpty()) expected = ToBooleanStub::Types::Generic();

  std::vector<llvm::BasicBlock*> check_blocks;
  for (auto i = ToBooleanStub::UNDEFINED;
      i < ToBooleanStub::NUMBER_OF_TYPES;
      i = static_cast<ToBooleanStub::Type>(i + 1)) {
    if (expected.Contains(i))
      check_blocks.push_back(NewBlock("BranchTagged Check Block"));
  }
  llvm::BasicBlock* merge_block = NewBlock("BranchTagged Merge Block");
  check_blocks.push_back(merge_block);

  DCHECK(check_blocks.size() > 1);
  unsigned cur_block = 0;
  __ CreateBr(check_blocks[cur_block]);

  if (expected.Contains(ToBooleanStub::UNDEFINED)) {
    __ SetInsertPoint(check_blocks[cur_block]);
    // undefined -> false.
    auto is_undefined = CompareRoot(value, Heap::kUndefinedValueRootIndex);
    __ CreateCondBr(is_undefined, false_target, check_blocks[++cur_block]);
  }

  if (expected.Contains(ToBooleanStub::BOOLEAN)) {
    __ SetInsertPoint(check_blocks[cur_block]);
    // true -> true.
    auto is_true = CompareRoot(value, Heap::kTrueValueRootIndex);
    llvm::BasicBlock* bool_second = NewBlock("BranchTagged Boolean Second Check");
    __ CreateCondBr(is_true, true_target, bool_second);
    // false -> false.
    __ SetInsertPoint(bool_second);
    auto is_false = CompareRoot(value, Heap::kFalseValueRootIndex);
    __ CreateCondBr(is_false, false_target, check_blocks[++cur_block]);
  }

  if (expected.Contains(ToBooleanStub::NULL_TYPE)) {
    __ SetInsertPoint(check_blocks[cur_block]);
    // 'null' -> false.
    auto is_null = CompareRoot(value, Heap::kNullValueRootIndex);
    __ CreateCondBr(is_null, false_target, check_blocks[++cur_block]);
  }
  if (expected.Contains(ToBooleanStub::SMI)) {
    UNIMPLEMENTED();
    // Smis: 0 -> false, all other -> true.
//    __ Cmp(reg, Smi::FromInt(0));
//    __ j(equal, instr->FalseLabel(chunk()));
//    __ JumpIfSmi(reg, instr->TrueLabel(chunk()));
  } else if (expected.NeedsMap()) {
    UNIMPLEMENTED();
    // If we need a map later and have a Smi -> deopt.
//    __ testb(reg, Immediate(kSmiTagMask));
//    DeoptimizeIf(zero, instr, Deoptimizer::kSmi);
  }

  if (expected.NeedsMap()) {
    UNIMPLEMENTED();
//    __ movp(map, FieldOperand(reg, HeapObject::kMapOffset));
//
//    if (expected.CanBeUndetectable()) {
//      // Undetectable -> false.
//      __ testb(FieldOperand(map, Map::kBitFieldOffset),
//               Immediate(1 << Map::kIsUndetectable));
//      __ j(not_zero, instr->FalseLabel(chunk()));
//    }
  }

  if (expected.Contains(ToBooleanStub::SPEC_OBJECT)) {
    UNIMPLEMENTED();
//    // spec object -> true.
//    __ CmpInstanceType(map, FIRST_SPEC_OBJECT_TYPE);
//    __ j(above_equal, instr->TrueLabel(chunk()));
  }

  if (expected.Contains(ToBooleanStub::STRING)) {
    UNIMPLEMENTED();
    // String value -> false iff empty.
//    Label not_string;
//    __ CmpInstanceType(map, FIRST_NONSTRING_TYPE);
//    __ j(above_equal, &not_string, Label::kNear);
//    __ cmpp(FieldOperand(reg, String::kLengthOffset), Immediate(0));
//    __ j(not_zero, instr->TrueLabel(chunk()));
//    __ jmp(instr->FalseLabel(chunk()));
//    __ bind(&not_string);
  }

  if (expected.Contains(ToBooleanStub::SYMBOL)) {
    UNIMPLEMENTED();
//    // Symbol value -> true.
//    __ CmpInstanceType(map, SYMBOL_TYPE);
//    __ j(equal, instr->TrueLabel(chunk()));
  }

  if (expected.Contains(ToBooleanStub::HEAP_NUMBER)) {
    UNIMPLEMENTED();
//    // heap number -> false iff +0, -0, or NaN.
//    Label not_heap_number;
//    __ CompareRoot(map, Heap::kHeapNumberMapRootIndex);
//    __ j(not_equal, &not_heap_number, Label::kNear);
//    XMMRegister xmm_scratch = double_scratch0();
//    __ xorps(xmm_scratch, xmm_scratch);
//    __ ucomisd(xmm_scratch, FieldOperand(reg, HeapNumber::kValueOffset));
//    __ j(zero, instr->FalseLabel(chunk()));
//    __ jmp(instr->TrueLabel(chunk()));
//    __ bind(&not_heap_number);
  }

  __ SetInsertPoint(merge_block) ; // TODO(llvm): not sure

  if (!expected.IsGeneric()) {
    // We've seen something for the first time -> deopt.
    // This can only happen if we are not generic already.
    auto no_condition = __ getTrue();
    DeoptimizeIf(no_condition); // kUnexpectedObject

    // Since we deoptimize on True the continue block is never reached.
    __ CreateUnreachable();
  }
}

void LLVMChunkBuilder::DoBranch(HBranch* instr) {
  HValue* value = instr->value();
  llvm::BasicBlock* true_target = Use(instr->SuccessorAt(0));
  llvm::BasicBlock* false_target = Use(instr->SuccessorAt(1));
  Representation r = value->representation();
  HType type = value->type();
  USE(type);
  if (r.IsInteger32()) {
    llvm::Value* zero = __ getInt32(0);
    llvm::Value* compare = __ CreateICmpNE(Use(value), zero);
    llvm::BranchInst* branch = __ CreateCondBr(compare,
                                               true_target, false_target);
    instr->set_llvm_value(branch);
  } else if (r.IsTagged()) {
    if (type.IsBoolean() || type.IsSmi() || type.IsJSArray()
        || type.IsHeapNumber() || type.IsString()) UNIMPLEMENTED();
    else {
      ToBooleanStub::Types expected = instr->expected_input_types();
      BranchTagged(instr, expected, true_target, false_target);
    }
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

  llvm::Value* js_function_val = __ getInt64(
      reinterpret_cast<uint64_t>(js_function_addr));
  llvm::Value* js_context_ptr_val = __ getInt64(
      reinterpret_cast<uint64_t>(js_context_ptr));
  js_context_ptr_val = __ CreateIntToPtr(js_context_ptr_val, Types::ptr_i64);

  auto argument_count = instr->argument_count() + 2; // rsi, rdi

  // Construct the function type (signature)
  std::vector<llvm::Type*> params(argument_count, nullptr);
  for (auto i = 0; i < argument_count; i++)
    params[i] = Types::i64;
  bool is_var_arg = false;
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      Types::i64, params, is_var_arg);

  // Get the callee's address
  // TODO(llvm): it is a pointer, not an int64
  llvm::PointerType* ptr_to_function = function_type->getPointerTo();
  llvm::PointerType* ptr_to_ptr_to_function = ptr_to_function->getPointerTo();

  llvm::Value* target_entry_ptr_val = __ CreateIntToPtr(
     __ getInt64(reinterpret_cast<int64_t>(target_entry_ptr)),
     ptr_to_ptr_to_function);
  llvm::Value* target_entry_val =  __ CreateAlignedLoad(
      target_entry_ptr_val, 1);

  // Set up the actual arguments
  std::vector<llvm::Value*> args(argument_count, nullptr);
  // FIXME(llvm): pointers, not int64
  args[0] = __ CreateAlignedLoad(js_context_ptr_val, 1);
  args[1] = js_function_val;

  DCHECK(pending_pushed_args_.length() + 2 == argument_count);
  // The order is reverse because X86_64_V8 is not implemented quite right.
  for (int i = 0; i < pending_pushed_args_.length(); i++) {
    args[argument_count - 1 - i] = pending_pushed_args_[i];
  }
  pending_pushed_args_.Clear();

  llvm::CallInst* call = __ CreateCall(target_entry_val, args);
  call->setCallingConv(llvm::CallingConv::X86_64_V8);
  instr->set_llvm_value(call);
}

void LLVMChunkBuilder::DoCallFunction(HCallFunction* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCallNew(HCallNew* instr) {
  int arity = instr->argument_count()-1;
  llvm::Value* arity_val_ = __ getInt64(arity);
  if (arity == 0) {
    arity_val_ = __ CreateXor(arity_val_, arity_val_);
  } else if (is_uint32(arity)) {
    arity_val_ = __ getInt32(static_cast<uint32_t>(arity));
  }
  LoadRoot(Heap::kUndefinedValueRootIndex);
  CallConstructStub stub(isolate(), NO_CALL_CONSTRUCTOR_FLAGS);
  Handle<Code> code = Handle<Code>::null();
  {
    AllowHandleAllocation allow_handles;
    AllowHeapAllocation allow_heap;
    code = stub.GetCode();
    // FIXME(llvm,gc): respect reloc info mode...
  }
  std::vector<llvm::Value*> params;
  for (int i = 0; i < instr->OperandCount(); i++)
    params.push_back(Use(instr->OperandAt(i)));
  pending_pushed_args_.Clear();
  llvm::Value* call = CallAddress(code->instruction_start(),
                                  llvm::CallingConv::X86_64_V8, params);
  instr->set_llvm_value(call);
}

void LLVMChunkBuilder::DoCallNewArray(HCallNewArray* instr) {
  int arity = instr->argument_count()-1;
  llvm::Value* arity_val_ = __ getInt64(arity);
  if (arity == 0) {
    arity_val_ = __ CreateXor(arity_val_, arity_val_);
  } else if (is_uint32(arity)) {
    arity_val_ = __ getInt32(static_cast<uint32_t>(arity));
  }
  LoadRoot(Heap::kUndefinedValueRootIndex);
  ElementsKind kind = instr->elements_kind();
  AllocationSiteOverrideMode override_mode =
      (AllocationSite::GetMode(kind) == TRACK_ALLOCATION_SITE)
          ? DISABLE_ALLOCATION_SITES
          : DONT_OVERRIDE;
  if (arity == 0) {
    UNIMPLEMENTED();
  } else if (arity == 1) {
    if (IsFastPackedElementsKind(kind)) {
      UNIMPLEMENTED();
    }
    ArraySingleArgumentConstructorStub stub(isolate(), kind, override_mode);
    Handle<Code> code = Handle<Code>::null();
    {
      AllowHandleAllocation allow_handles;
      AllowHeapAllocation allow_heap;
      code = stub.GetCode();
      // FIXME(llvm,gc): respect reloc info mode...
    }
    std::vector<llvm::Value*> params;
    for (int i = 0; i < instr->OperandCount(); i++)
      params.push_back(Use(instr->OperandAt(i)));
    pending_pushed_args_.Clear();
    llvm::Value* call = CallAddress(code->instruction_start(),
                                    llvm::CallingConv::X86_64_V8, params);
    llvm::Value* return_val = __ CreatePtrToInt(call,Types::i64);
    instr->set_llvm_value(return_val);
  }
  //UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCallRuntime(HCallRuntime* instr) {
  // FIXME(llvm): use instr->save_doubles()
  llvm::Value* val = CallRuntime(instr->function());
  llvm::Value* tagged_val = __ CreateBitOrPointerCast(val, Types::tagged);
  instr->set_llvm_value(tagged_val);
  // MarkAsCall
  // RecordSafepointWithLazyDeopt
}

void LLVMChunkBuilder::DoCallStub(HCallStub* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCapturedObject(HCapturedObject* instr) {
  instr->ReplayEnvironment(current_block_->last_environment());
  // There are no real uses of a captured object.
}

void LLVMChunkBuilder::ChangeDoubleToTagged(HValue* val, HChange* instr) {
  // TODO(llvm): this case in Crankshaft utilizes deferred calling.

  DCHECK(Use(val)->getType()->isDoubleTy());
  if (FLAG_inline_new) UNIMPLEMENTED();

  llvm::Value* new_heap_number = AllocateHeapNumber(); // i8*
  auto store_address = FieldOperand(new_heap_number, HeapNumber::kValueOffset);
  llvm::Value* casted_adderss = __ CreateBitCast(store_address,
                                                 Types::ptr_tagged);
  llvm::Value* casted_val = __ CreateBitCast(Use(val), Types::tagged);
  // [(i8*)new_heap_number + offset] = val;
  __ CreateStore(casted_val, casted_adderss);

  auto new_heap_number_casted = __ CreatePtrToInt(new_heap_number,
                                                  Types::tagged);
  instr->set_llvm_value(new_heap_number_casted); // no offset

  //  TODO(llvm): AssignPointerMap(Define(result, result_temp));
}

llvm::Value* LLVMChunkBuilder::LoadRoot(Heap::RootListIndex index) {
  Address root_array_start_address =
      ExternalReference::roots_array_start(isolate()).address();
  // TODO(llvm): Move(RelocInfo::EXTERNAL_REFERENCE)
  auto int64_address =
      __ getInt64(reinterpret_cast<uint64_t>(root_array_start_address));
  int offset = index << kPointerSizeLog2;
  auto load_address = ConstructAddress(int64_address, offset);
  auto casted_load_address = __ CreateBitCast(load_address, Types::ptr_i64);
  return __ CreateLoad(casted_load_address);
}

llvm::Value* LLVMChunkBuilder::CompareRoot(llvm::Value* operand,
                                           Heap::RootListIndex index) {
  llvm::Value* root_value_by_index = LoadRoot(index);
  llvm::Value* cmp_result = __ CreateICmpEQ(operand, root_value_by_index);
  return cmp_result;
}

void LLVMChunkBuilder::ChangeDoubleToI(HValue* val, HChange* instr) {
   if (instr->CanTruncateToInt32()) {
     llvm::Value* casted_int =  __ CreateFPToSI(Use(val), Types::i64);
     // FIXME: Figure out why we need this step. Fix for bitops-nsieve-bits
     auto result = __ CreateTruncOrBitCast(casted_int, Types::i32);
     instr->set_llvm_value(result);
     //TODO: Overflow case
   } else {
     UNIMPLEMENTED();
   }
}

void LLVMChunkBuilder::ChangeTaggedToDouble(HValue* val, HChange* instr) {
  bool can_convert_undefined_to_nan =
      instr->can_convert_undefined_to_nan();
  
  bool deoptimize_on_minus_zero = instr->deoptimize_on_minus_zero();
  
  llvm::BasicBlock* is_smi = NewBlock("NUMBER_CANDIDATE_IS_SMI");
  llvm::BasicBlock* is_any_tagged = NewBlock("NUMBER_CANDIDATE_IS_ANY_TAGGED");
  llvm::BasicBlock* merge_block = NewBlock("ChangeTaggedToDouble Merge");

  llvm::Value* is_heap_number = nullptr;
  llvm::Value* loaded_double_value = nullptr;
  llvm::Value* nan_value = nullptr;
  llvm::Value* llvm_val = Use(val);
  llvm::Value* cond = SmiCheck(llvm_val);
  llvm::BasicBlock* conversion_end = nullptr;

  if (!val->representation().IsSmi()) {
    __ CreateCondBr(cond, is_smi, is_any_tagged);
    __ SetInsertPoint(is_any_tagged);

    llvm::Value* vals_map = LoadFieldOperand(llvm_val, HeapObject::kMapOffset);
    is_heap_number = CompareRoot(vals_map, Heap::kHeapNumberMapRootIndex);

    llvm::Value* value_addr = FieldOperand(llvm_val, HeapNumber::kValueOffset);
    llvm::Value* value_as_double_addr = __ CreateBitCast(value_addr,
                                                         Types::ptr_float64);

    // On x64 it is safe to load at heap number offset before evaluating the map
    // check, since all heap objects are at least two words long.
    loaded_double_value = __ CreateLoad(value_as_double_addr);

    if (can_convert_undefined_to_nan) {
      auto conversion_start = NewBlock("can_convert_undefined_to_nan "
          "conversion_start");
      __ CreateCondBr(is_heap_number, merge_block, conversion_start);

      __ SetInsertPoint(conversion_start);
      auto is_undefined = CompareRoot(llvm_val, Heap::kUndefinedValueRootIndex);
      conversion_end = NewBlock("can_convert_undefined_to_nan: getting NaN");
      bool deopt_on_not_undefined = true;
      // kNotAHeapNumberUndefined
      DeoptimizeIf(is_undefined, deopt_on_not_undefined, conversion_end);
      nan_value = GetNan();
      __ CreateBr(merge_block);
    } else {
      bool deopt_on_not_equal = true;
      DeoptimizeIf(is_heap_number, deopt_on_not_equal, merge_block);
    }

    if (deoptimize_on_minus_zero) {
      UNIMPLEMENTED();
    }
  }
  
  __ SetInsertPoint(is_smi);
  auto int32_val = SmiToInteger32(val);
  auto double_val_from_smi = __ CreateSIToFP(int32_val, Types::float64);
  __ CreateBr(merge_block);

  __ SetInsertPoint(merge_block);
  llvm::PHINode* phi = __ CreatePHI(Types::float64,
                                    2 + can_convert_undefined_to_nan);
  phi->addIncoming(loaded_double_value, is_any_tagged);
  phi->addIncoming(double_val_from_smi, is_smi);
  if (can_convert_undefined_to_nan) phi->addIncoming(nan_value, conversion_end);
  instr->set_llvm_value(phi);
}

void LLVMChunkBuilder::ChangeTaggedToISlow(HValue* val, HChange* instr) {
  llvm::Value* cond = SmiCheck(Use(val));

  llvm::BasicBlock* is_smi = NewBlock("is Smi fast case");
  llvm::BasicBlock* not_smi = NewBlock("'deferred' case");
  llvm::BasicBlock* merge_and_ret = NewBlock("merge and ret");
  llvm::BasicBlock* not_smi_merge = nullptr;

  __ CreateCondBr(cond, is_smi, not_smi);

  __ SetInsertPoint(is_smi);
  llvm::Value* relult_for_smi = SmiToInteger32(val);
  __ CreateBr(merge_and_ret);

  __ SetInsertPoint(not_smi);
  llvm::Value* relult_for_not_smi = nullptr;
  bool truncating = instr->CanTruncateToInt32();

  llvm::Value* vals_map = LoadFieldOperand(Use(val), HeapObject::kMapOffset);
  llvm::Value* cmp = CompareRoot(vals_map, Heap::kHeapNumberMapRootIndex);

  if (truncating) {
    llvm::BasicBlock* truncate_heap_number = NewBlock("TruncateHeapNumberToI");
    llvm::BasicBlock* no_heap_number = NewBlock("Not a heap number");
    llvm::BasicBlock* merge_inner = NewBlock("inner merge");

    __ CreateCondBr(cmp, truncate_heap_number, no_heap_number);

    __ SetInsertPoint(truncate_heap_number);
    llvm::Value* value_addr = FieldOperand(Use(val), HeapNumber::kValueOffset);
    // cast to ptr to double, fetch the double and convert to i32
    llvm::Value* double_addr = __ CreateBitCast(value_addr, Types::ptr_float64);
    llvm::Value* double_val = __ CreateLoad(double_addr);
    llvm::Value* truncate_heap_number_result = __ CreateFPToSI(double_val,
                                                               Types::i32);

    // FIXME(llvm): add NaN check
    // cmpq(result_reg, Immediate(1)); (MacroAssembler::TruncateHeapNumberToI)
    // And implement the slow case call (SlowTruncateToI)

    __ CreateBr(merge_inner);

    __ SetInsertPoint(no_heap_number);
    Assert(__ getFalse()); // FIXME(llvm): deal with oddballs
    __ CreateBr(merge_inner);

    __ SetInsertPoint(merge_inner);
    llvm::PHINode* phi_inner = __ CreatePHI(Types::i32, 2);
    phi_inner->addIncoming(__ getInt32(0x0badbeef), no_heap_number); // FIXME
    phi_inner->addIncoming(truncate_heap_number_result, truncate_heap_number);
    relult_for_not_smi = phi_inner;
    not_smi_merge = merge_inner;
  } else {
    bool negate = true;
    DeoptimizeIf(cmp, negate); // Deoptimizer::kNotAHeapNumber

    auto address = FieldOperand(Use(val), HeapNumber::kValueOffset);
    auto double_addr = __ CreateBitCast(address, Types::ptr_float64);
    auto double_val = __ CreateLoad(double_addr);
    // Convert the double to int32; convert it back do double and
    // see it the 2 doubles are equal and neither is a NaN.
    // If not, deopt (kLostPrecision or kNaN)
    auto int32 = __ CreateFPToSI(double_val, Types::i32);
    auto double_2 = __ CreateSIToFP(int32, Types::float64);
    auto ordered_and_equal = __ CreateFCmpOEQ(double_val, double_2);
    negate = true;
    DeoptimizeIf(ordered_and_equal, negate);
    if (instr->GetMinusZeroMode() == FAIL_ON_MINUS_ZERO) {
      llvm::BasicBlock* not_zero = NewBlock("NOT ZERO");
      llvm::BasicBlock* done = NewBlock("DONE");
      llvm::Value* zero_val = llvm::ConstantFP::get(Types::float64, 0);
      auto not_equal_ = __ CreateFCmpONE(double_val, zero_val);
      __ CreateCondBr(not_equal_, not_zero, done);
      __ SetInsertPoint(not_zero);
      llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
             llvm::Intrinsic::x86_sse2_movmsk_pd);
      llvm::Value* input_val = __ CreateSIToFP(Use(val), Types::float64);
      llvm::Value* param_vect = __ CreateVectorSplat(2, input_val);
      __ CreateInsertElement(param_vect, double_val, __ getInt32(0));
      llvm::Value* call = __ CreateCall(intrinsic, param_vect);
      __ CreateAnd(call, __ getInt32(1));
      DeoptimizeIf(not_equal_, true);
      __ CreateBr(done);
      __ SetInsertPoint(done);
    }
    relult_for_not_smi = int32;
    not_smi_merge =  __ GetInsertBlock();
  }
  __ CreateBr(merge_and_ret);

  __ SetInsertPoint(merge_and_ret);
  llvm::PHINode* phi = __ CreatePHI(Types::i32, 2);
  phi->addIncoming(relult_for_smi, is_smi);
  phi->addIncoming(relult_for_not_smi, not_smi_merge);
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
        llvm::Value* cond = SmiCheck(Use(val), not_smi);
        DeoptimizeIf(cond); // Deoptimizer::kNotASmi
      }
      instr->set_llvm_value(Use(val));
    } else {
      DCHECK(to.IsInteger32());
      if (val->type().IsSmi() || val->representation().IsSmi()) {
        // convert smi to int32, no need to perform smi check
        // lithium codegen does __ AssertSmi(input)
        instr->set_llvm_value(SmiToInteger32(val));
      } else {
        ChangeTaggedToISlow(val, instr);

// TODO(llvm): if (!val->representation().IsSmi()) result = AssignEnvironment(result);
      }
    }
  } else if (from.IsDouble()) {
      if (to.IsInteger32()) {
        ChangeDoubleToI(val, instr);
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
      if (instr->CheckFlag(HValue::kCanOverflow) &&
          instr->value()->CheckFlag(HValue::kUint32)) {
        UNIMPLEMENTED();
      }
      instr->set_llvm_value(Integer32ToSmi(val));
      if (instr->CheckFlag(HValue::kCanOverflow) &&
          !instr->value()->CheckFlag(HValue::kUint32)) { 
        UNIMPLEMENTED();
      }
    } else {
      DCHECK(to.IsDouble());
      llvm::Value* double_val = __ CreateSIToFP(Use(val), Types::float64);
      instr->set_llvm_value(double_val);
      //UNIMPLEMENTED();
    }
  }
}

void LLVMChunkBuilder::DoCheckHeapObject(HCheckHeapObject* instr) {
  if (!instr->value()->type().IsHeapObject()) {
    llvm::Value* is_smi = SmiCheck(Use(instr->value()));
    DeoptimizeIf(is_smi);
  }
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
  DCHECK(instr->HasNoUses());
  llvm::Value* val = Use(instr->value());
  llvm::BasicBlock* success = NewBlock("CheckMaps success");
  std::vector<llvm::BasicBlock*> check_blocks;
  const UniqueSet<Map>* maps = instr->maps();
  for (int i = 0; i < maps->size(); i++)
    check_blocks.push_back(NewBlock("CheckMap"));
  DCHECK(maps->size() > 0);
  __ CreateBr(check_blocks[0]);
  for (int i = 0; i < maps->size() - 1; i++) {
    Handle<Map> map = maps->at(i).handle();
    __ SetInsertPoint(check_blocks[i]);
    llvm::Value* compare = CompareMap(val, map);
    __ CreateCondBr(compare, success, check_blocks[i + 1]);
  }
  __ SetInsertPoint(check_blocks[maps->size() - 1]);
  llvm::Value* compare = CompareMap(val, maps->at(maps->size() - 1).handle());
  if (instr->HasMigrationTarget()) {
    // Call deferred.
    bool deopt_on_equal = false;
    llvm::BasicBlock* defered_block = NewBlock("CheckMaps deferred");
    __ CreateCondBr(compare, success, defered_block);
    __ SetInsertPoint(defered_block);
    DCHECK(pending_pushed_args_.is_empty());
    pending_pushed_args_.Add(Use(instr->value()), info()->zone());
    llvm::Value* result = CallRuntimeViaId(Runtime::kTryMigrateInstance);
    llvm::Value* casted = __ CreateBitOrPointerCast(result, Types::i64);
    llvm::Value* and_result = __ CreateAnd(casted, __ getInt64(kSmiTagMask));
    llvm::Value* compare_result = __ CreateICmpEQ(and_result, __ getInt64(0));
    DeoptimizeIf(compare_result, deopt_on_equal, success);
    // Don't let the success BB go stray (__ SetInsertPoint).
    
  } else {
    bool deopt_on_not_equal = true;
    // kWrongMap
    DeoptimizeIf(compare, deopt_on_not_equal, success);
  }
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

void LLVMChunkBuilder::DoCompareNumericAndBranch(
    HCompareNumericAndBranch* instr) {
  Representation r = instr->representation();
  HValue* left = instr->left();
  HValue* right = instr->right();
  DCHECK(left->representation().Equals(r));
  DCHECK(right->representation().Equals(r));
  bool is_unsigned = r.IsDouble()
      || left->CheckFlag(HInstruction::kUint32)
      || right->CheckFlag(HInstruction::kUint32);

  bool is_double = instr->representation().IsDouble();
  llvm::CmpInst::Predicate pred = TokenToPredicate(instr->token(),
                                                   is_unsigned,
                                                   is_double);
  if (r.IsSmi()) {
    UNIMPLEMENTED();
  } else if (r.IsInteger32()) {
    llvm::Value* llvm_left = Use(left);
    llvm::Value* llvm_right = Use(right);
    llvm::Value* compare = __ CreateICmp(pred, llvm_left, llvm_right);
    llvm::Value* branch = __ CreateCondBr(compare,
        Use(instr->SuccessorAt(0)), Use(instr->SuccessorAt(1)));
    instr->set_llvm_value(branch);
  } else {
    DCHECK(r.IsDouble());
    llvm::Value* llvm_left = Use(left);
    llvm::Value* llvm_right = Use(right);
    llvm::Value* compare = __ CreateFCmp(pred, llvm_left, llvm_right);
    llvm::Value* branch = __ CreateCondBr(compare,
        Use(instr->SuccessorAt(0)), Use(instr->SuccessorAt(1)));
    instr->set_llvm_value(branch);
    //FIXME: Hanlde Nan case, parity_even case
    //UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoCompareHoleAndBranch(HCompareHoleAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCompareGeneric(HCompareGeneric* instr) {
  Token::Value op = instr->token();
  Handle<Code> ic = CodeFactory::CompareIC(isolate(), op).code();

  auto context = Use(instr->context());
  auto left = Use(instr->left());
  auto right = Use(instr->right());
  std::vector<llvm::Value*> params = { context, left, right };
  auto result = CallAddress(ic->instruction_start(),
                            llvm::CallingConv::C,
                            params);
  // Lithium comparison is a little strange, I think mine is all right.
  auto compare_result = __ CreateICmpNE(result, __ getInt64(0));
  auto compare_true = NewBlock("generic comparison true");
  auto compare_false = NewBlock("generic comparison false");
  llvm::Value* true_value = nullptr;
  llvm::Value* false_value = nullptr;
  auto merge = NewBlock("generic comparison merge");
  __ CreateCondBr(compare_result, compare_true, compare_false);

  __ SetInsertPoint(compare_true);
  true_value = LoadRoot(Heap::kTrueValueRootIndex);
  __ CreateBr(merge);

  __ SetInsertPoint(compare_false);
  false_value = LoadRoot(Heap::kFalseValueRootIndex);
  __ CreateBr(merge);

  __ SetInsertPoint(merge);
  auto phi = __ CreatePHI(Types::tagged, 2);
  phi->addIncoming(true_value, compare_true);
  phi->addIncoming(false_value, compare_false);
  instr->set_llvm_value(phi);
  UNIMPLEMENTED(); // calling convention should be v8_ic
}

void LLVMChunkBuilder::DoCompareMinusZeroAndBranch(HCompareMinusZeroAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCompareObjectEqAndBranch(HCompareObjectEqAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCompareMap(HCompareMap* instr) {
   auto compare = CompareMap(Use(instr->value()), instr->map().handle());
   llvm::BranchInst* branch = __ CreateCondBr(compare,
         Use(instr->SuccessorAt(0)), Use(instr->SuccessorAt(1)));
   instr->set_llvm_value(branch);
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
  Deoptimizer::BailoutType type = instr->type();
  // TODO(danno): Stubs expect all deopts to be lazy for historical reasons (the
  // needed return address), even though the implementation of LAZY and EAGER is
  // now identical. When LAZY is eventually completely folded into EAGER, remove
  // the special case below.
  if (info()->IsStub() && type == Deoptimizer::EAGER) {
    type = Deoptimizer::LAZY;
    UNIMPLEMENTED();
  }
  // we don't support lazy yet, since we have no test cases
  DCHECK(type == Deoptimizer::EAGER);
  auto reason = instr->reason();
  USE(reason);
  bool negate_condition = false;
  // It's unreacheable, but we don't care. We need it so that DeoptimizeIf()
  // does not create a new basic block which ends up unterminated.
  auto next_block = Use(instr->SuccessorAt(0));
  DeoptimizeIf(__ getTrue(), negate_condition, next_block);

}

void LLVMChunkBuilder::DoDiv(HDiv* instr) {
  if(instr->representation().IsInteger32() || instr->representation().IsSmi()) {
    DCHECK(instr->left()->representation().Equals(instr->representation()));
    DCHECK(instr->right()->representation().Equals(instr->representation()));
    HValue* dividend = instr->left();
    HValue* divisor = instr->right();
    llvm::Value* Div = __ CreateUDiv(Use(dividend), Use(divisor),"");
    instr->set_llvm_value(Div);
  } else if (instr->representation().IsDouble()) {
    DCHECK(instr->representation().IsDouble());
    DCHECK(instr->left()->representation().IsDouble());
    DCHECK(instr->right()->representation().IsDouble());
    HValue* left = instr->left();
    HValue* right = instr->right();
    llvm::Value* fDiv =  __ CreateFDiv(Use(left), Use(right), "");
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
  if(instr->offset()->IsConstant()) {
    uint32_t offset = (HConstant::cast(instr->offset()))->Integer32Value();
    llvm::Value* gep = ConstructAddress(Use(instr->base_object()), offset);
    auto result = __ CreatePtrToInt(gep, Types::i64);
    instr->set_llvm_value(result);
    //UNIMPLEMENTED();
  } else {
    UNIMPLEMENTED();
  }
  //UNIMPLEMENTED();
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
  int64_t value = reinterpret_cast<int64_t>(*(handle_value.location()));
  // TODO(llvm): RelocInfo::CELL Shall we?
  auto address_val = __ getInt64(value);
  auto gep = FieldOperand(address_val, 8);
  llvm::Value* casted_address = __ CreateBitCast(gep, Types::ptr_i64);
  llvm::Value* load_cell = __ CreateLoad(casted_address);
  instr->set_llvm_value(load_cell);
  if(instr->RequiresHoleCheck()){
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoLoadGlobalGeneric(HLoadGlobalGeneric* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoLoadKeyed(HLoadKeyed* instr) {
  if (instr->is_typed_elements()) {
    //DoLoadKeyedExternalArray(instr);
    UNIMPLEMENTED();
  } else if (instr->representation().IsDouble()) {
    //DoLoadKeyedFixedDoubleArray(instr);
    DoLoadKeyedFixedDoubleArray(instr);
  } else {
    DoLoadKeyedFixedArray(instr);
  }
}

void LLVMChunkBuilder::DoLoadKeyedFixedDoubleArray(HLoadKeyed* instr) {
  HValue* key = instr->key();
  int shift_size = ElementsKindToShiftSize(FAST_DOUBLE_ELEMENTS);
  uint32_t inst_offset = instr->base_offset();
  llvm::Value* gep_0 = nullptr;
  llvm::Value* casted_address = nullptr;
  if (kPointerSize == kInt32Size && !key->IsConstant() &&
      instr->IsDehoisted()) {
    UNIMPLEMENTED();
  }
  if (instr->RequiresHoleCheck()) {
    UNIMPLEMENTED();
  }
  if (key->IsConstant()) {
    uint32_t const_val = (HConstant::cast(key))->Integer32Value();
    gep_0 = ConstructAddress(Use(instr->elements()), (const_val << shift_size) + inst_offset);
  } else {
     llvm::Value* lkey = Use(key);
     llvm::Value* scale = nullptr;
     llvm::Value* offset = nullptr;
     if (key->representation().IsInteger32()) {
       scale = __ getInt32(8);
       offset = __ getInt32(inst_offset);
     } else {
       scale = __ getInt64(8);
       offset = __ getInt64(inst_offset);
     }
     llvm::Value* mul = __ CreateMul(lkey, scale);
     llvm::Value* add = __ CreateAdd(mul, offset);
     llvm::Value* int_ptr = __ CreateIntToPtr(Use(instr->elements()),
                                              Types::ptr_i8);
     gep_0 = __ CreateGEP(int_ptr, add);
  }
  casted_address = __ CreateBitCast(gep_0, Types::ptr_float64);
  llvm::Value* load = __ CreateLoad(casted_address);
  instr->set_llvm_value(load);
}

void LLVMChunkBuilder::DoLoadKeyedFixedArray(HLoadKeyed* instr) {
  HValue* key = instr->key();
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
    gep_0 = ConstructAddress(Use(instr->elements()), (const_val << shift_size) + inst_offset); 
  } else {
     llvm::Value* lkey = Use(key);
     llvm::Value* scale = nullptr;
     llvm::Value* offset = nullptr;
     if (key->representation().IsInteger32()) {
       scale = __ getInt32(8);
       offset = __ getInt32(inst_offset);
     } else {
       scale = __ getInt64(8);
       offset = __ getInt64(inst_offset);
     }
     llvm::Value* mul = __ CreateMul(lkey, scale);
     llvm::Value* add = __ CreateAdd(mul, offset);
     llvm::Value* int_ptr = __ CreateIntToPtr(Use(instr->elements()),
                                              Types::ptr_i8);
     gep_0 = __ CreateGEP(int_ptr, add); 
  }
  llvm::Value* casted_address = nullptr;
  if (instr->representation().IsInteger32()) {
    casted_address = __ CreateBitCast(gep_0, Types::ptr_i32);
  } else {
    casted_address = __ CreateBitCast(gep_0, Types::ptr_i64);
  }
  llvm::Value* load = __ CreateLoad(casted_address);
  if (requires_hole_check) {
    if (IsFastSmiElementsKind(instr->elements_kind())) {
      UNIMPLEMENTED();
    } else {
      // FIXME(access-nsieve): not tested
      llvm::Value* cmp = CompareRoot(load, Heap::kTheHoleValueRootIndex);
      DeoptimizeIf(cmp); // kHole
    }
  }
  instr->set_llvm_value(load);
}

void LLVMChunkBuilder::DoLoadKeyedGeneric(HLoadKeyedGeneric* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoLoadNamedField(HLoadNamedField* instr) {

  HObjectAccess access = instr->access();
  int offset = access.offset();
  if (access.IsExternalMemory()) {
    UNIMPLEMENTED();
  }

  if (instr->representation().IsDouble()){
    llvm::Value* address = FieldOperand(Use(instr->object()), offset);
    llvm::Value* cast_double = __ CreateBitCast(address, Types::ptr_float64);
    llvm::Value* result = __ CreateLoad(cast_double);
    instr->set_llvm_value(result);
    return;
  }

  if(!access.IsInobject()) {
    UNIMPLEMENTED();
  }

  Representation representation = access.representation();
  if (representation.IsSmi() && SmiValuesAre32Bits() &&
    instr->representation().IsInteger32()) {
    if(FLAG_debug_code) {
      UNIMPLEMENTED();
      // TODO(llvm):
      // Load(scratch, FieldOperand(object, offset), representation);
      // AssertSmi(scratch);
    }
    STATIC_ASSERT(kSmiTag == 0);
    DCHECK(kSmiTagSize + kSmiShiftSize == 32);
    offset += kPointerSize / 2;
    representation = Representation::Integer32();
  }
 
  llvm::Value* obj = FieldOperand(Use(instr->object()), offset);
  if (instr->representation().IsInteger32()) {
    llvm::Value* casted_address = __ CreateBitCast(obj, Types::ptr_i32);
    llvm::Value* res = __ CreateLoad(casted_address);
    instr->set_llvm_value(res);
  } else {
    llvm::Value* casted_address = __ CreateBitCast(obj, Types::ptr_i64);
    llvm::Value* res = __ CreateLoad(casted_address);
    instr->set_llvm_value(res);

  }
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
  if (instr->representation().IsSmiOrInteger32()) {
    if (instr->RightIsPowerOf2()) {
       DoModByPowerOf2I(instr);
    } else if (instr->right()->IsConstant()) {
      UNIMPLEMENTED();
      //return DoModByConstI(instr);
    } else {
      UNIMPLEMENTED();
      //return DoModI(instr);
    }
  } else if (instr->representation().IsDouble()) {
    UNIMPLEMENTED();
    //return DoArithmeticD(Token::MOD, instr);
  } else {
    UNIMPLEMENTED();
    //return DoArithmeticT(Token::MOD, instr);
  }
}

void LLVMChunkBuilder::DoModByPowerOf2I(HMod* instr) {
  llvm::BasicBlock* is_not_negative = NewBlock("DIVIDEND_IS_NOT_NEGATIVE");
  llvm::BasicBlock* near = NewBlock("Near");
  llvm::BasicBlock* done = NewBlock("DONE");

  HValue* dividend = instr->left();
  int32_t divisor = instr->right()->GetInteger32Constant();
  int32_t mask = divisor < 0 ? -(divisor + 1) : (divisor - 1);
  llvm::Value* l_mask = __ getInt32(mask);
  llvm::Value* div1 = nullptr;
  if (instr->CheckFlag(HValue::kLeftCanBeNegative)) {
    llvm::Value* zero = __ getInt32(0);
    llvm::Value* cmp =  __ CreateICmpSGT(Use(dividend), zero);
    __ CreateCondBr(cmp, is_not_negative, near);
    __ SetInsertPoint(near);
    __ CreateNeg(Use(dividend));
    div1 =  __ CreateAnd(Use(dividend), l_mask);
    if (instr->CheckFlag(HValue::kBailoutOnMinusZero)) {
      UNIMPLEMENTED();
    }
    __ CreateBr(done);
  }
  __ SetInsertPoint(is_not_negative);
  llvm::Value* div2 = __ CreateAnd(Use(dividend), l_mask);
  __ CreateBr(done);
  __ SetInsertPoint(done);
  llvm::PHINode* phi = __ CreatePHI(Types::i32, 2);
  phi->addIncoming(div1, near);
  phi->addIncoming(div2, is_not_negative);
  instr->set_llvm_value(phi);
}

void LLVMChunkBuilder::DoMul(HMul* instr) {
  if(instr->representation().IsInteger32() || instr->representation().IsSmi()) {
    DCHECK(instr->left()->representation().Equals(instr->representation()));
    DCHECK(instr->right()->representation().Equals(instr->representation()));
    HValue* left = instr->left();
    HValue* right = instr->right();
    llvm::Value* llvm_left = Use(left);
    llvm::Value* llvm_right = Use(right);
    if (instr->representation().IsSmi()) {
      // FIXME (llvm):
      // 1) overflow check?
      // 2) see if we can refactor using SmiToInteger32() or the like
      llvm::Value* shift = __ CreateAShr(llvm_left, 32);
      llvm::Value* Mul = __ CreateNSWMul(shift, llvm_right, "");
      instr->set_llvm_value(Mul);
    } else {
      llvm::Value* Mul = __ CreateNSWMul(llvm_left, llvm_right, "");
      instr->set_llvm_value(Mul);
    }
  } else if (instr->representation().IsDouble()) {
    DCHECK(instr->representation().IsDouble());
    DCHECK(instr->left()->representation().IsDouble());
    DCHECK(instr->right()->representation().IsDouble());
    HValue* left = instr->left();
    HValue* right = instr->right();
    llvm::Value* fMul =  __ CreateFMul(Use(left), Use(right), "");
    instr->set_llvm_value(fMul);
   }
  else {
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoOsrEntry(HOsrEntry* instr) {
  //UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoPower(HPower* instr) {
  Representation exponent_type = instr->right()->representation();
  
  if (exponent_type.IsSmi()) {
    UNIMPLEMENTED();
  } else if (exponent_type.IsTagged()) {
    UNIMPLEMENTED();
  } else if (exponent_type.IsInteger32()) {
    MathPowStub stub(isolate(), MathPowStub::INTEGER);
    Handle<Code> code = Handle<Code>::null();
    {
      AllowHandleAllocation allow_handles;
      AllowHeapAllocation allow_heap;
      code = stub.GetCode();
      // FIXME(llvm,gc): respect reloc info mode...
    }
    std::vector<llvm::Value*> params;
    for (int i = 0; i < instr->OperandCount(); i++)
      params.push_back(Use(instr->OperandAt(i))); 
    llvm::Value* call = CallAddressForMathPow(code->instruction_start(),
                                    llvm::CallingConv::X86_64_V8_S2, params);
    instr->set_llvm_value(call);
  } else {
    UNIMPLEMENTED();
  }
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
    llvm::Value* AShr = __ CreateAShr(Use(left), Use(right),"");
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
    llvm::Value* Shl = __ CreateShl(Use(left), Use(right),"");
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
    llvm::Value* LShr = __ CreateLShr(Use(left), Use(right),"");
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
    Handle<Cell> instr_cell = instr->cell().handle();
    auto ptr_value = reinterpret_cast<uint64_t>(*(instr_cell.location()));
    auto address_val = __ getInt64(ptr_value);
    auto gep = FieldOperand(address_val, 8);
    llvm::Value* casted_address = __ CreateBitCast(gep, Types::ptr_i64);
    auto loaded_value = __ CreateLoad(casted_address);
    llvm::Value* deopt_val = CompareRoot(loaded_value, Heap::kTheHoleValueRootIndex);
    DeoptimizeIf(deopt_val);
    llvm::Value* store_cell = __ CreateStore(Use(instr->value()), casted_address);
    instr->set_llvm_value(store_cell);    
    //UNIMPLEMENTED();
  } else {
    Handle<Object> handle_value = instr->cell().handle();
    int64_t value = reinterpret_cast<int64_t>(*(handle_value.location()));
    auto address_val = __ getInt64(value);
    auto gep = FieldOperand(address_val, 8);
    llvm::Value* casted_address = __ CreateBitCast(gep, Types::ptr_i64);
    llvm::Value* store_cell = __ CreateStore(Use(instr->value()), casted_address);
    instr->set_llvm_value(store_cell);
  }
}

void LLVMChunkBuilder::DoStoreKeyed(HStoreKeyed* instr) {
  if (instr->is_typed_elements()) {
    //DoStoreKeyedExternalArray(instr);
    UNIMPLEMENTED();
  } else if (instr->value()->representation().IsDouble()) {
    DoStoreKeyedFixedDoubleArray(instr);
    //UNIMPLEMENTED();
  } else {
    DoStoreKeyedFixedArray(instr);
  }
}

void LLVMChunkBuilder::DoStoreKeyedFixedDoubleArray(HStoreKeyed* instr) {
  HValue* key = instr->key();
  int shift_size = ElementsKindToShiftSize(FAST_DOUBLE_ELEMENTS);
  uint32_t inst_offset = instr->base_offset();
  llvm::Value* gep_0 = nullptr;
  if (kPointerSize == kInt32Size && !key->IsConstant()
      && instr->IsDehoisted()) {
    UNIMPLEMENTED();
  }
  if (instr->NeedsCanonicalization()) {
    llvm::Value* val_ = __ getInt64(0);
    __ CreateXor(val_, val_);
    llvm::Value* double_val_ = __ CreateSIToFP(val_, Types::float64);
    llvm::Value* double_input_ = __ CreateSIToFP(Use(instr->value()), Types::float64);
    __ CreateFSub(double_input_, double_val_);
  }
  if (key->IsConstant()) {
    uint32_t const_val = (HConstant::cast(key))->Integer32Value();
    gep_0 = ConstructAddress(Use(instr->elements()), (const_val << shift_size) + inst_offset);
  } else {
     llvm::Value* lkey = Use(key);
     llvm::Value* scale = nullptr;
     llvm::Value* offset = nullptr;
     if (key->representation().IsInteger32()) {
       scale = __ getInt32(8);
       offset = __ getInt32(inst_offset);
     } else {
       scale = __ getInt64(8);
       offset = __ getInt64(inst_offset);
     }
     llvm::Value* mul = __ CreateMul(lkey, scale);
     llvm::Value* add = __ CreateAdd(mul, offset);
     llvm::Value* int_ptr = __ CreateIntToPtr(Use(instr->elements()),
                                              Types::ptr_i8);
     gep_0 = __ CreateGEP(int_ptr, add);
  }
    llvm::Value* casted_address = __ CreateBitCast(gep_0, Types::ptr_float64);
    llvm::Value* Store = __ CreateStore(Use(instr->value()), casted_address);
    instr->set_llvm_value(Store);
}

void LLVMChunkBuilder::DoStoreKeyedFixedArray(HStoreKeyed* instr) {
  HValue* key = instr->key();
  Representation representation = instr->value()->representation();
  int shift_size = ElementsKindToShiftSize(FAST_ELEMENTS);
  uint32_t inst_offset = instr->base_offset();
  llvm::Value* gep_0 = nullptr;
  if (kPointerSize == kInt32Size && !key->IsConstant() &&
      instr->IsDehoisted()) {
    UNIMPLEMENTED();
  }
  if (representation.IsInteger32() && SmiValuesAre32Bits()) {
    DCHECK(instr->store_mode() == STORE_TO_INITIALIZED_ENTRY);
    DCHECK(instr->elements_kind() == FAST_SMI_ELEMENTS);
    if (FLAG_debug_code) {
      UNIMPLEMENTED();
    }
    inst_offset += kPointerSize / 2;

  }
  if (key->IsConstant()) {
    uint32_t const_val = (HConstant::cast(key))->Integer32Value();
    gep_0 = ConstructAddress(Use(instr->elements()), (const_val << shift_size) + inst_offset);
  } else {
     llvm::Value* lkey = Use(key);
     llvm::Value* scale = nullptr;
     llvm::Value* offset = nullptr;
     if (key->representation().IsInteger32()) {
       scale = __ getInt32(8);
       offset = __ getInt32(inst_offset);
     } else {
       scale = __ getInt64(8);
       offset = __ getInt64(inst_offset);
     }
     llvm::Value* mul = __ CreateMul(lkey, scale);
     llvm::Value* add = __ CreateAdd(mul, offset);
     llvm::Value* int_ptr = __ CreateIntToPtr(Use(instr->elements()),
                                              Types::ptr_i8);
     gep_0 = __ CreateGEP(int_ptr, add);
  }
 
  HValue* hValue = instr->value();
  llvm::Value* store = nullptr;
  if (hValue->representation().IsInteger32()) {
    llvm::Value* casted_adderss = __ CreateBitCast(gep_0,
                                                   Types::ptr_i32);
    store = __ CreateStore(Use(hValue), casted_adderss);
  } else if (hValue->representation().IsSmi() || !hValue->IsConstant()){
    llvm::Value* casted_adderss = __ CreateBitCast(gep_0,
                                                   Types::ptr_i64);
    store = __ CreateStore(Use(hValue), casted_adderss);
  } else {
    DCHECK(hValue->IsConstant());
    HConstant* constant = HConstant::cast(instr->value());
    Handle<Object> handle_value = constant->handle(isolate());
    llvm::Value* casted_adderss = __ CreateBitCast(gep_0,
                                                  Types::ptr_i64);
    auto llvm_val = MoveHeapObject(handle_value);
    store = __ CreateStore(llvm_val, casted_adderss);
  } 
  instr->set_llvm_value(store);
  if (instr->NeedsWriteBarrier()) {
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoStoreKeyedGeneric(HStoreKeyedGeneric* instr) {
  UNIMPLEMENTED();
  DCHECK(instr->object()->representation().IsTagged());
  DCHECK(instr->key()->representation().IsTagged());
  DCHECK(instr->value()->representation().IsTagged());

  Handle<Code> ic = CodeFactory::KeyedStoreICInOptimizedCode(
                        isolate(), instr->language_mode(),
                        instr->initialization_state()).code();

  // TODO(llvm): RecordSafepointWithLazyDeopt (and reloc info) + MarkAsCall

  std::vector<llvm::Value*> no_params;
  auto result = CallAddress(ic->instruction_start(), llvm::CallingConv::C,
                            no_params);
  instr->set_llvm_value(result);
}

void LLVMChunkBuilder::DoStoreNamedField(HStoreNamedField* instr) {
  Representation representation = instr->representation();

   HObjectAccess access = instr->access();
   int offset = access.offset() - 1;

  if (access.IsExternalMemory()) { 
    UNIMPLEMENTED();
  }

  AssertNotSmi(Use(instr->object()));

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
  } else {
    HValue* hValue = instr->value();
    if (hValue->representation().IsInteger32()) {
      llvm::Value* store_address = ConstructAddress(Use(instr->object()), offset);
      llvm::Value* casted_adderss = __ CreateBitCast(store_address,
                                                     Types::ptr_i32);
      llvm::Value* casted_value = __ CreateBitCast(Use(hValue), Types::i32);
      __ CreateStore(casted_value, casted_adderss);
    } else if (hValue->representation().IsSmi() || !hValue->IsConstant()){
      llvm::Value* store_address = ConstructAddress(Use(instr->object()), offset);
      llvm::Value* casted_adderss = __ CreateBitCast(store_address,
                                                     Types::ptr_i64);
      llvm::Value* casted_value = __ CreateBitCast(Use(hValue), Types::i64);
      __ CreateStore(casted_value, casted_adderss);
    } else {
      DCHECK(hValue->IsConstant());
      HConstant* constant = HConstant::cast(instr->value());
      Handle<Object> handle_value = constant->handle(isolate());
      llvm::Value* store_address = ConstructAddress(Use(instr->object()),
                                                    offset);
      llvm::Value* casted_adderss = __ CreateBitCast(store_address,
                                                     Types::ptr_i64);
      auto llvm_val = MoveHeapObject(handle_value);
      __ CreateStore(llvm_val, casted_adderss);

    }
  }

  if (instr->NeedsWriteBarrier()) {
    // UNIMPLEMENTED(); FIXME temporary for testing store_key
  }
}

void LLVMChunkBuilder::DoStoreNamedGeneric(HStoreNamedGeneric* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoStringAdd(HStringAdd* instr) {
//  llvm::Value* context = __ CreateLoad(instr->context()->llvm_value(), "RSI");
  // see GetContext()!
  StringAddStub stub(isolate(),
                     instr->flags(),
                     instr->pretenure_flag());

  //llvm::Function* callStrAdd = llvm::Function::Create(&LCodeGen::CallCode, llvm::Function::ExternalLinkage );
  //LCodeGen(NULL, NULL, NULL).CallCode(stub.GetCode(), RelocInfo::CODE_TARGET, NULL);
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoStringCharCodeAt(HStringCharCodeAt* instr) {
  std::vector<llvm::Value*> args;
  llvm::Value* str = Integer32ToSmi(instr->string());
  args.push_back(str);
  //TODO : implement non constant case
  if(instr->index()->IsConstant()) {
    llvm::Value* const_index = Integer32ToSmi(instr->index());
    args.push_back(const_index);
  } else {
    UNIMPLEMENTED(); 
  }
  llvm::Value* alloc = CallRuntimeFromDeferred(
      Runtime::kStringCharCodeAtRT, Use(instr->context()), args);
  auto alloc_casted = __ CreatePtrToInt(alloc, Types::i64);
  instr->set_llvm_value(alloc_casted);
}

void LLVMChunkBuilder::DoStringCharFromCode(HStringCharFromCode* instr) {
  //TODO:Fast case implementation
  std::vector<llvm::Value*> args;
  llvm::Value* arg1 = Integer32ToSmi(instr->value());
  args.push_back(arg1);
  llvm::Value* alloc =  CallRuntimeFromDeferred(Runtime::kCharFromCode, Use(instr->context()), args);
  auto alloc_casted = __ CreatePtrToInt(alloc, Types::i64);
  instr->set_llvm_value(alloc_casted);
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
    llvm::Value* Sub = __ CreateSub(Use(left), Use(right), "");
    instr->set_llvm_value(Sub);
  } else if (instr->representation().IsDouble()) {
    DCHECK(instr->representation().IsDouble());
    DCHECK(instr->left()->representation().IsDouble());
    DCHECK(instr->right()->representation().IsDouble());
    HValue* left = instr->left();
    HValue* right = instr->right();
    llvm::Value* fSub =  __ CreateFSub(Use(left), Use(right), "");
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

void LLVMChunkBuilder::DoTransitionElementsKind(
    HTransitionElementsKind* instr) {
  DCHECK(instr->HasNoUses());
  auto object = Use(instr->object());
  Handle<Map> from_map = instr->original_map().handle();
  Handle<Map> to_map = instr->transitioned_map().handle();
  ElementsKind from_kind = instr->from_kind();
  ElementsKind to_kind = instr->to_kind();

  llvm::BasicBlock* end = NewBlock("TransitionElementsKind end");
  llvm::BasicBlock* cont = NewBlock("TransitionElementsKind meat");

  auto comp = Compare(LoadFieldOperand(object, HeapObject::kMapOffset),
                      from_map);
  __ CreateCondBr(comp, cont, end);
  __ SetInsertPoint(cont);

  if (IsSimpleMapChangeTransition(from_kind, to_kind)) {
    // map is an i64.
    auto new_map = Move(to_map, RelocInfo::EMBEDDED_OBJECT);
    auto store_addr = FieldOperand(object, HeapObject::kMapOffset);
    auto casted_store_addr = __ CreateBitCast(store_addr, Types::ptr_i64);
    __ CreateStore(new_map, casted_store_addr);
    // Write barrier. TODO(llvm): give llvm.gcwrite and company a thought.
    RecordWriteForMap(object, new_map);
    __ CreateBr(end);
  } else {
    UNIMPLEMENTED();
  }
  __ SetInsertPoint(end);
}

void LLVMChunkBuilder::RecordWriteForMap(llvm::Value* object,
                                         llvm::Value* map) {
  AssertNotSmi(object);

  if (emit_debug_code()) {
    auto maps_equal = CompareMap(map, isolate()->factory()->meta_map());
    Assert(maps_equal);
  }

  if (!FLAG_incremental_marking) {
    return;
  }

  if (emit_debug_code()) {
    // FIXME(llvm): maybe we should dereference the FieldOperand
    Assert(Compare(map, LoadFieldOperand(object, HeapObject::kMapOffset)));
  }

  auto map_address = FieldOperand(object, HeapObject::kMapOffset); // dst
  map_address = __ CreateBitOrPointerCast(map_address, Types::tagged);

  auto equal = CheckPageFlag(map,
                             MemoryChunk::kPointersToHereAreInterestingMask);

  auto cont = NewBlock("CheckPageFlag OK");
  auto call_stub = NewBlock("Call RecordWriteStub");
  __ CreateCondBr(equal, cont, call_stub);

  __ SetInsertPoint(call_stub);
  // The following are the registers expected by the calling convention.
  // They can be changed, but the CC must be adjusted accordingly.
  Register object_reg = rbx;
  Register map_reg = rcx;
  Register dst_reg = rdx;
  RecordWriteStub stub(isolate(), object_reg, map_reg, dst_reg,
                       OMIT_REMEMBERED_SET, kDontSaveFPRegs);
  Handle<Code> code = Handle<Code>::null();
  {
    AllowHandleAllocation allow_handles;
    AllowHeapAllocation allow_heap_alloc;
    code = stub.GetCode();
    // FIXME(llvm,gc): respect reloc info mode...
  }
  std::vector<llvm::Value*> params = { object, map, map_address };
  CallAddress(code->instruction_start(),
              llvm::CallingConv::X86_64_V8_RWS,
              params);
  __ CreateBr(cont);

  __ SetInsertPoint(cont);

  // Count number of write barriers in generated code.
  isolate()->counters()->write_barriers_static()->Increment();
  IncrementCounter(isolate()->counters()->write_barriers_dynamic(), 1);
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

void LLVMChunkBuilder::DoIntegerMathAbs(HUnaryMathOperation* instr) {
  llvm::BasicBlock* is_negative = NewBlock("INTEGER CANDIDATE IS NEGATIVE");
  llvm::BasicBlock* is_positive = NewBlock("INTEGER CANDIDATE IS POSITIVE");

  llvm::Value* zero = __ getInt64(0);
  llvm::Value* cmp =  __ CreateICmpSLT(Use(instr->value()), zero);
  __ CreateCondBr(cmp, is_negative, is_positive);
  __ SetInsertPoint(is_negative);
  llvm::Value* neg_val =  __ CreateNeg(Use(instr->value()));
  bool negate_condition = true;
  DeoptimizeIf(cmp,  negate_condition);
  __ CreateBr(is_positive);
  __ SetInsertPoint(is_positive);
  llvm::Value* val = Use(instr->value());
  llvm::PHINode* phi = __ CreatePHI(Types::i64, 2);
  phi->addIncoming(neg_val, is_negative);
  phi->addIncoming(val, is_positive);
  instr->set_llvm_value(phi);
}

void LLVMChunkBuilder::DoMathAbs(HUnaryMathOperation* instr) {
  Representation r = instr->representation();
  if (r.IsDouble()) {
    UNIMPLEMENTED();
  } else if (r.IsInteger32()) {
    DoIntegerMathAbs(instr);
  } else if (r.IsSmi()) {
    UNIMPLEMENTED();
  } else {
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoMathPowHalf(HUnaryMathOperation* instr) {
  //TODO : add -infinity and  infinity checks
  llvm::Value* input_ =  Use(instr->value());
  llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
          llvm::Intrinsic::sqrt, Types::float64);
  std::vector<llvm::Value*> params;
  params.push_back(input_);
  llvm::Value* call = __ CreateCall(intrinsic, params);
  instr->set_llvm_value(call); 
  
}

void LLVMChunkBuilder::DoMathSqrt(HUnaryMathOperation* instr) {
   llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
          llvm::Intrinsic::sqrt, Types::float64);
   std::vector<llvm::Value*> params;
   params.push_back(Use(instr->value()));
   llvm::Value* sqrt = __ CreateCall(intrinsic, params);
   instr->set_llvm_value(sqrt);
}

void LLVMChunkBuilder::DoUnaryMathOperation(HUnaryMathOperation* instr) {
  switch (instr->op()) {
    case kMathAbs:
      DoMathAbs(instr);
      break;
    case kMathPowHalf:
      DoMathPowHalf(instr);
      break;
    case kMathFloor:
      UNIMPLEMENTED();
    case kMathRound:
      UNIMPLEMENTED();
    case kMathFround:
      UNIMPLEMENTED();
    case kMathLog:
      UNIMPLEMENTED();
    case kMathExp:
      UNIMPLEMENTED();
    case kMathSqrt: {
      DoMathSqrt(instr);
      break;
    }
    case kMathClz32:
      UNIMPLEMENTED();
    default:
      UNREACHABLE();
  }
}

void LLVMChunkBuilder::DoUnknownOSRValue(HUnknownOSRValue* instr) {
  UNIMPLEMENTED();
  int env_index = instr->index();
  //if (env_index == 0) return;
  int index = 0;
  if (instr->environment()->is_parameter_index(env_index)) {
    index =  env_index - info()->num_parameters() - 1; //chunk()->GetParameterStackSlot(env_index);
    int num_parameters = info()->num_parameters() + 3;
    llvm::Function::arg_iterator it = function_->arg_begin();
    //index = -index;
    while (--index + num_parameters > 0) ++it;
    instr->set_llvm_value(it);

  } else {
    //UNIMPLEMENTED();
    index = env_index - instr->environment()->first_local_index();
    if (index > LUnallocated::kMaxFixedSlotIndex) {
      UNIMPLEMENTED();
     // Retry(kTooManySpillSlotsNeededForOSR);
     // spill_index = 0;
    }
  }

}

void LLVMChunkBuilder::DoUseConst(HUseConst* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoWrapReceiver(HWrapReceiver* instr) {
  UNIMPLEMENTED();
}

void LLVMEnvironment::AddValue(llvm::Value* value,
                               Representation representation,
                               bool is_uint32) {
  DCHECK(value->getType() == LLVMChunkBuilder::GetLLVMType(representation));
  values_.Add(value, zone());
  if (representation.IsSmiOrTagged()) {
    DCHECK(!is_uint32);
    is_tagged_.Add(values_.length() - 1, zone());
  }

  if (is_uint32) {
    is_uint32_.Add(values_.length() - 1, zone());
  }
}

#undef __

} }  // namespace v8::internal
