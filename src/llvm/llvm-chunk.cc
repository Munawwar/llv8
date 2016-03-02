// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <cstdio>
#include <iomanip>

#include "src/code-factory.h"
#include "src/disassembler.h"
#include "src/hydrogen-osr.h"
#include "src/ic/ic.h"
#include "src/safepoint-table.h"
#include "llvm-chunk.h"
#include "pass-normalize-phis.h"
#include "src/profiler/cpu-profiler.h"
#include <llvm/IR/InlineAsm.h>
#include "llvm-stackmaps.h"

namespace v8 {
namespace internal {

#define __ llvm_ir_builder_->

auto LLVMGranularity::x64_target_triple = "x86_64-unknown-linux-gnu";
const char* LLVMChunkBuilder::kGcStrategyName = "v8-gc";
const std::string LLVMChunkBuilder::kPointersPrefix = "pointer_";
llvm::Type* Types::i8 = nullptr;
llvm::Type* Types::i32 = nullptr;
llvm::Type* Types::i64 = nullptr;
llvm::Type* Types::float32 = nullptr;
llvm::Type* Types::float64 = nullptr;
llvm::PointerType* Types::ptr_i8 = nullptr;
llvm::PointerType* Types::ptr_i16 = nullptr;
llvm::PointerType* Types::ptr_i32 = nullptr;
llvm::PointerType* Types::ptr_i64 = nullptr;
llvm::PointerType* Types::ptr_float32 = nullptr;
llvm::PointerType* Types::ptr_float64 = nullptr;
llvm::Type* Types::tagged = nullptr;
llvm::PointerType* Types::ptr_tagged = nullptr;
llvm::Type* Types::smi = nullptr;
llvm::Type* Types::ptr_smi = nullptr;

LLVMChunk::~LLVMChunk() {}

static void DumpSafepoints(Code* code) {
  SafepointTable table(code);
  std::cerr << "Safepoints (size = " << table.size() << ")\n";
  for (unsigned i = 0; i < table.length(); i++) {
    unsigned pc_offset = table.GetPcOffset(i);
    std::cerr << static_cast<const void*>(code->instruction_start() + pc_offset) << "  ";
    std::cerr << std::setw(4) << pc_offset << "  ";
    table.PrintEntry(i, std::cerr);
    std::cerr << " (sp -> fp)  ";
    SafepointEntry entry = table.GetEntry(i);
    if (entry.deoptimization_index() != Safepoint::kNoDeoptimizationIndex) {
      std::cerr << std::setw(6) << entry.deoptimization_index();
    } else {
      std::cerr << "<none>";
    }
    if (entry.argument_count() > 0) {
      std::cerr << " argc: " << entry.argument_count();
    }
    std::cerr << "\n";
  }
  std::cerr << "\n";
}

Handle<Code> LLVMChunk::Codegen() {
  uint64_t address = LLVMGranularity::getInstance().GetFunctionAddress(
      llvm_function_id_);
  auto buf = LLVMGranularity::getInstance().memory_manager_ref()
      ->LastAllocatedCode().buffer;
  USE(buf);
#ifdef DEBUG
  std::cerr << "\taddress == " <<  reinterpret_cast<void*>(address) << std::endl;
  std::cerr << "\tlast allocated code section start == "
      << static_cast<void*>(buf) << std::endl;
  // FIXME(llvm):
  // The right thing is address. But for now it's harder to get. So there.
  if (reinterpret_cast<void*>(address) != static_cast<void*>(buf))
    UNIMPLEMENTED();
  LLVMGranularity::getInstance().Err();
#else
  USE(address);
#endif

  Isolate* isolate = info()->isolate();
  CodeDesc& code_desc =
      LLVMGranularity::getInstance().memory_manager_ref()->LastAllocatedCode();

  // This is of course totally untrue.
  code_desc.origin = &masm_;

#ifdef DEBUG
  LLVMGranularity::getInstance().Disass(
      code_desc.buffer, code_desc.buffer + code_desc.instr_size);
#endif

  StackMaps stackmaps = GetStackMaps();

  // It is important that this call goes before EmitSafepointTable()
  // because it patches nop sequences to calls (and EmitSafepointTable
  // looks for calls in the instruction stream to determine their sizes).
  std::vector<RelocInfo> reloc_info_from_patchpoints =
      SetUpRelativeCalls(buf, stackmaps);

  // This assembler owns it's buffer (it contains our SafepointTable).
  // FIXME(llvm): assembler shouldn't care for kGap for our case...
  auto initial_buffer_size = Max(code_desc.buffer_size / 6, 32);
  Assembler assembler(isolate, nullptr, initial_buffer_size);
  EmitSafepointTable(&assembler, stackmaps, buf);
  CodeDesc safepoint_table_desc;
  assembler.GetCode(&safepoint_table_desc);

  Vector<byte> reloc_bytevector = GetFullRelocationInfo(
      code_desc, reloc_info_from_patchpoints);

  // Allocate and install the code.
  if (info()->IsStub()) UNIMPLEMENTED(); // Probably different flags for stubs.
  Code::Flags flags = Code::ComputeFlags(info()->output_code_kind());
  Handle<Code> code = isolate->factory()->NewLLVMCode(
      code_desc, safepoint_table_desc, &reloc_bytevector, flags);
  isolate->counters()->total_compiled_code_size()->Increment(
      code->instruction_size());

  SetUpDeoptimizationData(code, stackmaps);

  // TODO(llvm): it is not thread-safe. It's not anything-safe.
  // We assume a new function gets attention after the previous one
  // has been fully processed by llv8.
  LLVMGranularity::getInstance().memory_manager_ref()->DropStackmaps();
#ifdef DEBUG
  std::cerr << "Instruction start: "
      << reinterpret_cast<void*>(code->instruction_start()) << std::endl;
#endif

#ifdef DEBUG
  LLVMGranularity::getInstance().Disass(
      code->instruction_start(),
      code->instruction_start() + code->instruction_size());

  std::cerr << "\nRelocInfo (size = " << code->relocation_size() << ")\n";
  for (RelocIterator it(*code.location()); !it.done(); it.next()) {
    it.rinfo()->Print(isolate, std::cerr);
  }
  std::cerr << "\n";

  DumpSafepoints(*code);
#else
  USE(DumpSafepoints);
#endif
  return code;
}


void LLVMChunk::WriteTranslation(LLVMEnvironment* environment,
                                 Translation* translation,
                                 const StackMaps& stackmaps,
                                 int32_t patchpoint_id,
                                 int start_index) {
  if (!environment) return;

  if (environment->outer()) {
    WriteTranslation(environment->outer(), translation, stackmaps, patchpoint_id,
                     start_index - environment->outer()->translation_size());
  }

  // TODO(llvm): this isn't very good performance-wise. Esp. considering
  // the result of this call is the same across recursive invocations.
  auto stackmap_record = stackmaps.computeRecordMap()[patchpoint_id];

  // The translation includes one command per value in the environment.
  int translation_size = environment->translation_size();
  // The output frame height does not include the parameters.
  int height = translation_size - environment->parameter_count();

  int shared_id = deopt_data_->DefineDeoptimizationLiteral(
      environment->entry() ? environment->entry()->shared()
                           : info()->shared_info());
  int closure_id = deopt_data_->DefineDeoptimizationLiteral(
      environment->closure());
  // WriteTranslationFrame
  switch (environment->frame_type()) {
    case JS_FUNCTION:
      translation->BeginJSFrame(environment->ast_id(), shared_id, height);
      if (info()->closure().is_identical_to(environment->closure())) {
        translation->StoreJSFrameFunction();
      } else {
        translation->StoreLiteral(closure_id);
      }
      break;
    case JS_CONSTRUCT:
      translation->BeginConstructStubFrame(shared_id, translation_size);
      if (info()->closure().is_identical_to(environment->closure())) {
        translation->StoreJSFrameFunction();
      } else {
        translation->StoreLiteral(closure_id);
      }
      break;
    case JS_GETTER:
      UNIMPLEMENTED();
      break;
    case JS_SETTER:
      UNIMPLEMENTED();
      break;
    case ARGUMENTS_ADAPTOR:
      translation->BeginArgumentsAdaptorFrame(shared_id, translation_size);
      if (info()->closure().is_identical_to(environment->closure())) {
        translation->StoreJSFrameFunction();
      } else {
        translation->StoreLiteral(closure_id);
      }
      break;
    case STUB:
      UNIMPLEMENTED();
      break;
    default:
      UNIMPLEMENTED();
  }

  int object_index = 0;
  int dematerialized_index = 0;

  DCHECK_LE(start_index + translation_size - 1,
            stackmap_record.locations.size());
  for (int i = 0; i < translation_size; ++i) {
    // i-th Location in a stackmap record Location corresponds to the i-th
    // llvm::Value*. Here's an excerpt from the doc:
    // > The runtime must be able to interpret the stack map record given only
    // > the ID, offset, and the order of the locations, *which LLVM preserves*.

    // FIXME(llvm): It seems we cannot really use llvm::Value* here, because
    // since we generated them optimization has happened
    // (therefore those values are now invalid).

    llvm::Value* value = environment->values()->at(i);
    StackMaps::Location location = stackmap_record.locations[i + start_index];
    AddToTranslation(environment,
                     translation,
                     value,
                     location,
                     stackmaps.constants,
                     environment->HasTaggedValueAt(i),
                     environment->HasUint32ValueAt(i),
                     environment->HasDoubleValueAt(i),
                     &object_index,
                     &dematerialized_index);
  }
}

// As far as I understand, index is CallerPC-relative offset
// i.e. relative to the stack cell holding the ret address.
static int FpRelativeOffsetToIndex(int32_t offset) {
  //  ........................
  // index                      fp-relative offset (decimal)
  //  -1   | arg N (the last) | +16
  //   0   |       RET        | +8
  //   1   |     saved FP     |  0
  //   2   | saved context    | -8
  //  ........................

  DCHECK(offset % kInt32Size == 0);
  if (FLAG_enable_embedded_constant_pool) // This would change the pic above.
    UNIMPLEMENTED();
  auto index = -offset / kPointerSize + 1;
  return index;
}

void LLVMChunk::AddToTranslation(
    LLVMEnvironment* environment, // TODO(llvm): unused?
    Translation* translation,
    llvm::Value* op,
    StackMaps::Location& location,
    const std::vector<StackMaps::Constant> constants,
    bool is_tagged,
    bool is_uint32,
    bool is_double,
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
    auto index = FpRelativeOffsetToIndex(location.offset);
    if (is_tagged) {
      DCHECK(location.size == kPointerSize);
      translation->StoreStackSlot(index);
    } else if (is_uint32) {
      DCHECK(location.size == kInt32Size);
      translation->StoreUint32StackSlot(index);
    } else {
      if (is_double) {
        DCHECK(location.size == kDoubleSize);
        translation->StoreDoubleStackSlot(index);
      } else {
        DCHECK(location.size == kInt32Size);
        translation->StoreInt32StackSlot(index);
      }
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
    auto value = constants[location.offset].integer;

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

  // All of the LLVMEnvironments (closure of env by ->outer())
  // share the same associated patchpoint_id.
  auto patchpoint_id = deopt_data_->GetPatchpointIdByEnvironment(env);
  // But they have different start indices in the corresponding
  // Stack Map record. Layout of the Stack Map record (order of Locations)
  // is the same as that of the TranslationBuffer i.e. the most outer first.
  auto stackmap_record = stackmaps.computeRecordMap()[patchpoint_id];
  auto total_size = IntHelper::AsInt(stackmap_record.locations.size());
  auto start_index_inner = total_size - env->translation_size();
  WriteTranslation(
      env, &translation, stackmaps, patchpoint_id, start_index_inner);
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

void* LLVMDeoptData::GetKey(int32_t patchpoint_id) {
  DCHECK(patchpoint_id >= 0);
  auto new_int = new(zone_) int32_t;
  *new_int = patchpoint_id;
  return new_int;
}

uint32_t LLVMDeoptData::GetHash(int32_t patchpoint_id) {
  DCHECK(patchpoint_id >= 0);
  return static_cast<uint32_t>(patchpoint_id);
}

void LLVMDeoptData::Add(LLVMEnvironment* environment, int32_t patchpoint_id) {
  auto key = GetKey(patchpoint_id);
  auto hash = GetHash(patchpoint_id);
  auto entry = deoptimizations_.LookupOrInsert(key, hash,
                                               ZoneAllocationPolicy(zone_));
  entry->value = environment;

  reverse_deoptimizations_[environment] = patchpoint_id;
}

LLVMEnvironment* LLVMDeoptData::GetEnvironmentByPatchpointId(
    int32_t patchpoint_id) {
  auto key = GetKey(patchpoint_id);
  auto hash = GetHash(patchpoint_id);
  auto entry = deoptimizations_.Lookup(key, hash);
  return static_cast<LLVMEnvironment*>(entry->value);
}

int32_t LLVMDeoptData::GetPatchpointIdByEnvironment(LLVMEnvironment* env) {
  return reverse_deoptimizations_[env];
}


std::vector<RelocInfo> LLVMChunk::SetUpRelativeCalls(
    Address start,
    const StackMaps& stackmaps) {
  std::vector<RelocInfo> result;

  for (auto i = 0; i < stackmaps.records.size(); i++) {
    auto record = stackmaps.records[i];
    auto id = record.patchpointID;
    if (!reloc_data_->IsPatchpointIdReloc(id)) continue;

    auto pc_offset = start + record.instructionOffset;
    *pc_offset++ = 0xE8; // Call relative offset.

    if (reloc_data_->IsPatchpointIdDeopt(id)) {
      // Record relocatable runtime entry (deoptimization bailout target).
      RelocInfo reloc_info(pc_offset, RelocInfo::RUNTIME_ENTRY, 0, nullptr);
      result.push_back(reloc_info);
      auto delta = deopt_target_offset_for_ppid_[id];
      Memory::int32_at(pc_offset) = IntHelper::AsInt32(delta);

      // Record deoptimization reason.
      if (FLAG_trace_deopt || isolate()->cpu_profiler()->is_profiling()) {
        RelocInfo deopt_reason(pc_offset - 1, RelocInfo::DEOPT_REASON,
                               reloc_data_->GetDeoptReason(id), nullptr);
        result.push_back(deopt_reason);
      }
    } else {
      if (reloc_data_->IsPatchpointIdRelocNop(id)) {
        // TODO(llvm): it's always CODE_TARGET for now.
        RelocInfo reloc_info(pc_offset, RelocInfo::CODE_TARGET, 0, nullptr);
        result.push_back(reloc_info);
        Memory::uint32_at(pc_offset) = target_index_for_ppid_[id];
        pc_offset = pc_offset + 4;
        *pc_offset++ = Assembler::kNopByte;
      } else {
        DCHECK(reloc_data_->IsPatchpointIdReloc(id));
        // TODO(llvm): it's always CODE_TARGET for now.
        RelocInfo reloc_info(pc_offset, RelocInfo::CODE_TARGET, 0, nullptr);
        result.push_back(reloc_info);
        Memory::uint32_at(pc_offset) = target_index_for_ppid_[id];
      }
    }
  }

  return result;
}

StackMaps LLVMChunk::GetStackMaps() {
  List<byte*>& stackmap_list =
      LLVMGranularity::getInstance().memory_manager_ref()->stackmaps();

  if (stackmap_list.length() == 0) {
    StackMaps empty;
    return empty;
  }

  DCHECK(stackmap_list.length() == 1);

  StackMaps stackmaps;
  DataView view(stackmap_list[0]);
  stackmaps.parse(&view);
#ifdef DEBUG
  stackmaps.dumpMultiline(std::cerr, "  ");
#endif

  // Because we only have one function. This could change in the future.
  DCHECK(stackmaps.stack_sizes.size() == 1);

//  uint64_t address = LLVMGranularity::getInstance().GetFunctionAddress(
//      llvm_function_id_);
//  auto it = std::find_if(stackmaps.stack_sizes.begin(),
//                         stackmaps.stack_sizes.end(),
//                         [address](const StackMaps::StackSize& s) {
//                           return s.functionOffset ==  address;
//                         });
//  DCHECK(it != std::end(stackmaps.stack_sizes));
  return stackmaps;
}

void LLVMChunk::EmitSafepointTable(Assembler* assembler,
                                   StackMaps& stackmaps,
                                   Address instruction_start) {
  SafepointTableBuilder safepoints_builder(zone());

  // TODO(llvm): safepoints should probably be sorted by position in the code (!)
  // As of today, the search @ SafepointTable::FindEntry is linear though.

  int safepoint_arguments = 0;
  // TODO(llvm): There's also kWithRegisters. And with doubles...
  Safepoint::Kind kind = Safepoint::kSimple;
  Safepoint::DeoptMode deopt_mode = Safepoint::kLazyDeopt;

  for (auto stackmap_record : stackmaps.records) {
    auto patchpoint_id = stackmap_record.patchpointID;
    if (!reloc_data_->IsPatchpointIdSafepoint(patchpoint_id)) continue;

    unsigned pc_offset = stackmap_record.instructionOffset;
    int call_instr_size = LLVMGranularity::getInstance().CallInstructionSizeAt(
        instruction_start + pc_offset);
    DCHECK_GT(call_instr_size, 0);
    pc_offset += call_instr_size;
    Safepoint safepoint = safepoints_builder.DefineSafepoint(
        pc_offset, kind, safepoint_arguments, deopt_mode);

    // First three locations are constants describing the calling convention,
    // flags passed to the statepoint intrinsic and the number of following
    // deopt Locations.
    CHECK(stackmap_record.locations.size() >= 3);

    for (auto i = 3; i < stackmap_record.locations.size(); i++) {
      auto location = stackmap_record.locations[i];
      // FIXME(llvm): LLVM bug (should be Indirect). See discussion here:
      // http://lists.llvm.org/pipermail/llvm-dev/2015-November/092394.html
      if (location.kind == StackMaps::Location::kDirect) {
        Register reg = location.dwarf_reg.reg().IntReg();
        if (!reg.is(rbp)) UNIMPLEMENTED();
        DCHECK_LT(location.offset, 0);
        DCHECK_EQ(location.size, kPointerSize);
        auto index = -location.offset / kPointerSize;
        // Safepoint table indices are 0-based from the beginning of the spill
        // slot area, adjust appropriately.
        index -= kPhonySpillCount;
        safepoint.DefinePointerSlot(index, zone());
      } else if (location.kind == StackMaps::Location::kIndirect) {
        UNIMPLEMENTED();
      } else if (location.kind == StackMaps::Location::kConstantIndex) {
        // FIXME(llvm): why do we have these kinds of locations?
      } else {
        UNIMPLEMENTED();
      }
    }
  }

  bool llvmed = true;
  safepoints_builder.Emit(assembler, SpilledCount(stackmaps), llvmed);
}

Vector<byte> LLVMChunk::GetFullRelocationInfo(
    CodeDesc& code_desc,
    const std::vector<RelocInfo>& reloc_data_from_patchpoints) {
  // Relocation info comes from 2 sources:
  // 1) reloc info already present in reloc_data_;
  // 2) patchpoints (CODE_TARGET reloc info has to be extracted from them).
  const std::vector<RelocInfo>& reloc_data_2 = reloc_data_from_patchpoints;
  std::vector<RelocInfo> reloc_data_1 = LLVMGranularity::getInstance().Patch(
      code_desc.buffer, code_desc.buffer + code_desc.instr_size,
      reloc_data_->reloc_map());
  RelocInfoBuffer buffer_writer(8, code_desc.buffer);
  // Mege reloc infos, sort all of them by pc_ and write to the buffer.
  std::vector<RelocInfo> reloc_data_merged;
  reloc_data_merged.insert(reloc_data_merged.end(),
                           reloc_data_1.begin(), reloc_data_1.end());
  reloc_data_merged.insert(reloc_data_merged.end(),
                           reloc_data_2.begin(), reloc_data_2.end());
  std::sort(reloc_data_merged.begin(), reloc_data_merged.end(),
            [](const RelocInfo a, const RelocInfo b) {
              return a.pc() < b.pc();
            });
  for (auto r : reloc_data_merged) buffer_writer.Write(&r);
  v8::internal::Vector<byte> reloc_bytevector = buffer_writer.GetResult();
  // TODO(llvm): what's up with setting reloc_info's host_ to *code?
  return reloc_bytevector;
}

int LLVMChunk::SpilledCount(const StackMaps& stackmaps) {
  int stack_size = IntHelper::AsInt(stackmaps.stackSize());
  DCHECK(stack_size / kStackSlotSize - kPhonySpillCount >= 0);
  return stack_size / kStackSlotSize - kPhonySpillCount;
}

void LLVMChunk::SetUpDeoptimizationData(Handle<Code> code,
                                        StackMaps& stackmaps) {
  code->set_stack_slots(SpilledCount(stackmaps));

  std::vector<uint32_t> sorted_ids;
  int max_deopt = 0;
  for (auto i = 0; i < stackmaps.records.size(); i++) {
    auto id = stackmaps.records[i].patchpointID;
    if (reloc_data_->IsPatchpointIdDeopt(id)) {
      sorted_ids.push_back(id);
      int bailout_id = reloc_data_->GetBailoutId(id);
      if (bailout_id > max_deopt) max_deopt = bailout_id;
    }
  }
  CHECK(max_deopt >= 0);
  std::sort(sorted_ids.begin(), sorted_ids.end());
  size_t true_deopt_count = max_deopt + 1;
  Handle<DeoptimizationInputData> data =
      DeoptimizationInputData::New(isolate(),
                                   IntHelper::AsInt(true_deopt_count), TENURED);

  if (true_deopt_count == 0) return;

  // It's important. It seems something expects deopt entries to be stored
  // is the same order they were added.
  for (auto deopt_entry_number = 0;
      deopt_entry_number < sorted_ids.size();
      deopt_entry_number++) {

    auto stackmap_id = sorted_ids[deopt_entry_number];
    CHECK(reloc_data_->IsPatchpointIdDeopt(stackmap_id));
    int bailout_id = reloc_data_->GetBailoutId(stackmap_id);

    auto env = deopt_data_->GetEnvironmentByPatchpointId(stackmap_id);

    env->set_has_been_used();
    if (env->HasBeenRegistered()) continue;

    int translation_index = WriteTranslationFor(env, stackmaps);
    // pc offset can be obtained from the stackmap TODO(llvm):
    // but we do not support lazy deopt yet (and for eager it should be -1)
    env->Register(bailout_id, translation_index, -1);

    data->SetAstId(bailout_id, env->ast_id());
    data->SetTranslationIndex(bailout_id,
                              Smi::FromInt(translation_index));
    data->SetArgumentsStackHeight(bailout_id,
                                  Smi::FromInt(env->arguments_stack_height()));
    data->SetPc(deopt_entry_number, Smi::FromInt(-1));
  }

  auto length_before = deopt_data_->deoptimization_literals().length();
  for (auto function : inlined_functions()) {
    deopt_data_->DefineDeoptimizationLiteral(function);
  }
  auto inlined_function_count = deopt_data_->deoptimization_literals().length()
      - length_before;
  data->SetInlinedFunctionCount(Smi::FromInt(inlined_function_count));

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
  data->SetOsrPcOffset(Smi::FromInt(6));

  code->set_deoptimization_data(*data);
}

// TODO(llvm): refactor. DRY + move disass features to a separate file.
// Also, we shall not need an instance of LLVMGranularity for such things.
// Returns size of the instruction starting at pc or -1 if an error occurs.
int LLVMGranularity::CallInstructionSizeAt(Address pc) {
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
  llvm::MCContext mc_context(mai.get(), mri.get(), nullptr);
  std::unique_ptr<llvm::MCDisassembler> disasm(
      target->createMCDisassembler(*sti, mc_context));
  DCHECK(disasm);

  llvm::MCInst inst;
  uint64_t size;
  auto max_instruction_lenght = 15; // True for x64.

  llvm::MCDisassembler::DecodeStatus s = disasm->getInstruction(
      inst /* out */, size /* out */,
      llvm::ArrayRef<uint8_t>(pc, pc + max_instruction_lenght),
      0, llvm::nulls(), llvm::nulls());

  std::unique_ptr<const llvm::MCInstrAnalysis> mia(
      target->createMCInstrAnalysis(mii.get()));
  DCHECK(mia);

  if (s == llvm::MCDisassembler::Success && mia->isCall(inst))
    return IntHelper::AsInt(size);
  else
    return -1;
}

std::vector<RelocInfo> LLVMGranularity::Patch(
    Address start, Address end, LLVMRelocationData::RelocMap& reloc_map) {
  std::vector<RelocInfo> updated_reloc_infos;

  // TODO(llvm):
  // This dumb duplication from Disass() looks like it has to be refactored.
  // But this Patch() technique itself is not a production quality solution
  // so it should be gone and is not worth refactoring.
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
          intptr_t data = rinfo.data();
          // Our invariant which is a hack. See RecrodRelocInfo().
          DCHECK_EQ(static_cast<uint64_t>(data), imm);
          if (minfo.cell_extended) { // immediate was extended from 32 bit to 64.
            DCHECK((imm & 0xffffffff) == LLVMChunkBuilder::kExtFillingValue);
            Memory::uintptr_at(pos + 2) = imm >> 32;
            data >>= 32;
          }
          rinfo.set_pc(pos + 2);
          rinfo.set_data(data);
          updated_reloc_infos.push_back(rinfo);
        } else {
          UNIMPLEMENTED();
        }
      }
    }
    pos += size;
  }
  return updated_reloc_infos;
}

int LLVMChunk::GetParameterStackSlot(int index) const {
  // The receiver is at index 0, the first parameter at index 1, so we
  // shift all parameter indexes down by the number of parameters, and
  // make sure they end up negative so they are distinguishable from
  // spill slots.
  int result = index - info()->num_parameters() - 1;

  DCHECK(result < 0);
  return result;
}
LLVMChunk* LLVMChunk::NewChunk(HGraph *graph) {
  DisallowHandleAllocation no_handles;
  DisallowHeapAllocation no_gc;
  graph->DisallowAddingNewValues();
  // int values = graph->GetMaximumValueID();
  CompilationInfo* info = graph->info();

  LLVMChunkBuilder builder(info, graph);
  LLVMChunk* chunk = builder
      .Build()
      .NormalizePhis()
      .GiveNamesToPointerValues()
      .PlaceStatePoints()
      .RewriteStatePoints()
      .Optimize()
      .Create();
  if (chunk == NULL) return NULL;
  return chunk;
}

int32_t LLVMRelocationData::GetNextUnaccountedPatchpointId() {
  return ++last_patchpoint_id_;
}

int32_t LLVMRelocationData::GetNextDeoptPatchpointId() {
  int32_t next_id = ++last_patchpoint_id_;
  DeoptIdMap map {next_id, -1};
  is_deopt_.Add(map, zone_);
  return next_id;
}

int32_t LLVMRelocationData::GetNextSafepointPatchpointId() {
  int32_t next_id = ++last_patchpoint_id_;
  is_safepoint_.Add(next_id, zone_);
  return next_id;
}

int32_t LLVMRelocationData::GetNextRelocPatchpointId(bool is_safepoint) {
  int32_t next_id = ++last_patchpoint_id_;
  is_reloc_.Add(next_id, zone_);
  if (is_safepoint)
    is_safepoint_.Add(next_id, zone_);
  return next_id;
}

int32_t LLVMRelocationData::GetNextRelocNopPatchpointId(bool is_safepoint) {
  int32_t next_id = GetNextRelocPatchpointId(is_safepoint);
  is_reloc_with_nop_.Add(next_id, zone_);
  return next_id;
}

int32_t LLVMRelocationData::GetNextDeoptRelocPatchpointId() {
  auto next_id = GetNextRelocPatchpointId();
  DeoptIdMap map {next_id, -1};
  is_deopt_.Add(map, zone_);
  return next_id;
}

void LLVMRelocationData::SetDeoptReason(int32_t patchpoint_id,
                                        Deoptimizer::DeoptReason reason) {
  deopt_reasons_[patchpoint_id] = reason;
}

Deoptimizer::DeoptReason LLVMRelocationData::GetDeoptReason(
    int32_t patchpoint_id) {
  DCHECK(deopt_reasons_.count(patchpoint_id));
  return deopt_reasons_[patchpoint_id];
}

void LLVMRelocationData::SetBailoutId(int32_t patchpoint_id, int bailout_id) {
  CHECK(IsPatchpointIdDeopt(patchpoint_id));
  for (int i = 0; i < is_deopt_.length(); ++i) {
     if (is_deopt_[i].patchpoint_id == patchpoint_id) {
       is_deopt_[i].bailout_id = bailout_id;
       return;
     }
  }
  UNREACHABLE();
}

int LLVMRelocationData::GetBailoutId(int32_t patchpoint_id) {
  CHECK(IsPatchpointIdDeopt(patchpoint_id));
  for (int i = 0; i < is_deopt_.length(); ++i) {
     if (is_deopt_[i].patchpoint_id == patchpoint_id) {
       CHECK(is_deopt_[i].bailout_id != -1);
       return is_deopt_[i].bailout_id;
     }
  }
  UNREACHABLE();
  return -1;
}

bool LLVMRelocationData::IsPatchpointIdDeopt(int32_t patchpoint_id) {
  for (int i = 0; i < is_deopt_.length(); ++i) {
     if (is_deopt_[i].patchpoint_id == patchpoint_id)
       return true;
  }
  return false;
}

bool LLVMRelocationData::IsPatchpointIdSafepoint(int32_t patchpoint_id) {
  return is_safepoint_.Contains(patchpoint_id);
}

bool LLVMRelocationData::IsPatchpointIdReloc(int32_t patchpoint_id) {
  return is_reloc_.Contains(patchpoint_id);
}

bool LLVMRelocationData::IsPatchpointIdRelocNop(int32_t patchpoint_id) {
  return is_reloc_with_nop_.Contains(patchpoint_id);
}


// TODO(llvm): I haven't yet decided if it's profitable to use llvm statepoint
// mechanism to place safepoint polls. This function should either be used
// or removed.
void LLVMChunkBuilder::CreateSafepointPollFunction() {
  DCHECK(module_);
  DCHECK(llvm_ir_builder_);
  auto new_func = module_->getOrInsertFunction("gc.safepoint_poll",
                                               __ getVoidTy(), nullptr);
  safepoint_poll_ = llvm::cast<llvm::Function>(new_func);
  __ SetInsertPoint(NewBlock("Safepoint poll entry", safepoint_poll_));
  __ CreateRetVoid();
}

LLVMChunkBuilder& LLVMChunkBuilder::Build() {
  llvm::LLVMContext& llvm_context = LLVMGranularity::getInstance().context();
  chunk_ = new(zone()) LLVMChunk(info(), graph());
  module_ = LLVMGranularity::getInstance().CreateModule();
  module_->setTargetTriple(LLVMGranularity::x64_target_triple);
  llvm_ir_builder_ = llvm::make_unique<llvm::IRBuilder<>>(llvm_context);
  pointers_.clear();
  Types::Init(llvm_ir_builder_.get());
  status_ = BUILDING;

  // If compiling for OSR, reserve space for the unoptimized frame,
  // which will be subsumed into this frame.
  //if (graph()->has_osr()) {
  //  for (int i = graph()->osr()->UnoptimizedFrameSlots(); i > 0; i--) {
  //    chunk()->GetNextSpillIndex(GENERAL_REGISTERS);
  //  }
  //}

  // TODO(llvm): decide whether to have llvm insert safepoint polls.
  //  CreateSafepointPollFunction();

  // First param is context (v8, js context) which goes to rsi,
  // second param is the callee's JSFunction object (rdi),
  // third param is rbx for detecting osr entry,
  // fourth param is Parameter 0 which is `this`.
  int num_parameters = info()->num_parameters() + 4;

  std::vector<llvm::Type*> params(num_parameters, Types::tagged);
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      Types::tagged, params, false);
  function_ = llvm::cast<llvm::Function>(
      module_->getOrInsertFunction(module_->getModuleIdentifier(),
                                   function_type));

  llvm::AttributeSet attr_set = function_->getAttributes();
  // rbp based frame so the runtime can walk the stack as before
  attr_set = attr_set.addAttribute(llvm_context,
                                   llvm::AttributeSet::FunctionIndex,
                                   "no-frame-pointer-elim", "true");
  // Emit jumptables to .text instead of .rodata so relocation is easy.
  attr_set = attr_set.addAttribute(llvm_context,
                                   llvm::AttributeSet::FunctionIndex,
                                   "put-jumptable-in-fn-section", "true");
  // Same for constant pools.
  attr_set = attr_set.addAttribute(llvm_context,
                                   llvm::AttributeSet::FunctionIndex,
                                   "put-constantpool-in-fn-section", "true");

  function_->setAttributes(attr_set);
  function_->setGC(kGcStrategyName);
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
      llvm::BasicBlock* operand_block = operand->block()->llvm_end_basic_block();
      // The problem is that in hydrogen there are Phi nodes whit parameters
      // which are located in the same block. string-base64 -> base64ToString
      // This parameters then translted into gap instructions in  the phi predecessor blocks.
      DCHECK(phi->OperandCount() ==  phi->block()->predecessors()->length());
      operand_block = phi->block()->predecessors()->at(j)->llvm_end_basic_block();
      // We need this, otherwise we  will insert Use(operand) in the last block
      __ SetInsertPoint(operand_block);
      llvm_phi->addIncoming(Use(operand), operand_block);
      
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
      return Types::smi;
    case Representation::Kind::kDouble:
      return Types::float64;
    default:
      //UNIMPLEMENTED(); //3d-cube->DrawQube flows here
      return nullptr;
  }
}

// FIXME(llvm): unused (remove).
std::vector<llvm::Value*> LLVMChunkBuilder::GetSafepointValues(
    HInstruction* call_instr) {
  // TODO(llvm): Refactor out the AssingPointerMap() part.

  // TODO(llvm): obviously it's a very naive and unoptimal algorithm.
  // Why? Because the biggest improvement in performance is the
  // non-working-to-working. Other words: optimization would be premature.

  std::vector<llvm::Value*> mapped_values;

  // FIXME(llvm): what about phi-functions?
  const ZoneList<HBasicBlock*>* blocks = graph()->blocks();
  for (int i = 0; i < blocks->length(); i++) {
    HBasicBlock* block = blocks->at(i);
    // Iterate over all instructions of the graph.
    for (HInstructionIterator it(block); !it.Done(); it.Advance()) {
      HInstruction* def = it.Current();
      if (!def->llvm_value()) continue;
      if (def == call_instr) continue;
      if (!HasTaggedValue(def)) continue;
      if (!def->Dominates(call_instr)) continue;
      for (HUseIterator it(def->uses()); !it.Done(); it.Advance()) {
        HValue* use = it.value();
        if (use == HValue::cast(call_instr)) continue;
        if (use->IsReacheableFrom(call_instr)) {
          // Use(def) goes to stackmap
          if (def->llvm_value()) mapped_values.push_back(def->llvm_value());
          break;
        }
      }
    }
  }
  return mapped_values;
}

void LLVMChunkBuilder::DoDummyUse(HInstruction* instr) {
  Representation r = instr->representation();
  llvm::Type* type = GetLLVMType(r);
  llvm::Value* dummy_constant = nullptr;
  if (r.IsInteger32()) {
    dummy_constant = __ getInt32(0xdead);
  } else {
    dummy_constant = __ getInt64(0xdead);
  }
  auto casted_dummy_constant = dummy_constant;
  if (type)
    casted_dummy_constant = __ CreateBitOrPointerCast(dummy_constant, type);
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

  argument_count_ += current->argument_delta();
  DCHECK(argument_count_ >= 0);

  current_instruction_ = old_current;
}

llvm::Value* LLVMChunkBuilder::CreateConstant(HConstant* instr,
                                              HBasicBlock* block) {
  Representation r = instr->representation();
  if (r.IsSmi()) {
    // TODO(llvm): use/write a function for that
    // FIXME(llvm): this block was not tested
    int64_t int32_value = instr->Integer32Value();
    // FIXME(llvm): getInt64 takes uint64_t! And we want to pass signed int64.
    return __ getInt64(int32_value << (kSmiShift));
  } else if (r.IsInteger32()) {
    return __ getInt32(instr->Integer32Value());
  } else if (r.IsDouble()) {
    return llvm::ConstantFP::get(Types::float64,
                                 instr->DoubleValue());
  } else if (r.IsExternal()) {
    // TODO(llvm): tagged type
    // TODO(llvm): RelocInfo::EXTERNAL_REFERENCE
    Address external_address = instr->ExternalReferenceValue().address();
    auto as_i64 = __ getInt64(reinterpret_cast<uint64_t>(external_address));
    return __ CreateBitOrPointerCast(as_i64, Types::tagged);
  } else if (r.IsTagged()) {
    AllowHandleAllocation allow_handle_allocation;
    AllowHeapAllocation allow_heap_allocation;
    Handle<Object> object = instr->handle(isolate());
    auto current_block = __ GetInsertBlock();
    if (block) {
      // TODO(llvm): use binary search or a search tree.
      // if constant is alredy defined in block, return that const
      for (int i = 0; i < block->defined_consts()->length(); ++i) {
         HValue* constant = block->defined_consts()->at(i);
         if (constant->id() == instr->id()) {
           DCHECK(constant->llvm_value());
           return constant->llvm_value();
         }
      }
      // Record constant in the current block
      block->RecordConst(instr);
      auto const_block = block->llvm_start_basic_block();
      // Define objects constant in the "first llvm_block"
      // to avoid domination problem
      __ SetInsertPoint(const_block);
    }
    auto result = MoveHeapObject(object);
    instr->set_llvm_value(result);
    __ SetInsertPoint(current_block);
    return result;
  } else {
    UNREACHABLE();
    llvm::Value* fictive_value = nullptr;
    return fictive_value;
  }
}

llvm::BasicBlock* LLVMChunkBuilder::NewBlock(const std::string& name,
                                             llvm::Function* function) {
  LLVMContext& llvm_context = LLVMGranularity::getInstance().context();
  if (!function) function = function_;
  return llvm::BasicBlock::Create(llvm_context, name, function);
}

llvm::BasicBlock* LLVMChunkBuilder::Use(HBasicBlock* block) {
  if (!block->llvm_start_basic_block()) {
    llvm::BasicBlock* llvm_block = NewBlock(
        std::string("BlockEntry") + std::to_string(block->block_id()));
    block->set_llvm_start_basic_block(llvm_block);
  }
  DCHECK(block->llvm_start_basic_block());
  return block->llvm_start_basic_block();
}

llvm::Value* LLVMChunkBuilder::Use(HValue* value) {
  if (value->EmitAtUses()) {
    HInstruction* instr = HInstruction::cast(value);
    VisitInstruction(instr);
  }
  DCHECK(value->llvm_value());
  DCHECK_EQ(value->llvm_value()->getType(),
            GetLLVMType(value->representation()));
  if (HasTaggedValue(value))
    pointers_.insert(value->llvm_value());
  return value->llvm_value();
}

llvm::Value* LLVMChunkBuilder::SmiToInteger32(HValue* value) {
  llvm::Value* res = nullptr;
  if (SmiValuesAre32Bits()) {
    // The smi can have tagged representation.
    auto as_smi = __ CreateBitOrPointerCast(Use(value), Types::smi);
    res = __ CreateLShr(as_smi, kSmiShift);
    res = __ CreateTrunc(res, Types::i32);
  } else {
    DCHECK(SmiValuesAre31Bits());
    UNIMPLEMENTED();
    // TODO(llvm): just implement sarl(dst, Immediate(kSmiShift));
  }
  return res;
}

llvm::Value* LLVMChunkBuilder::SmiCheck(llvm::Value* value, bool negate) {
  llvm::Value* value_as_smi = __ CreateBitOrPointerCast(value, Types::smi);
  llvm::Value* res = __ CreateAnd(value_as_smi, __ getInt64(1));
  return __ CreateICmp(negate ? llvm::CmpInst::ICMP_NE : llvm::CmpInst::ICMP_EQ,
      res, __ getInt64(0));
}

void LLVMChunkBuilder::InsertDebugTrap() {
  llvm::Function* debug_trap = llvm::Intrinsic::getDeclaration(
      module_.get(), llvm::Intrinsic::debugtrap);
  __ CreateCall(debug_trap);
}

void LLVMChunkBuilder::Assert(llvm::Value* condition,
                              llvm::BasicBlock* next_block) {
  if (!next_block) next_block = NewBlock("After assertion");
  auto fail = NewBlock("Fail assertion");
  __ CreateCondBr(condition, next_block, fail);
  __ SetInsertPoint(fail);
  InsertDebugTrap();
  __ CreateUnreachable();
  __ SetInsertPoint(next_block);
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
  auto calling_conv = llvm::CallingConv::C; // We don't really care.
  auto empty = std::vector<llvm::Value*>();
  bool record_safepoint = false;
  return CallVal(target_adderss, calling_conv, empty, __ getVoidTy(),
                 record_safepoint);
}

llvm::Value* LLVMChunkBuilder::CallVal(llvm::Value* callable_value,
                                       llvm::CallingConv::ID calling_conv,
                                       std::vector<llvm::Value*>& params,
                                       llvm::Type* return_type,
                                       bool record_safepoint) {
  bool is_var_arg = false;

  if (!return_type)
    return_type = __ getVoidTy();

  std::vector<llvm::Type*> param_types;
  for (auto param : params)
    param_types.push_back(param->getType());
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      return_type, param_types, is_var_arg);
  llvm::PointerType* ptr_to_function = function_type->getPointerTo();
  auto casted = __ CreateBitOrPointerCast(callable_value, ptr_to_function);

  llvm::CallInst* call_inst = __ CreateCall(casted, params);
  call_inst->setCallingConv(calling_conv);

  if (record_safepoint) {
    int32_t stackmap_id = reloc_data_->GetNextSafepointPatchpointId();
    call_inst->addAttribute(llvm::AttributeSet::FunctionIndex,
                            "statepoint-id", std::to_string(stackmap_id));
  } else {
    call_inst->addAttribute(llvm::AttributeSet::FunctionIndex,
                            "no-statepoint-please", "true");
  }

  return call_inst;
}

llvm::Value* LLVMChunkBuilder::CallCode(Handle<Code> code,
                                        llvm::CallingConv::ID calling_conv,
                                        std::vector<llvm::Value*>& params,
                                        bool record_safepoint) {
  auto index = chunk()->masm().GetCodeTargetIndex(code);
  int nop_size;
  int32_t pp_id;
  if (code->kind() == Code::BINARY_OP_IC ||
      code->kind() == Code::COMPARE_IC) {
    pp_id = reloc_data_->GetNextRelocNopPatchpointId(record_safepoint);
    nop_size = 6; // call relative i32 takes 5 bytes: `e8` + i32 + nop
  } else {
    pp_id = reloc_data_->GetNextRelocPatchpointId(record_safepoint);
    nop_size = 5; // call relative i32 takes 5 bytes: `e8` + i32
  }

  // If we didn't have to also have a safe point at the call site,
  // simple call to the patchpoint intrinsic would suffice. However
  // LLVM does not support statepoints upon patchpoints (or any other intrinsics
  // for that matter). Luckily, patchpoint's functionality is a subset of that
  // of the statepoint intrinsic.
  auto llvm_null = llvm::ConstantPointerNull::get(Types::ptr_i8);
  auto result = CallStatePoint(pp_id, llvm_null, calling_conv, params, nop_size);

  // Map pp_id -> index in code_targets_.
  chunk()->target_index_for_ppid()[pp_id] = index;


  if (code->kind() == Code::BINARY_OP_IC ||
      code->kind() == Code::COMPARE_IC) {
    // This will be optimized out anyway
    llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
       llvm::Intrinsic::donothing);
    __ CreateCall(intrinsic);
  }

  return result;
}

llvm::Value* LLVMChunkBuilder::CallAddress(Address target,
                                           llvm::CallingConv::ID calling_conv,
                                           std::vector<llvm::Value*>& params,
                                           llvm::Type* return_type) {
  auto val_addr = __ getInt64(reinterpret_cast<uint64_t>(target));
  return CallVal(val_addr, calling_conv, params, return_type);
}

llvm::Value* LLVMChunkBuilder::CallRuntimeViaId(Runtime::FunctionId id) {
 return CallRuntime(Runtime::FunctionForId(id));
}

llvm::Value* LLVMChunkBuilder::CallRuntime(const Runtime::Function* function) {
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
  }

  // 1) emit relative 32 call to index which would follow the calling convention
  // 2) record reloc info when we know the pc offset (RelocInfo::CODE...)


  auto llvm_nargs = __ getInt64(arg_count);
  auto target_temp = __ getInt64(reinterpret_cast<uint64_t>(rt_target));
  auto llvm_rt_target = target_temp; //__ CreateIntToPtr(target_temp, Types::ptr_i8);
  auto context = GetContext();
  std::vector<llvm::Value*> args(arg_count + 3, nullptr);
  args[0] = llvm_nargs;
  args[1] = llvm_rt_target;
  args[2] = context;

  for (int i = 0; i < pending_pushed_args_.length(); i++) {
    args[arg_count + 3 - 1 - i] = pending_pushed_args_[i];
  }
  pending_pushed_args_.Clear();

  return CallCode(code, llvm::CallingConv::X86_64_V8_CES, args);
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
    AllowHeapAllocation allow_heap;
    code = ces.GetCode();
    // FIXME(llvm,gc): respect reloc info mode...
  }

  // bool is_var_arg = false;
  auto llvm_nargs = __ getInt64(arg_count);
  auto target_temp = __ getInt64(reinterpret_cast<uint64_t>(rt_target));
  auto llvm_rt_target = __ CreateIntToPtr(target_temp, Types::ptr_i8);
  std::vector<llvm::Value*> actualParams;
  actualParams.push_back(llvm_nargs);
  actualParams.push_back(llvm_rt_target);
  actualParams.push_back(context);
  for (auto i = 0; i < params.size(); ++i)
     actualParams.push_back(params[i]);
  llvm::Value* call_inst = CallCode(code, llvm::CallingConv::X86_64_V8_CES, actualParams);
  return call_inst;
}


llvm::Value* LLVMChunkBuilder::FieldOperand(llvm::Value* base, int offset) {
  // The problem is (volatile_0 + imm) + offset == volatile_0 + (imm + offset),
  // so...
  auto offset_val = ConstFoldBarrier(__ getInt64(offset - kHeapObjectTag));
  // I don't know why, but it works OK even if base was already an i8*
  llvm::Value* base_casted = __ CreateIntToPtr(base, Types::ptr_i8);
  return __ CreateGEP(base_casted, offset_val);
}

// TODO(llvm): It should probably become 'load field operand as type'
// with tagged as default.
llvm::Value* LLVMChunkBuilder::LoadFieldOperand(llvm::Value* base, int offset,
                                                const char* name) {
  llvm::Value* address = FieldOperand(base, offset);
  llvm::Value* casted_address = __ CreatePointerCast(address,
                                                     Types::ptr_tagged);
  return __ CreateLoad(casted_address, name);
}

llvm::Value* LLVMChunkBuilder::ConstructAddress(llvm::Value* base, int64_t offset) {
  // The problem is (volatile_0 + imm) + offset == volatile_0 + (imm + offset),
  // so...
  llvm::Value* offset_val = ConstFoldBarrier(__ getInt64(offset));
  llvm::Value* base_casted = __ CreateBitOrPointerCast(base, Types::ptr_i8);
  auto constructed_address = __ CreateGEP(base_casted, offset_val);
  return __ CreateBitOrPointerCast(constructed_address, base->getType());
}

llvm::Value* LLVMChunkBuilder::ValueFromSmi(Smi* smi) {
   intptr_t intptr_value = reinterpret_cast<intptr_t>(smi);
   llvm::Value* value = __ getInt64(intptr_value);
   return value;
}

llvm::Value* LLVMChunkBuilder::MoveHeapObject(Handle<Object> object) {
  if (object->IsSmi()) {
    // TODO(llvm): use/write a function for that
    Smi* smi = Smi::cast(*object);
    llvm::Value* value = ValueFromSmi(smi);
    return __ CreateBitOrPointerCast(value, Types::tagged);
  } else { // Heap object
    // MacroAssembler::MoveHeapObject
    AllowHeapAllocation allow_allocation;
    AllowHandleAllocation allow_handles;
    DCHECK(object->IsHeapObject());
    if (isolate()->heap()->InNewSpace(*object)) {
      Handle<Cell> new_cell = isolate()->factory()->NewCell(object);
      llvm::Value* value = Move(new_cell, RelocInfo::CELL);
      llvm::BasicBlock* current_block = __ GetInsertBlock();
      auto last_instr = current_block-> getTerminator();
      // if block has terminator we must insert before it
      if (!last_instr) {
        llvm::Value* ptr = __ CreateBitOrPointerCast(value, Types::ptr_tagged);
        return  __ CreateLoad(ptr);
      }
      llvm::Value* ptr = new llvm::BitCastInst(value, Types::ptr_tagged, "", last_instr);
      return new llvm::LoadInst(ptr, "", last_instr);
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
  auto the_pointer = RecordRelocInfo(intptr_value, rmode);
  pointers_.insert(the_pointer);
  return the_pointer;
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
  auto object_as_i64 = __ CreateBitOrPointerCast(object, Types::i64);
  auto masked_object = __ CreateAnd(object_as_i64, page_align_mask,
                                    "CheckPageFlag1");
  auto flags_address = ConstructAddress(masked_object,
                                        MemoryChunk::kFlagsOffset);
  auto i32_ptr_flags_address = __ CreateBitOrPointerCast(flags_address,
                                                         Types::ptr_i32);
  auto flags = __ CreateLoad(i32_ptr_flags_address);
  auto llvm_mask = __ getInt32(mask);
  auto and_result = __ CreateAnd(flags, llvm_mask);
  return __ CreateICmpEQ(and_result, __ getInt32(0), "CheckPageFlag");
}

llvm::Value* LLVMChunkBuilder::AllocateHeapNumberSlow(HValue* unuse, llvm::Value* null_ptr) {

  // return an i8*
  llvm::Value* allocated = CallRuntimeViaId(Runtime::kAllocateHeapNumber);
  // RecordSafepointWithRegisters...
  return allocated;
}

void LLVMChunkBuilder::UpdateAllocationTopHelper(llvm::Value* result_end,
                                                 AllocationFlags flags) {
  if (emit_debug_code()) {
    UNIMPLEMENTED();
  }

  ExternalReference allocation_top =
      AllocationUtils::GetAllocationTopReference(isolate(), flags);
  // Update new top.
  llvm::Value* top_address = __ getInt64(reinterpret_cast<uint64_t>
                                             (allocation_top.address()));
  llvm::Value* address = __ CreateIntToPtr(top_address, Types::ptr_i64);
  __ CreateStore(result_end, address);
}

llvm::Value* LLVMChunkBuilder::LoadAllocationTopHelper(AllocationFlags flags) {
  ExternalReference allocation_top =
      AllocationUtils::GetAllocationTopReference(isolate(), flags);
  // Just return if allocation top is already known.
  if ((flags & RESULT_CONTAINS_TOP) != 0) {
    UNIMPLEMENTED();
  }
  // Safe code.
  llvm::Value* top_address = __ getInt64(reinterpret_cast<uint64_t>
                                             (allocation_top.address()));
  llvm::Value* address = __ CreateIntToPtr(top_address, Types::ptr_i64);
  llvm::Value* result = __ CreateLoad(address);
  return result;
}


llvm::Value* LLVMChunkBuilder::AllocateHeapNumber(MutableMode mode){
  llvm::Value* (LLVMChunkBuilder::*fptr)(HValue*, llvm::Value*);
  fptr = &LLVMChunkBuilder::AllocateHeapNumberSlow;
  llvm::Value* result = Allocate(__ getInt32(HeapNumber::kSize), fptr, TAG_OBJECT);
  Heap::RootListIndex map_index = mode == MUTABLE
      ? Heap::kMutableHeapNumberMapRootIndex
      : Heap::kHeapNumberMapRootIndex;
  // Set the map.
  llvm::Value* root = LoadRoot(map_index);
  llvm::Value* address = FieldOperand(result, HeapObject::kMapOffset);
  llvm::Value* casted_address = __ CreatePointerCast(address,
                                                     Types::ptr_tagged);
  __ CreateStore(root, casted_address);
  __ CreatePtrToInt(casted_address, Types::i64);
  return result;
}

llvm::Value* LLVMChunkBuilder::GetContext() {
  // First parameter is our context (rsi).
  return function_->arg_begin();
}

llvm::Value* LLVMChunkBuilder::GetNan() {
  auto zero = llvm::ConstantFP::get(Types::float64, 0);
  return __ CreateFDiv(zero, zero);
}

LLVMEnvironment* LLVMChunkBuilder::AssignEnvironment() {
  HEnvironment* hydrogen_env = current_block_->last_environment();
  int argument_index_accumulator = 0;
  ZoneList<HValue*> objects_to_materialize(0, zone());
  return CreateEnvironment(
      hydrogen_env, &argument_index_accumulator, &objects_to_materialize);
}

void LLVMChunkBuilder::GetAllEnvironmentValues(
    LLVMEnvironment* environment, std::vector<llvm::Value*>& mapped_values) {
  if (!environment) return;
  GetAllEnvironmentValues(environment->outer(), mapped_values);
  for (auto val : *environment->values())
    mapped_values.push_back(val);
}

void LLVMChunkBuilder::DeoptimizeIf(llvm::Value* compare,
                                    Deoptimizer::DeoptReason deopt_reason,
                                    bool negate,
                                    llvm::BasicBlock* next_block) {
  LLVMEnvironment* environment = AssignEnvironment();
  auto patchpoint_id = reloc_data_->GetNextDeoptRelocPatchpointId();
  deopt_data_->Add(environment, patchpoint_id);
  int bailout_id = deopt_data_->DeoptCount() - 1;
  reloc_data_->SetBailoutId(patchpoint_id, bailout_id);
  reloc_data_->SetDeoptReason(patchpoint_id, deopt_reason);

  if (!next_block) next_block = NewBlock("BlockCont");
  llvm::BasicBlock* saved_insert_point = __ GetInsertBlock();

  if (FLAG_deopt_every_n_times != 0 && !info()->IsStub()) UNIMPLEMENTED();
  if (info()->ShouldTrapOnDeopt()) {
    // Our trap on deopt does not allow to proceed to the actual deopt.
    // It could be avoided if we ever need this though. But be prepared
    // that implementation would involve some careful BB management.
    if (!negate) { // We assert !compare, so if negate, we assert !!compare.
      auto one = true;
      compare = __ CreateXor(__ getInt1(one), compare);
    }
    Assert(compare, next_block);
    return;
  }

  Deoptimizer::BailoutType bailout_type = info()->IsStub()
      ? Deoptimizer::LAZY
      : Deoptimizer::EAGER;
  DCHECK_EQ(bailout_type, Deoptimizer::EAGER); // We don't support lazy yet.

  Address entry;
  {
    AllowHandleAllocation allow;
    // TODO(llvm): what if we use patchpoint_id here?
    entry = Deoptimizer::GetDeoptimizationEntry(isolate(),
        bailout_id, bailout_type);
  }
  if (entry == NULL) {
    Abort(kBailoutWasNotPrepared);
    return;
  }

  // TODO(llvm): create Deoptimizer::DeoptInfo & Deoptimizer::JumpTableEntry (?)

  llvm::BasicBlock* deopt_block = NewBlock("DeoptBlock");
  __ SetInsertPoint(deopt_block);

  std::vector<llvm::Value*> mapped_values;
  GetAllEnvironmentValues(environment, mapped_values);

  // Store offset relative to the Code Range start. It always fits in 32 bits.
  // Will be fixed up later (in Code::CopyFrom).
  Address start = isolate()->code_range()->start();
  chunk()->deopt_target_offset_for_ppid()[patchpoint_id] = entry - start;

  std::vector<llvm::Value*> empty;
  int nop_size = 5; // Call relative i32 takes 5 bytes: `e8` + i32
  auto llvm_null = llvm::ConstantPointerNull::get(Types::ptr_i8);
  CallPatchPoint(patchpoint_id, llvm_null, empty, mapped_values, nop_size);
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

bool LLVMChunkBuilder::HasTaggedValue(HValue* value) {
  return value != NULL &&
      value->representation().IsTagged() && !value->type().IsSmi();
}

class PassInfoPrinter {
 public:
  PassInfoPrinter(const char* name, llvm::Module* module)
     : name_(name),
       module_(module) {
    USE(name_);
    USE(module_);
#if DEBUG
    if (!only_after) {
      llvm::errs() << filler << "vvv Module BEFORE " << name_ <<" vvv"
          << filler << "\n";
      llvm::errs() << *module_;
      llvm::errs() << filler << "^^^ Module BEFORE " << name_ <<" ^^^"
          << filler << "\n";
      only_after = true;
    }
#endif
  }
  ~PassInfoPrinter() {
#if DEBUG
    llvm::errs() << filler << "vvv Module  AFTER " << name_ <<" vvv"
        << filler << "\n";
    llvm::errs() << *module_;
    llvm::errs() << filler << "^^^ Module  AFTER " << name_ <<" ^^^"
        << filler << "\n";
#endif
  }
 private:
  static bool only_after;
  static const char* filler;
  const char* name_;
  llvm::Module* module_;
};

const char* PassInfoPrinter::filler = "====================";
bool PassInfoPrinter::only_after = false;

// PlaceStatePoints and RewriteStatePoints may move things around a bit
// (by deleting and adding instructions) so we can't refer to anything
// by llvm::Value*.
// This function gives names (which are preserved) to the values we want
// to track.
// Warning: same method may not work for all transformation passes,
// because names might not be preserved.
LLVMChunkBuilder& LLVMChunkBuilder::GiveNamesToPointerValues() {
  PassInfoPrinter printer("GiveNamesToPointerValues", module_.get());
  DCHECK_EQ(number_of_pointers_, -1);
  number_of_pointers_ = 0;
  for (auto value : pointers_) {
      value->setName(kPointersPrefix + std::to_string(number_of_pointers_++));
  }
  // Now we have names. llvm::Value*s will soon become invalid.
  pointers_.clear();
  return *this;
}

void LLVMChunkBuilder::DumpPointerValues() {
  DCHECK_GE(number_of_pointers_, 0);
#ifdef DEBUG
  std::cerr << "< POINTERS:" << "\n";
  for (auto i = 0 ; i < number_of_pointers_; i++) {
    std::string name = kPointersPrefix + std::to_string(i);
    auto value = function_->getValueSymbolTable().lookup(name);
    if (value)
      llvm::errs() << value->getName() << " | " << *value << "\n";
  }
  std::cerr << "POINTERS >" << "\n";
#endif
}

LLVMChunkBuilder& LLVMChunkBuilder::NormalizePhis() {
  PassInfoPrinter printer("normalization", module_.get());
  llvm::legacy::FunctionPassManager pass_manager(module_.get());
  if (FLAG_phi_normalize) pass_manager.add(createNormalizePhisPass());
  pass_manager.doInitialization();
  pass_manager.run(*function_);
  return *this;
}

LLVMChunkBuilder& LLVMChunkBuilder::PlaceStatePoints() {
  PassInfoPrinter printer("PlaceStatePoints", module_.get());
  DumpPointerValues();
  llvm::legacy::FunctionPassManager pass_manager(module_.get());
  pass_manager.add(llvm::createPlaceSafepointsPass());
  pass_manager.doInitialization();
  pass_manager.run(*function_);
  pass_manager.doFinalization();
  return *this;
}

LLVMChunkBuilder& LLVMChunkBuilder::RewriteStatePoints() {
  PassInfoPrinter printer("RewriteStatepointsForGC", module_.get());
  DumpPointerValues();

  std::set<llvm::Value*> pointer_values;
  for (auto i = 0 ; i < number_of_pointers_; i++) {
    std::string name = kPointersPrefix + std::to_string(i);
    auto value = function_->getValueSymbolTable().lookup(name);
    if (value)
      pointer_values.insert(value);
  }

  llvm::legacy::PassManager pass_manager;
  pass_manager.add(v8::internal::createRewriteStatepointsForGCPass(
      pointer_values));
  pass_manager.run(*module_.get());
  return *this;
}


LLVMChunkBuilder& LLVMChunkBuilder::Optimize() {
  DCHECK(module_);
#ifdef DEBUG
  llvm::verifyFunction(*function_, &llvm::errs());
#endif
  PassInfoPrinter printer("optimization", module_.get());

  LLVMGranularity::getInstance().OptimizeFunciton(module_.get(), function_);
  LLVMGranularity::getInstance().OptimizeModule(module_.get());
  return *this;
}

// FIXME(llvm): obsolete.
void LLVMChunkBuilder::CreateVolatileZero() {
  volatile_zero_address_ = __ CreateAlloca(Types::i64);
  bool is_volatile = true;
  __ CreateStore(__ getInt64(0), volatile_zero_address_, is_volatile);
}

llvm::Value* LLVMChunkBuilder::GetVolatileZero() {
  bool is_volatile = true;
  return __ CreateLoad(volatile_zero_address_, is_volatile, "volatile_zero");
}

llvm::Value* LLVMChunkBuilder::ConstFoldBarrier(llvm::Value* imm) {
//  return __ CreateAdd(GetVolatileZero(), imm);
  return imm;
}

void LLVMChunkBuilder::PatchReceiverToGlobalProxy() {
  if (info()->MustReplaceUndefinedReceiverWithGlobalProxy()) {
    auto receiver = GetParameter(0); // `this'
    auto is_undefined = CompareRoot(receiver, Heap::kUndefinedValueRootIndex);
    auto patch_receiver = NewBlock("Patch receiver");
    auto receiver_ok = NewBlock("Receiver OK");
    __ CreateCondBr(is_undefined, patch_receiver, receiver_ok);
    __ SetInsertPoint(patch_receiver);
    auto context_as_ptr_to_tagged = __ CreateBitOrPointerCast(GetContext(),
                                                              Types::ptr_tagged);
    auto global_object_operand_address = ConstructAddress(
        context_as_ptr_to_tagged,
        Context::SlotOffset(Context::GLOBAL_OBJECT_INDEX));
    auto global_object_operand = __ CreateLoad(global_object_operand_address);
    auto global_receiver = LoadFieldOperand(global_object_operand,
                                            GlobalObject::kGlobalProxyOffset);
    __ CreateBr(receiver_ok);
    __ SetInsertPoint(receiver_ok);
    auto phi = __ CreatePHI(Types::tagged, 2);
    phi->addIncoming(receiver, receiver_ok);
    phi->addIncoming(global_receiver, patch_receiver);
    global_receiver_ = phi;
  }
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
    PatchReceiverToGlobalProxy();
    // Ensure every function has an associated Stack Map section.
    std::vector<llvm::Value*> empty;
    CallStackMap(reloc_data_->GetNextUnaccountedPatchpointId(), empty);

    //If function contains OSR entry, it's first instruction must be osr_branch
    if (graph_->has_osr()) { 
      osr_preserved_values_.Clear();
      // We need to move llvm spill index by UnoptimizedFrameSlots count
      // in order to preserve Full-Codegen local values
      for (int i = 0; i < graph_->osr()->UnoptimizedFrameSlots(); ++i) {
         auto alloc = __ CreateAlloca(Types::tagged);
         osr_preserved_values_.Add(alloc, info()->zone());
      }
      HBasicBlock* osr_block = graph_->osr()->osr_entry();
      llvm::BasicBlock* not_osr_target = NewBlock("NO_OSR_CONTINUE");
      llvm::BasicBlock* osr_target = Use(osr_block);
      llvm::Value* zero = __ getInt64(0);
      llvm::Value* zero_as_tagged = __ CreateBitOrPointerCast(zero,
                                                              Types::tagged);
      llvm::Function::arg_iterator it = function_->arg_begin();
      int i = 0;
      while (++i < 3) ++it;
      llvm::Value* osr_value  = it;
      // Branch to OSR block
      llvm::Value* compare = __ CreateICmpEQ(osr_value, zero_as_tagged);
      __ CreateCondBr(compare, not_osr_target, osr_target);
      __ SetInsertPoint(not_osr_target);
    }
    // CreateVolatileZero();
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
      if (value->IsConstant()) {
        HConstant* instr = HConstant::cast(value);
        op = CreateConstant(instr, current_block_);
      } else {
        op = Use(value);
      }
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

llvm::Value* LLVMChunkBuilder::GetParameter(int index) {
  DCHECK_GE(index, 0);

  // `this' might be patched.
  if (index == 0 && global_receiver_)
    return global_receiver_;

  int num_parameters = info()->num_parameters() + 4;
  llvm::Function::arg_iterator it = function_->arg_begin();
  // First off, skip first 2 parameters: context (rsi)
  // and callee's JSFunction object (rdi).
  // Now, I couldn't find a way to tweak the calling convention through LLVM
  // in a way that parameters are passed left-to-right on the stack.
  // So for now they are passed right-to-left, as in cdecl.
  // And therefore we do the magic here.
  index = -index;
  while (--index + num_parameters > 0) ++it;

  return it;
}

void LLVMChunkBuilder::DoParameter(HParameter* instr) {
  instr->set_llvm_value(GetParameter(instr->index()));
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
//  llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
//      llvm::Intrinsic::read_register, { Types::i64 });
//  auto metadata =
//    llvm::MDNode::get(llvm_context, llvm::MDString::get(llvm_context, "rsp"));
//  llvm::MetadataAsValue* val = llvm::MetadataAsValue::get(
//      llvm_context, metadata);
//  llvm::Value* rsp_value = __ CreateCall(intrinsic, val);
//  auto above_equal = CompareRoot(rsp_value, Heap::kStackLimitRootIndex,
//                                 llvm::CmpInst::ICMP_UGE);
//  Assert(above_equal);
}

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

// Note: we don't set calling convention here.
// We return the call instruction so the caller can do it.
llvm::CallInst* LLVMChunkBuilder::CallPatchPoint(
    int32_t stackmap_id,
    llvm::Value* target_function,
    std::vector<llvm::Value*>& function_args,
    std::vector<llvm::Value*>& live_values,
    int covering_nop_size) {
  llvm::Function* patchpoint = llvm::Intrinsic::getDeclaration(
      module_.get(), llvm::Intrinsic::experimental_patchpoint_i64);

  auto llvm_patchpoint_id = __ getInt64(stackmap_id);
  auto nop_size = __ getInt32(covering_nop_size);
  auto num_args = __ getInt32(IntHelper::AsUInt32(function_args.size()));

  std::vector<llvm::Value*>  patchpoint_args =
    { llvm_patchpoint_id, nop_size, target_function, num_args };

  patchpoint_args.insert(patchpoint_args.end(),
                         function_args.begin(), function_args.end());
  patchpoint_args.insert(patchpoint_args.end(),
                         live_values.begin(), live_values.end());

  return __ CreateCall(patchpoint, patchpoint_args);
}


// Returns the value of gc.result (call instruction would be irrelevant).
llvm::Value* LLVMChunkBuilder::CallStatePoint(
    int32_t stackmap_id,
    llvm::Value* target_function,
    llvm::CallingConv::ID calling_conv,
    std::vector<llvm::Value*>& function_args,
    int covering_nop_size) {

  auto return_type = Types::tagged;

  // The statepoint intrinsic is overloaded by the function pointer type.
  std::vector<llvm::Type*> params;
  for (int i = 0; i < function_args.size(); i++)
    params.push_back(function_args[i]->getType());
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      return_type, params, false);
  auto function_type_ptr = function_type->getPointerTo();
  llvm::Type* statepoint_arg_types[] =
    { llvm::cast<llvm::PointerType>(function_type_ptr) };

  auto casted_target = __ CreateBitOrPointerCast(target_function,
                                                 function_type_ptr);

  llvm::Function* statepoint = llvm::Intrinsic::getDeclaration(
      module_.get(), llvm::Intrinsic::experimental_gc_statepoint,
      statepoint_arg_types);

  auto llvm_patchpoint_id = __ getInt64(stackmap_id);
  auto nop_size = __ getInt32(covering_nop_size);
  auto num_args = __ getInt32(IntHelper::AsUInt32(function_args.size()));
  auto flags = __ getInt32(0);
  auto num_transition_args = __ getInt32(0);
  auto num_deopt_args = __ getInt32(0);

  std::vector<llvm::Value*>  statepoint_args =
    { llvm_patchpoint_id, nop_size, casted_target, num_args, flags };

  statepoint_args.insert(statepoint_args.end(),
                         function_args.begin(), function_args.end());

  statepoint_args.insert(statepoint_args.end(),
                         { num_transition_args, num_deopt_args });

  auto token = __ CreateCall(statepoint, statepoint_args);
  token->setCallingConv(calling_conv);

  llvm::Function* gc_result = llvm::Intrinsic::getDeclaration(
      module_.get(), llvm::Intrinsic::experimental_gc_result, { return_type });

  return __ CreateCall(gc_result, { token });
}

llvm::Value* LLVMChunkBuilder::RecordRelocInfo(uint64_t intptr_value,
                                               RelocInfo::Mode rmode) {
  bool extended = false;
  if (is_uint32(intptr_value)) {
    intptr_value = (intptr_value << 32) | kExtFillingValue;
    extended = true;
  }

  // Here we use the intptr_value (data) only to identify the entry in the map
  RelocInfo rinfo(rmode, intptr_value);
  LLVMRelocationData::ExtendedInfo meta_info;
  meta_info.cell_extended = extended;
  reloc_data_->Add(rinfo, meta_info);

  bool is_var_arg = false;
  auto return_type = Types::tagged;
  auto param_types = { Types::i64 };
  auto func_type = llvm::FunctionType::get(return_type, param_types,
                                           is_var_arg);
  // AT&T syntax.
  const char* asm_string = "movabsq $1, $0";
  // i = 64-bit integer (on x64), q = register (like r, but more regs allowed).
  const char* constraints = "=q,i,~{dirflag},~{fpsr},~{flags}";
  bool has_side_effects = true;
  llvm::InlineAsm* inline_asm = llvm::InlineAsm::get(func_type,
                                                     asm_string,
                                                     constraints,
                                                     has_side_effects);
  llvm::BasicBlock* current_block = __ GetInsertBlock();
  auto last_instr = current_block-> getTerminator();
  // if block has terminator we must insert before last instruction
  if (!last_instr) 
    return __ CreateCall(inline_asm, __ getInt64(intptr_value));
  auto call = llvm::CallInst::Create(inline_asm, __ getInt64(intptr_value), "reloc", last_instr);
  //call->insertBefore(last_instr);
  return call;
}

void LLVMChunkBuilder::DoConstant(HConstant* instr) {
  llvm::Value* const_value = CreateConstant(instr, current_block_);
  instr->set_llvm_value(const_value);
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
      DeoptimizeIf(overflow, Deoptimizer::kOverflow);
    }
  } else if (instr->representation().IsDouble()) {
      DCHECK(instr->left()->representation().IsDouble());
      DCHECK(instr->right()->representation().IsDouble());
      HValue* left = instr->BetterLeftOperand();
      HValue* right = instr->BetterRightOperand();
      llvm::Value* fadd = __ CreateFAdd(Use(left), Use(right));
      instr->set_llvm_value(fadd);
  } else if (instr->representation().IsExternal()) {
    //TODO: not tested string-validate-input.js in doTest
    DCHECK(instr->IsConsistentExternalRepresentation());
    DCHECK(!instr->CheckFlag(HValue::kCanOverflow));
    //FIXME: possibly wrong
    llvm::Value* left_as_i64 = __ CreateBitOrPointerCast(Use(instr->left()),
                                                         Types::i64);
    llvm::Value* right_as_i64 = __ CreateBitOrPointerCast(Use(instr->right()),
                                                          Types::i64);
    llvm::Value* sum = __ CreateAdd(left_as_i64, right_as_i64);
    llvm::Value* sum_as_external = __ CreateBitOrPointerCast(
        sum, GetLLVMType(Representation::External()));
    instr->set_llvm_value(sum_as_external);
  } else {
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoAllocateBlockContext(HAllocateBlockContext* instr) {
  UNIMPLEMENTED();
}


llvm::Value* LLVMChunkBuilder::Allocate(llvm::Value* object_size,
                                        llvm::Value* (LLVMChunkBuilder::*fptr)
                                                     (HValue* instr,
                                                      llvm::Value* temp),
                                        AllocationFlags flags,
                                        HValue* instr,
                                        llvm::Value* temp){
  DCHECK((flags & (RESULT_CONTAINS_TOP | SIZE_IN_WORDS)) == 0);
//  DCHECK(object_size <= Page::kMaxRegularHeapObjectSize);
  if (!FLAG_inline_new) {
    UNIMPLEMENTED();
  }

  // Load address of new object into result.
  llvm::Value* result = LoadAllocationTopHelper(flags);

  if ((flags & DOUBLE_ALIGNMENT) != 0) {
    if (kPointerSize == kDoubleSize) {
      if (FLAG_debug_code) {
        UNIMPLEMENTED();
      }
    } else {
      UNIMPLEMENTED();
    }
  }

  // Calculate new top and bail out if new space is exhausted.
  ExternalReference allocation_limit =
      AllocationUtils::GetAllocationLimitReference(isolate(), flags);
  llvm::BasicBlock* not_carry = NewBlock("Allocate add is correct");
  llvm::BasicBlock* merge = NewBlock("Allocate merge");
  llvm::BasicBlock* deferred = NewBlock("Allocate deferred");
  llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
          llvm::Intrinsic::uadd_with_overflow, Types::i64);
  llvm::Value* size = __ CreateIntCast(object_size, Types::i64, true);
  llvm::Value* params[] = { result, size };
  llvm::Value* call = __ CreateCall(intrinsic, params);
  llvm::Value* sum = __ CreateExtractValue(call, 0);
  llvm::Value* overflow = __ CreateExtractValue(call, 1);
  __ CreateCondBr(overflow, deferred, not_carry);

  __ SetInsertPoint(not_carry);
  llvm::Value* top_address = __ getInt64(reinterpret_cast<uint64_t>
                                           (allocation_limit.address()));
  llvm::Value* address = __ CreateIntToPtr(top_address, Types::ptr_i64);
  llvm::Value* limit_operand = __ CreateLoad(address);
  llvm::BasicBlock* limit_is_valid = NewBlock("Allocate limit is valid");
  llvm::Value* cmp_limit = __ CreateICmpUGT(sum, limit_operand);
  __ CreateCondBr(cmp_limit, deferred, limit_is_valid);

  __ SetInsertPoint(limit_is_valid);
  // Update allocation top.
  UpdateAllocationTopHelper(sum, flags);
  bool tag_result = (flags & TAG_OBJECT) != 0;
  llvm::Value* final_result = nullptr;
  if (tag_result) {
    llvm::Value* inc = __ CreateAdd(result, __ getInt64(1));
    final_result = __ CreateIntToPtr(inc, Types::tagged);
    __ CreateBr(merge);
  } else {
    final_result = __ CreateIntToPtr(result, Types::tagged);
    __ CreateBr(merge);
  }

  __ SetInsertPoint(deferred);
  llvm::Value* deferred_result = (this->*fptr)(instr, temp);
  __ CreateBr(merge);

  __ SetInsertPoint(merge);
  llvm::PHINode* phi = __ CreatePHI(Types::tagged, 2);
  phi->addIncoming(final_result, limit_is_valid);
  phi->addIncoming(deferred_result, deferred);
  return phi;
}


llvm::Value* LLVMChunkBuilder::AllocateSlow(HValue* obj, llvm::Value* temp){
  HAllocate* instr = HAllocate::cast(obj);
  std::vector<llvm::Value*> args;
  llvm::Value* arg1 = Integer32ToSmi(instr->size());
  int flags = 0;
  if (instr->IsOldSpaceAllocation()) {
    DCHECK(!instr->IsNewSpaceAllocation());
    flags = AllocateTargetSpace::update(flags, OLD_SPACE);
  } else {
    flags = AllocateTargetSpace::update(flags, NEW_SPACE);
  }

  llvm::Value* value = __ getInt32(flags);
  llvm::Value* arg2 = Integer32ToSmi(value);
  args.push_back(arg2);
  args.push_back(arg1);
  llvm::Value* alloc = CallRuntimeFromDeferred(Runtime::kAllocateInTargetSpace,
                                               Use(instr->context()), args);
  return alloc;
}


void LLVMChunkBuilder::DoAllocate(HAllocate* instr) {
  // Allocate memory for the object.
  AllocationFlags flags = TAG_OBJECT;
  if (instr->MustAllocateDoubleAligned()) {
    flags = static_cast<AllocationFlags>(flags | DOUBLE_ALIGNMENT);
  }
  if (instr->IsOldSpaceAllocation()) {
    DCHECK(!instr->IsNewSpaceAllocation());
    flags = static_cast<AllocationFlags>(flags | PRETENURE);
  }
  if (instr->MustPrefillWithFiller()) {
    UNIMPLEMENTED();
  }

  DCHECK(instr->size()->representation().IsInteger32());
  llvm::Value* size = Use(instr->size());
  llvm::Value* (LLVMChunkBuilder::*fptr)(HValue*, llvm::Value*);
  fptr = &LLVMChunkBuilder::AllocateSlow;
  llvm::Value* res = Allocate(size, fptr, flags, instr);
  instr->set_llvm_value(res);
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
      DCHECK(instr->HasNoUses());
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
    DeoptimizeIf(compare, Deoptimizer::kOutOfBounds, negate);
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
  llvm::Value* value_as_i64 = __ CreateBitOrPointerCast(value, Types::i64);

  if (expected.IsEmpty()) expected = ToBooleanStub::Types::Generic();

  std::vector<llvm::BasicBlock*> check_blocks;
  for (auto i = ToBooleanStub::UNDEFINED;
      i < ToBooleanStub::NUMBER_OF_TYPES;
      i = static_cast<ToBooleanStub::Type>(i + 1)) {
    if (i == ToBooleanStub::SMI){
      if (expected.Contains(i)){
        check_blocks.push_back(NewBlock("BranchTagged Check Block"));
      } else if (expected.NeedsMap()){
        check_blocks.push_back(NewBlock("BranchTagged NeedsMapCont"));
      }
      if (expected.NeedsMap()) {
        if (!expected.CanBeUndetectable())
          check_blocks.push_back(NewBlock("BranchTagged NonUndetachable"));
        else
          check_blocks.push_back(NewBlock("BranchTagged CanBeUndetectable"));
      }
    } else if (expected.Contains(i)) {
      check_blocks.push_back(NewBlock("BranchTagged Check Block"));
    }
  }
  llvm::BasicBlock* merge_block = NewBlock("BranchTagged Merge Block");
  check_blocks.push_back(merge_block);

  DCHECK(check_blocks.size() > 1);
  unsigned cur_block = 0;
  llvm::BasicBlock* next = check_blocks[cur_block];
  __ CreateBr(check_blocks[cur_block]);

  if (expected.Contains(ToBooleanStub::UNDEFINED)) {
    __ SetInsertPoint(check_blocks[cur_block]);
    // undefined -> false.
    auto is_undefined = CompareRoot(value, Heap::kUndefinedValueRootIndex);
    __ CreateCondBr(is_undefined, false_target, check_blocks[++cur_block]);
    next = check_blocks[cur_block];
  }

  if (expected.Contains(ToBooleanStub::BOOLEAN)) {
    __ SetInsertPoint(next);
    // true -> true.
    auto is_true = CompareRoot(value, Heap::kTrueValueRootIndex);
    llvm::BasicBlock* bool_second = NewBlock("BranchTagged Boolean Second Check");
    __ CreateCondBr(is_true, true_target, bool_second);
    // false -> false.
    __ SetInsertPoint(bool_second);
    auto is_false = CompareRoot(value, Heap::kFalseValueRootIndex);
    __ CreateCondBr(is_false, false_target, check_blocks[++cur_block]);
    next =  check_blocks[cur_block];
  }

  if (expected.Contains(ToBooleanStub::NULL_TYPE)) {
    __ SetInsertPoint(next);
    // 'null' -> false.
    auto is_null = CompareRoot(value, Heap::kNullValueRootIndex);
    __ CreateCondBr(is_null, false_target, check_blocks[++cur_block]);
    next = check_blocks[cur_block];
  }

  // TODO: Test (till the end) 3d-cube.js DrawQube
  if (expected.Contains(ToBooleanStub::SMI)) {
    __ SetInsertPoint(next);
    // Smis: 0 -> false, all other -> true.
    llvm::BasicBlock* not_zero = NewBlock("BranchTagged Smi Non Zero");
    auto cmp_zero = __ CreateICmpEQ(value_as_i64, __ getInt64(0));
    __ CreateCondBr(cmp_zero, false_target, not_zero);
    __ SetInsertPoint(not_zero);
    llvm::Value* smi_cond = SmiCheck(value, false);
    __ CreateCondBr(smi_cond, true_target, check_blocks[++cur_block]);
    next = check_blocks[cur_block];
  } else if (expected.NeedsMap()) {
    // If we need a map later and have a Smi -> deopt.
    //TODO: Not tested, string-fasta fastaRandom
    __ SetInsertPoint(next);
    auto smi_and = __ CreateAnd(value_as_i64, __ getInt64(kSmiTagMask));
    auto is_smi = __ CreateICmpEQ(smi_and, __ getInt64(0));
    DeoptimizeIf(is_smi, Deoptimizer::kSmi, false, check_blocks[++cur_block]);
    next = check_blocks[cur_block];
  }

  llvm::Value* map = nullptr;
  if (expected.NeedsMap()) {
    __ SetInsertPoint(next);
    map = LoadFieldOperand(value, HeapObject::kMapOffset);
    if (!expected.CanBeUndetectable()) {
      __ CreateBr(check_blocks[++cur_block]);
      next = check_blocks[cur_block];
    } else {
      auto map_bit_offset = LoadFieldOperand(map, Map::kBitFieldOffset);
      auto int_map_bit_offset = __ CreatePtrToInt(map_bit_offset, Types::i64);
      auto map_detach = __ getInt64(1 << Map::kIsUndetectable);
      auto test = __ CreateAnd(int_map_bit_offset, map_detach);
      auto cmp_zero = __ CreateICmpEQ(test, __ getInt64(0));
      __ CreateCondBr(cmp_zero, check_blocks[++cur_block], false_target);
      next = check_blocks[cur_block];
    }
  }

  if (expected.Contains(ToBooleanStub::SPEC_OBJECT)) {
    // spec object -> true.
    DCHECK(map); //FIXME: map can be null here
    __ SetInsertPoint(next);
    llvm::Value* cmp_instance = CmpInstanceType(map, FIRST_SPEC_OBJECT_TYPE, llvm::CmpInst::ICMP_UGE);
    __ CreateCondBr(cmp_instance, true_target, check_blocks[++cur_block]);
    next = check_blocks[cur_block];
  }

  if (expected.Contains(ToBooleanStub::STRING)) {
    // String value -> false iff empty.
    DCHECK(map); //FIXME: map can be null here
    __ SetInsertPoint(next);
    llvm::BasicBlock* is_string_bb = NewBlock("BranchTagged ToBoolString IsString");
    llvm::Value* cmp_instance = CmpInstanceType(map, FIRST_NONSTRING_TYPE, llvm::CmpInst::ICMP_UGE);
     __ CreateCondBr(cmp_instance, check_blocks[++cur_block], is_string_bb);
     __ SetInsertPoint(is_string_bb);
     next = check_blocks[cur_block];
     auto str_length = LoadFieldOperand(value, String::kLengthOffset);
     auto casted_str_length = __ CreatePtrToInt(str_length, Types::i64);
     auto cmp_length = __ CreateICmpEQ(casted_str_length, __ getInt64(0));
     __ CreateCondBr(cmp_length, false_target, true_target);
  }

  if (expected.Contains(ToBooleanStub::SIMD_VALUE)) {
    // SIMD value -> true.
    DCHECK(map); //FIXME: map can be null here
    __ SetInsertPoint(next);
    llvm::Value* cmp_simd = CmpInstanceType(map, SIMD128_VALUE_TYPE);
    __ CreateCondBr(cmp_simd, true_target, check_blocks[++cur_block]);
    next = check_blocks[cur_block];
  }

  if (expected.Contains(ToBooleanStub::SYMBOL)) {
    // Symbol value -> true.
    __ SetInsertPoint(next);
    DCHECK(map); //FIXME: map can be null here
    llvm::Value* cmp_instance = CmpInstanceType(map, SYMBOL_TYPE, llvm::CmpInst::ICMP_EQ);
    __ CreateCondBr(cmp_instance, true_target, check_blocks[++cur_block]);
    next = check_blocks[cur_block];
  }

  if (expected.Contains(ToBooleanStub::HEAP_NUMBER)) {
    // heap number -> false iff +0, -0, or NaN.
    DCHECK(map); //FIXME: map can be null here
    __ SetInsertPoint(next);
    llvm::BasicBlock* is_heap_bb = NewBlock("BranchTagged ToBoolString IsHeapNumber");
    auto cmp_root = CompareRoot(map, Heap::kHeapNumberMapRootIndex, llvm::CmpInst::ICMP_NE); 
    __ CreateCondBr(cmp_root, check_blocks[++cur_block], is_heap_bb);
    __ SetInsertPoint(is_heap_bb);
    next = check_blocks[cur_block];
    llvm::Value* zero_val = llvm::ConstantFP::get(Types::float64, 0);    
    auto value_addr = FieldOperand(value, HeapNumber::kValueOffset);
    llvm::Value* value_as_double_addr = __ CreateBitCast(value_addr,
                                                         Types::ptr_float64);
    auto load_val = __ CreateLoad(value_as_double_addr);

    llvm::Value* compare = __ CreateFCmpOEQ(load_val, zero_val);
    __ CreateCondBr(compare, false_target, true_target);  
  }

  __ SetInsertPoint(next) ; // TODO(llvm): not sure

  if (!expected.IsGeneric()) {
    // We've seen something for the first time -> deopt.
    // This can only happen if we are not generic already.
    auto no_condition = __ getTrue();
    DeoptimizeIf(no_condition, Deoptimizer::kUnexpectedObject);
    // Since we deoptimize on True the continue block is never reached.
    __ CreateUnreachable();
  } else {
     // TODO(llvm): not sure
     __ CreateUnreachable();
  }
}

llvm::Value* LLVMChunkBuilder::CmpInstanceType(llvm::Value* map,
                             InstanceType type,
                             llvm::CmpInst::Predicate predicate) {
   llvm::Value* field_operand = LoadFieldOperand(map, Map::kInstanceTypeOffset);
   llvm::Value* int_field_operand = __ CreatePtrToInt(field_operand, Types::i64);
   llvm::Value* int_type = __ getInt64(static_cast<int>(type));
   llvm::Value* cmp_result = __ CreateICmp(predicate, int_field_operand,
                                           int_type, "CmpInstanceType");
   return cmp_result;
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
  } else if (r.IsSmi()) {
    UNIMPLEMENTED();
  } else if (r.IsDouble()) {
    llvm::Value* zero = llvm::ConstantFP::get(Types::float64, 0);
    llvm::Value* compare = __ CreateFCmpUNE(Use(value), zero);
    llvm::BranchInst* branch = __ CreateCondBr(compare,
                                               true_target, false_target);
    instr->set_llvm_value(branch);
  } else {
    DCHECK(r.IsTagged());
    llvm::Value* value = Use(instr->value());
    if (type.IsBoolean()) {
      DCHECK(!info()->IsStub());
      llvm::Value* cmp_root = CompareRoot(value, Heap::kTrueValueRootIndex);
      llvm::BranchInst* branch =  __ CreateCondBr(cmp_root, true_target, false_target);
      instr->set_llvm_value(branch);
    } else if (type.IsString()) {
      DCHECK(!info()->IsStub());
      llvm::Value* zero = __ getInt64(0);
      llvm::Value* length = LoadFieldOperand(value, String::kLengthOffset);
      llvm::Value* casted_length = __ CreatePtrToInt(length, Types::i64);
      llvm::Value* compare = __ CreateICmpNE(casted_length, zero);
      llvm::BranchInst* branch = __ CreateCondBr(compare, true_target,
                                                 false_target);
      instr->set_llvm_value(branch);
    } else if (type.IsSmi() || type.IsJSArray() || type.IsHeapNumber()) {
      UNIMPLEMENTED();
    } else {
      ToBooleanStub::Types expected = instr->expected_input_types();
      BranchTagged(instr, expected, true_target, false_target);
    }
  }
}

llvm::CallingConv::ID LLVMChunkBuilder::GetCallingConv(CallInterfaceDescriptor descriptor) {
  if (descriptor.GetRegisterParameterCount() == 4) {
    if (descriptor.GetRegisterParameter(0).is(rdi) &&
        descriptor.GetRegisterParameter(1).is(rbx) &&
        descriptor.GetRegisterParameter(2).is(rcx) &&
        descriptor.GetRegisterParameter(3).is(rdx))
      return llvm::CallingConv::X86_64_V8_S1;
   return -1;
  }
  if (descriptor.GetRegisterParameterCount() == 3) {
    //FIXME: // Change CallingConv
    if (descriptor.GetRegisterParameter(0).is(rdi) &&
        descriptor.GetRegisterParameter(1).is(rax) &&
        descriptor.GetRegisterParameter(2).is(rbx))
      return llvm::CallingConv::X86_64_V8_S3;
  }
  if (descriptor.GetRegisterParameterCount() == 1) {
    if (descriptor.GetRegisterParameter(0).is(rax))
      return llvm::CallingConv::X86_64_V8_S11;
    if (descriptor.GetRegisterParameter(0).is(rbx))
      return llvm::CallingConv::X86_64_V8_S8;
    return -1;
  }
  return -1;
}

void LLVMChunkBuilder::DoCallWithDescriptor(HCallWithDescriptor* instr) {
  CallInterfaceDescriptor descriptor = instr->descriptor();
  llvm::CallingConv::ID conv = GetCallingConv(descriptor);
  // FIXME(llvm): not very good because CallingConv::ID is unsigned.
  if (conv == -1) UNIMPLEMENTED();

  //TODO: Do wee need this check here?
  if (descriptor.GetRegisterParameterCount() != instr->OperandCount() - 2) UNIMPLEMENTED();
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
    // TODO(llvm)::
    // LPointerMap* pointers = instr->pointer_map();
    // SafepointGenerator generator(this, pointers, Safepoint::kLazyDeopt);

    if (target->IsConstant()) {
      Handle<Object> handle = HConstant::cast(target)->handle(isolate());
      Handle<Code> code = Handle<Code>::cast(handle);
      // TODO(llvm, gc): reloc info mode of the code (CODE_TARGET)...
      llvm::Value* call = CallCode(code, conv, params);
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

  // Code that follows relies on this assumption
  // (well, maybe it's not, we haven't seen a test case yet)
  if (!instr->function()->IsConstant()) UNIMPLEMENTED();
  // TODO(llvm): self call

  // TODO(llvm): record safepoints...
  auto function_object = Use(instr->function()); // It's an int constant (a ptr)
  auto target_entry = LoadFieldOperand(function_object,
                                       JSFunction::kCodeEntryOffset,
                                       "target_entry");
  auto target_context = LoadFieldOperand(function_object,
                                         JSFunction::kContextOffset,
                                         "target_context");

  int actual_arg_count = 4; //rsi, rdi, rbx (OSR), rax
  auto argument_count = instr->argument_count() + actual_arg_count;
  //TODO:// get rid of this
  // Set up the actual arguments
  std::vector<llvm::Value*> args(argument_count, nullptr);
  args[0] = target_context;
  args[1] = function_object;
  args[2] = __ getInt64(0);
  //FIXME: This case needs farther investigation. Do we need new Calling Convention here?
  // crypto-aes AESDecryptCtr fails without this
  args[3] = __ getInt64(instr->argument_count()-1);
  DCHECK(pending_pushed_args_.length() + actual_arg_count == argument_count);
  // The order is reverse because X86_64_V8 is not implemented quite right.
  for (int i = 0; i < pending_pushed_args_.length(); i++) {
    args[argument_count - 1 - i] = pending_pushed_args_[i];
  }
  pending_pushed_args_.Clear();

  bool record_safepoint = true;
  auto call = CallVal(target_entry, llvm::CallingConv::X86_64_V8_E, args,
                      Types::tagged, record_safepoint);
  instr->set_llvm_value(call);
}

void LLVMChunkBuilder::DoCallFunction(HCallFunction* instr) {
  int arity = instr->argument_count() - 1;
  CallFunctionFlags flags = instr->function_flags();
  llvm::Value* context = Use(instr->context());
  llvm::Value* function = Use(instr->function());
  llvm::Value* result = nullptr;

  if (instr->HasVectorAndSlot()) {
    //UNIMPLEMENTED();
    AllowDeferredHandleDereference vector_structure_check;
    AllowHandleAllocation allow_handles;
    Handle<TypeFeedbackVector> feedback_vector = instr->feedback_vector();
    int index = feedback_vector->GetIndex(instr->slot());

    CallICState::CallType call_type =
        (flags & CALL_AS_METHOD) ? CallICState::METHOD : CallICState::FUNCTION;

    Handle<Code> ic =
        CodeFactory::CallICInOptimizedCode(isolate(), arity, call_type).code();
    llvm::Value* vector = MoveHeapObject(feedback_vector);
    std::vector<llvm::Value*> params;
    params.push_back(context);
    params.push_back(function);
    params.push_back(vector);
    Smi* smi_index = Smi::FromInt(index);
    params.push_back(ValueFromSmi(smi_index));
    for (int i = pending_pushed_args_.length()-1; i >=0; --i)
      params.push_back(pending_pushed_args_[i]);
    pending_pushed_args_.Clear();
    result = CallCode(ic, llvm::CallingConv::X86_64_V8_S6,
                             params);
  } else {
    CallFunctionStub stub(isolate(), arity, flags);
    AllowHandleAllocation allow_handles;
    AllowHeapAllocation allow_heap;
    std::vector<llvm::Value*> params;
    params.push_back(context);
    params.push_back(function);
    for (int i = pending_pushed_args_.length()-1; i >=0; --i)
      params.push_back(pending_pushed_args_[i]);
    pending_pushed_args_.Clear();
    result = CallCode(stub.GetCode(), llvm::CallingConv::X86_64_V8_S13, params);
  }
  instr->set_llvm_value(result);
}

void LLVMChunkBuilder::DoCallNew(HCallNew* instr) {
  // TODO: not tested
  // FIXME: don't we need pending_push_args ?
  int arity = instr->argument_count()-1;
  llvm::Value* arity_val = __ getInt64(arity);
  llvm::Value* load_r = LoadRoot(Heap::kUndefinedValueRootIndex);
  CallConstructStub stub(isolate(), NO_CALL_CONSTRUCTOR_FLAGS);
  Handle<Code> code = Handle<Code>::null();
  {
    AllowHandleAllocation allow_handles;
    AllowHeapAllocation allow_heap;
    code = stub.GetCode();
  }
  std::vector<llvm::Value*> params;
  for (int i = 0; i < instr->OperandCount(); ++i)
    params.push_back(Use(instr->OperandAt(i)));
  params.push_back(arity_val);
  params.push_back(load_r);
  for (int i = pending_pushed_args_.length()-1; i >=0; --i)
      params.push_back(pending_pushed_args_[i]);
  pending_pushed_args_.Clear();
  llvm::Value* call = CallCode(code, llvm::CallingConv::X86_64_V8_S3, params);
  instr->set_llvm_value(call);
}

void LLVMChunkBuilder::DoCallNewArray(HCallNewArray* instr) {
  //TODO: Respect RelocInfo
  int arity = instr->argument_count() - 1;
  llvm::Value* arity_val = __ getInt64(arity);
  llvm::Value* result_packed_elem = nullptr;
  llvm::BasicBlock* packed_continue = nullptr;
  llvm::Value* load_root = LoadRoot(Heap::kUndefinedValueRootIndex);
  ElementsKind kind = instr->elements_kind();
  AllocationSiteOverrideMode override_mode =
      (AllocationSite::GetMode(kind) == TRACK_ALLOCATION_SITE)
          ? DISABLE_ALLOCATION_SITES
          : DONT_OVERRIDE;
  if (arity == 0) {
    UNIMPLEMENTED();
  } else if (arity == 1) {
    llvm::BasicBlock* done = nullptr;  
    llvm::BasicBlock* packed_case = NewBlock("CALL NEW ARRAY PACKED CASE");
    load_root = MoveHeapObject(instr->site());
    if (IsFastPackedElementsKind(kind)) {
      packed_continue = NewBlock("CALL NEW ARRAY PACKED CASE CONTINUE");
      DCHECK_GE(pending_pushed_args_.length(), 1);
      llvm::Value* first_arg = pending_pushed_args_[0];
      first_arg = __ CreateBitOrPointerCast(first_arg, Types::i64);
      llvm::Value* cmp_eq = __ CreateICmpEQ(first_arg, __ getInt64(0));
      __ CreateCondBr(cmp_eq, packed_case, packed_continue);
      __ SetInsertPoint(packed_continue);
      ElementsKind holey_kind = GetHoleyElementsKind(kind);
      ArraySingleArgumentConstructorStub stub(isolate(),
                                              holey_kind,
                                              override_mode);
      Handle<Code> code = Handle<Code>::null();
      {
        AllowHandleAllocation allow_handles;
        AllowHeapAllocation allow_heap;
        code = stub.GetCode();
        // FIXME(llvm,gc): respect reloc info mode...
      }
      std::vector<llvm::Value*> params;
      params.push_back(Use(instr->context()));
      for (int i = 1; i < instr->OperandCount(); ++i)
        params.push_back(Use(instr->OperandAt(i)));
      params.push_back(arity_val);
      params.push_back(load_root);
      for (int i = pending_pushed_args_.length() - 1; i >=0; --i)
        params.push_back(pending_pushed_args_[i]);
      pending_pushed_args_.Clear();
      result_packed_elem = CallCode(code, llvm::CallingConv::X86_64_V8_S3,
                                    params);
      done =  NewBlock("CALL NEW ARRAY END");
      __ CreateBr(done);
    }
    else {
      done = NewBlock("CALL NEW ARRAY END");
      __ CreateBr(packed_case);
    }
    //__ CreateBr(packed_case);
    __ SetInsertPoint(packed_case);
    ArraySingleArgumentConstructorStub stub(isolate(), kind, override_mode);
    Handle<Code> code = Handle<Code>::null();
    {
      AllowHandleAllocation allow_handles;
      AllowHeapAllocation allow_heap;
      code = stub.GetCode();
      // FIXME(llvm,gc): respect reloc info mode...
    }
    std::vector<llvm::Value*> params;
    params.push_back(Use(instr->context())); 
    for (int i = 1; i < instr->OperandCount(); ++i)
      params.push_back(Use(instr->OperandAt(i)));
    params.push_back(arity_val);
    params.push_back(load_root);
    for (int i = pending_pushed_args_.length()-1; i >=0; --i)
      params.push_back(pending_pushed_args_[i]);
    pending_pushed_args_.Clear();
    llvm::Value* return_val = CallCode(code, llvm::CallingConv::X86_64_V8_S3,
                                       params);
    __ CreateBr(done);
    __ SetInsertPoint(done);
    llvm::PHINode* phi = __ CreatePHI(Types::tagged, result_packed_elem ? 2 : 1);
    phi->addIncoming(return_val, packed_case);
    if (result_packed_elem) {
      DCHECK(packed_continue);
      phi->addIncoming(result_packed_elem, packed_continue);
    }
    instr->set_llvm_value(phi);
  } else {
    std::vector<llvm::Value*> params;
    params.push_back(Use(instr->context()));
    params.push_back(Use(instr->constructor()));
    params.push_back(arity_val);
    params.push_back(load_root);
    for (int i = pending_pushed_args_.length()-1; i >=0; --i)
       params.push_back(pending_pushed_args_[i]);
    pending_pushed_args_.Clear();
 
    ArrayNArgumentsConstructorStub stub(isolate(), kind, override_mode);
    Handle<Code> code = Handle<Code>::null();
    {
      AllowHandleAllocation allow_handles;
      AllowHeapAllocation allow_heap;
      code = stub.GetCode();
    }
    //DCHECK(code)
    auto result = CallCode(code, llvm::CallingConv::X86_64_V8_S3, params);
    instr->set_llvm_value(result);
  }
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
  llvm::Value* context = Use(instr->context());
  std::vector<llvm::Value*> params;
  params.push_back(context);
  for (int i = pending_pushed_args_.length()-1; i >=0; --i)
     params.push_back(pending_pushed_args_[i]);
  pending_pushed_args_.Clear();
  switch (instr->major_key()) {
    case CodeStub::RegExpExec: {
      RegExpExecStub stub(isolate());
      AllowHandleAllocation allow_handle;
      llvm::Value* call = CallCode(stub.GetCode(), 
                                   llvm::CallingConv::X86_64_V8_Stub,
                                   params);
      llvm::Value* result =  __ CreatePtrToInt(call, Types::i64);
      instr->set_llvm_value(result);
      break;
    }
    case CodeStub::SubString: {
      SubStringStub stub(isolate());
      AllowHandleAllocation allow_handle;
      llvm::Value* call = CallCode(stub.GetCode(),
                                   llvm::CallingConv::X86_64_V8_Stub,
                                   params);
      instr->set_llvm_value(call);
      break;
    }
    default:
      UNREACHABLE();
  }
}

void LLVMChunkBuilder::DoCapturedObject(HCapturedObject* instr) {
  instr->ReplayEnvironment(current_block_->last_environment());
  // There are no real uses of a captured object.
}

void LLVMChunkBuilder::ChangeDoubleToTagged(HValue* val, HChange* instr) {
  // TODO(llvm): this case in Crankshaft utilizes deferred calling.
  llvm::Value* new_heap_number = nullptr;
  DCHECK(Use(val)->getType()->isDoubleTy());
  if (FLAG_inline_new) {
    new_heap_number = AllocateHeapNumber();
  } else {
    auto slow_heap = AllocateHeapNumberSlow(); // i8*
    new_heap_number = __ CreateBitOrPointerCast(slow_heap, Types::i64);
  }

  llvm::Value* store_address = FieldOperand(new_heap_number,
                                            HeapNumber::kValueOffset);
  llvm::Value* casted_address = __ CreateBitCast(store_address,
                                                 Types::ptr_i64);
  llvm::Value* casted_intermediate = __ CreateBitCast(Use(val), Types::i64);
  llvm::Value* casted_val = __ CreateBitOrPointerCast(casted_intermediate,
                                                      Types::i64);
  // [(i8*)new_heap_number + offset] = val;
  __ CreateStore(casted_val, casted_address);
  auto result = __ CreateIntToPtr(new_heap_number, Types::tagged);
  instr->set_llvm_value(result); // no offset

  //  TODO(llvm): AssignPointerMap(Define(result, result_temp));
}

llvm::Value* LLVMChunkBuilder::LoadRoot(Heap::RootListIndex index) {
  Address root_array_start_address =
      ExternalReference::roots_array_start(isolate()).address();
  // TODO(llvm): Move(RelocInfo::EXTERNAL_REFERENCE)
  auto int64_address =
      __ getInt64(reinterpret_cast<uint64_t>(root_array_start_address));
  auto address = __ CreateBitOrPointerCast(int64_address, Types::ptr_tagged);
  int offset = index << kPointerSizeLog2;
  auto load_address = ConstructAddress(address, offset);
  return __ CreateLoad(load_address);
}

llvm::Value* LLVMChunkBuilder::CompareRoot(llvm::Value* operand,
                                           Heap::RootListIndex index,
                                           llvm::CmpInst::Predicate predicate) {
  llvm::Value* root_value_by_index = LoadRoot(index);
  llvm::Value* cmp_result = __ CreateICmp(predicate, operand,
                                          root_value_by_index, "CompareRoot");
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
  llvm::BasicBlock* merge_block = NewBlock(
      std::string("ChangeTaggedToDouble Merge ") + std::to_string(instr->id()));

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
      DeoptimizeIf(is_undefined, Deoptimizer::kNotAHeapNumberUndefined,
                   deopt_on_not_undefined, conversion_end);
      nan_value = GetNan();
      __ CreateBr(merge_block);
    } else {
      bool deopt_on_not_equal = true;
      DeoptimizeIf(is_heap_number, Deoptimizer::kNotAHeapNumber,
                   deopt_on_not_equal, merge_block);
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
  llvm::BasicBlock* not_smi = NewBlock("is smi 'deferred' case");
  llvm::BasicBlock* zero_block = nullptr;
  llvm::BasicBlock* merge_and_ret = NewBlock(
      std::string("merge and ret ") + std::to_string(instr->id()));
  llvm::BasicBlock* not_smi_merge = nullptr;

  __ CreateCondBr(cond, is_smi, not_smi);

  __ SetInsertPoint(is_smi);
  llvm::Value* relult_for_smi = SmiToInteger32(val);
  __ CreateBr(merge_and_ret);

  __ SetInsertPoint(not_smi);
  llvm::Value* relult_for_not_smi = nullptr;
  llvm::Value* minus_zero_result = nullptr;
  bool truncating = instr->CanTruncateToInt32();

  llvm::Value* vals_map = LoadFieldOperand(Use(val), HeapObject::kMapOffset);
  llvm::Value* cmp = CompareRoot(vals_map, Heap::kHeapNumberMapRootIndex,
                                 llvm::CmpInst::ICMP_NE);

  if (truncating) {
    llvm::BasicBlock* truncate_heap_number = NewBlock("TruncateHeapNumberToI");
    llvm::BasicBlock* no_heap_number = NewBlock("Not a heap number");
    llvm::BasicBlock* merge_inner = NewBlock("inner merge");

    __ CreateCondBr(cmp, no_heap_number, truncate_heap_number);

    __ SetInsertPoint(truncate_heap_number);
    llvm::Value* value_addr = FieldOperand(Use(val), HeapNumber::kValueOffset);
    // cast to ptr to double, fetch the double and convert to i32
    llvm::Value* double_addr = __ CreateBitCast(value_addr, Types::ptr_float64);
    llvm::Value* double_val = __ CreateLoad(double_addr);
    llvm::Value* truncate_heap_number_result = __ CreateFPToSI(double_val,
                                                               Types::i32);

    //TruncateHeapNumberToI
    llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(
          module_.get(), llvm::Intrinsic::ssub_with_overflow, Types::i32);
    llvm::Value* params[] = { __ getInt32(1), truncate_heap_number_result };
    llvm::Value* call = __ CreateCall(intrinsic, params);
    llvm::Value* overflow = __ CreateExtractValue(call, 1);
    llvm::BasicBlock* slow_case = NewBlock("ChangeTaggedToISlow slow case");
    llvm::BasicBlock* done = NewBlock("ChangeTaggedToISlow done");
    __ CreateCondBr(overflow, slow_case, done);

    __ SetInsertPoint(slow_case);
    Register input_reg = rbx;
    Register result_reg = rax;
    int offset = HeapNumber::kValueOffset - kHeapObjectTag;
    DoubleToIStub stub(isolate(), input_reg, result_reg, offset, true);
    Handle<Code> code = Handle<Code>::null();
    {
      AllowHandleAllocation allow_handles;
      AllowHeapAllocation allow_heap_alloc;
      code = stub.GetCode();
    }
    Assert(__ getFalse()); // FIXME(llvm): Not tested this case
    llvm::Value* fictive_val = __ getInt32(0); //fictive
    std::vector<llvm::Value*> args = {fictive_val, truncate_heap_number_result};
    llvm::Value* result_intrisic = CallCode(code,
                                            llvm::CallingConv::X86_64_V8_S12,
                                            args);
    llvm::Value* casted_result_intrisic = __ CreatePtrToInt(result_intrisic,
                                                            Types::i32);
    __ CreateBr(done);

    __ SetInsertPoint(done);
    llvm::PHINode* phi_done = __ CreatePHI(Types::i32, 2);
    phi_done->addIncoming(casted_result_intrisic, slow_case);
    phi_done->addIncoming(truncate_heap_number_result, truncate_heap_number);
    __ CreateBr(merge_inner);

    __ SetInsertPoint(no_heap_number);
    // Check for Oddballs. Undefined/False is converted to zero and True to one
    // for truncating conversions.
    llvm::BasicBlock* check_bools = NewBlock("ChangeTaggedToISlow check_bool");
    llvm::BasicBlock* no_check_bools = NewBlock("ChangeTaggedToISlow"
                                                " no_check_bools");
    llvm::Value* cmp_undefined = CompareRoot(Use(val),
                                             Heap::kUndefinedValueRootIndex,
                                             llvm::CmpInst::ICMP_NE);
    __ CreateCondBr(cmp_undefined, check_bools, no_check_bools);
    __ SetInsertPoint(no_check_bools);
    llvm::Value* result_no_check_bools = __ getInt32(0);
    __ CreateBr(merge_inner);

    __ SetInsertPoint(check_bools);
    llvm::BasicBlock* check_true = NewBlock("ChangeTaggedToISlow check_true");
    llvm::BasicBlock* check_false = NewBlock("ChangeTaggedToISlow check_false");
    llvm::Value* cmp_true = CompareRoot(Use(val), Heap::kTrueValueRootIndex,
                                   llvm::CmpInst::ICMP_NE);
    __ CreateCondBr(cmp_true, check_false, check_true);

    __ SetInsertPoint(check_true);
    llvm::Value* result_check_true = __ getInt32(1);
    __ CreateBr(merge_inner);

    __ SetInsertPoint(check_false);
    llvm::Value* cmp_false = CompareRoot(Use(val), Heap::kFalseValueRootIndex,
                                         llvm::CmpInst::ICMP_NE);
    DeoptimizeIf(cmp_false, Deoptimizer::kNotAHeapNumberUndefinedBoolean);
    llvm::Value* result_check_false = __ getInt32(0);
    __ CreateBr(merge_inner);

    __ SetInsertPoint(merge_inner);
    llvm::PHINode* phi_inner = __ CreatePHI(Types::i32, 4);
    phi_inner->addIncoming(result_no_check_bools, no_check_bools);
    phi_inner->addIncoming(result_check_true, check_true);
    phi_inner->addIncoming(result_check_false, check_false);
    phi_inner->addIncoming(phi_done, done);
    relult_for_not_smi = phi_inner;
    not_smi_merge = merge_inner;
  } else {
    // The comparison is already done with NE, so no need for negation here.
    DeoptimizeIf(cmp, Deoptimizer::kNotAHeapNumber);

    auto address = FieldOperand(Use(val), HeapNumber::kValueOffset);
    auto double_addr = __ CreateBitCast(address, Types::ptr_float64);
    auto double_val = __ CreateLoad(double_addr);
    // Convert the double to int32; convert it back do double and
    // see it the 2 doubles are equal and neither is a NaN.
    // If not, deopt (kLostPrecision or kNaN)
    relult_for_not_smi = __ CreateFPToSI(double_val, Types::i32);
    auto converted_double = __ CreateSIToFP(relult_for_not_smi, Types::float64);
    auto ordered_and_equal = __ CreateFCmpOEQ(double_val, converted_double);
    bool negate = true;
    // TODO(llvm): in case they are unordered or equal, reason should be
    // kLostPrecision.
    DeoptimizeIf(ordered_and_equal, Deoptimizer::kNaN, negate);
    if (instr->GetMinusZeroMode() == FAIL_ON_MINUS_ZERO) {
      zero_block = NewBlock("TaggedToISlow ZERO");
      auto equal_ = __ CreateICmpEQ(relult_for_not_smi, __ getInt32(0));
      __ CreateCondBr(equal_, zero_block, merge_and_ret);
      __ SetInsertPoint(zero_block);
      InsertDebugTrap();
      minus_zero_result = __ getInt32(0);
      // __ CreateBr(not_smi_merge);
      /*llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
             llvm::Intrinsic::x86_sse2_movmsk_pd);
      llvm::Value* param_vect = __ CreateVectorSplat(2, input_val);
      __ CreateInsertElement(param_vect, double_val, __ getInt32(0));
      llvm::Value* call = __ CreateCall(intrinsic, param_vect);
      __ CreateAnd(call, __ getInt32(1)); //FIXME(llvm)://Possibly wrong
      auto is_zero = __ CreateICmpEQ(call, __ getInt64(0));
      DeoptimizeIf(is_zero, false, merge_and_ret); */
    }
    not_smi_merge =  __ GetInsertBlock();
  }
  __ CreateBr(merge_and_ret);

  __ SetInsertPoint(merge_and_ret);
  llvm::PHINode* phi = nullptr;
  if (instr->GetMinusZeroMode() == FAIL_ON_MINUS_ZERO) {
    phi = __ CreatePHI(Types::i32, 3);
    phi->addIncoming(minus_zero_result, zero_block);
    phi->addIncoming(relult_for_smi, is_smi);
    phi->addIncoming(relult_for_not_smi, not_smi_merge);
  }
  else { 
    phi = __ CreatePHI(Types::i32, 2);
    phi->addIncoming(relult_for_smi, is_smi);
    phi->addIncoming(relult_for_not_smi, not_smi_merge);
  }
  instr->set_llvm_value(phi);
}

void LLVMChunkBuilder::DoChange(HChange* instr) {
  Representation from = instr->from();
  Representation to = instr->to();
  HValue* val = instr->value();
  if (from.IsSmi()) {
    if (to.IsTagged()) {
      auto as_tagged = __ CreateBitOrPointerCast(Use(val), Types::tagged);
      instr->set_llvm_value(as_tagged);
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
        DeoptimizeIf(cond, Deoptimizer::kNotASmi);
      }
      auto val_as_smi = __ CreateBitOrPointerCast(Use(val), Types::smi);
      instr->set_llvm_value(val_as_smi);
    } else {
      DCHECK(to.IsInteger32());
      if (val->type().IsSmi() || val->representation().IsSmi()) {
        // convert smi to int32, no need to perform smi check
        // lithium codegen does __ AssertSmi(input)
        instr->set_llvm_value(SmiToInteger32(val));
      } else {
        ChangeTaggedToISlow(val, instr);
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
        auto smi_as_tagged = __ CreateBitOrPointerCast(Integer32ToSmi(val),
                                                       Types::tagged);
        instr->set_llvm_value(smi_as_tagged);
      } else if (instr->value()->CheckFlag(HInstruction::kUint32)) {
        UIntToTag(instr);
      } else {
        UNIMPLEMENTED();
      }
    } else if (to.IsSmi()) {
      //TODO: not tested
      if (instr->CheckFlag(HValue::kCanOverflow) &&
          instr->value()->CheckFlag(HValue::kUint32)) {
        llvm::Value* cmp = nullptr;
        if (SmiValuesAre32Bits()) {
          //This will check if the high bit is set
          //If it set we can't convert int to smi
          cmp = __ CreateICmpSLT(Use(val), __ getInt32(0));
         } else {
            UNIMPLEMENTED();
            DCHECK(SmiValuesAre31Bits());
         }
         DeoptimizeIf(cmp, Deoptimizer::kOverflow);
         // UNIMPLEMENTED();
      }
      llvm::Value* result = Integer32ToSmi(val);
      instr->set_llvm_value(result);
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
void LLVMChunkBuilder::UIntToTag(HChange* instr ){

  llvm::Value* val = Use(instr->value());
  llvm::Value* cmp = __ CreateICmpUGT(val, __ getInt32(Smi::kMaxValue));
  llvm::BasicBlock* def_entr = NewBlock("IntToTag Deferred entry");
  llvm::BasicBlock* cont = NewBlock("IntToTag Continue");
  llvm::BasicBlock* done = NewBlock("IntToTag Done");
  __ CreateCondBr(cmp, def_entr, cont);

  __ SetInsertPoint(cont);
  
  llvm::Value* result = Integer32ToSmi(val);
  __ CreateBr(done);

  __ SetInsertPoint(def_entr);
  
  if (FLAG_debug_code) {
    UNIMPLEMENTED();
  }

  // Trap, wrong implementation.
  InsertDebugTrap();
  auto new_heap_number_casted = __ getInt64(0);
  /*llvm::Value* double_value = __ CreateSIToFP(val, Types::float64);
  
  if (FLAG_inline_new) {
   UNIMPLEMENTED();
  }
  
  llvm::Value* new_heap_number = AllocateHeapNumber();
  
  llvm::Value* store_address = FieldOperand(new_heap_number,
                                            HeapNumber::kValueOffset);
  llvm::Value* casted_adderss = __ CreateBitCast(store_address,
                                                 Types::ptr_tagged);
  llvm::Value* casted_val = __ CreateBitCast(double_value, Types::tagged);
  __ CreateStore(casted_val, casted_adderss);
  llvm::Value* new_heap_number_casted = __ CreatePtrToInt(new_heap_number,
                                                          Types::tagged); */
  __ CreateBr(done);

  __ SetInsertPoint(done);
  llvm::PHINode* phi = __ CreatePHI(Types::i64, 2);
  phi->addIncoming(result, cont);
  phi->addIncoming(new_heap_number_casted, def_entr);
  instr->set_llvm_value(phi);
}

void LLVMChunkBuilder::DoCheckHeapObject(HCheckHeapObject* instr) {
  if (!instr->value()->type().IsHeapObject()) {
    llvm::Value* is_smi = SmiCheck(Use(instr->value()));
    DeoptimizeIf(is_smi, Deoptimizer::kSmi);
  }
}

void LLVMChunkBuilder::DoCheckInstanceType(HCheckInstanceType* instr) {
  llvm::Value* value = LoadFieldOperand(Use(instr->value()),
                                        HeapObject::kMapOffset);
  if (instr->is_interval_check()) {
    InstanceType first;
    InstanceType last;
    instr->GetCheckInterval(&first, &last);
    
    llvm::Value* instance = LoadFieldOperand(value, Map::kInstanceTypeOffset);
    llvm::Value* imm_first = __ getInt64(static_cast<int>(first));

    // If there is only one type in the interval check for equality.
    if (first == last) {
      llvm::Value* cmp = __ CreateICmpNE(instance, imm_first);
      DeoptimizeIf(cmp, Deoptimizer::kWrongInstanceType);
    } else {
      llvm::Value* cmp = __ CreateICmpULT(instance, imm_first);
      DeoptimizeIf(cmp, Deoptimizer::kWrongInstanceType);
      // Omit check for the last type.
      if (last != LAST_TYPE) {
        llvm::Value* imm_last = __ getInt64(static_cast<int>(last));
        llvm::Value* cmp = __ CreateICmpUGT(instance, imm_last);
        DeoptimizeIf(cmp, Deoptimizer::kWrongInstanceType);
      }
    }
  } else {
    uint8_t mask;
    uint8_t tag;
    instr->GetCheckMaskAndTag(&mask, &tag);
    
    if (base::bits::IsPowerOfTwo32(mask)) {
      llvm::Value* addr = FieldOperand(value , Map::kInstanceTypeOffset);
      llvm::Value* cast_to_int = __ CreateBitCast(addr, Types::ptr_i64);
      llvm::Value* val = __ CreateLoad(cast_to_int);
      llvm::Value* test = __ CreateAnd(val, __ getInt64(mask));
      llvm::Value* cmp = nullptr;
      if (tag == 0) {
        cmp = __ CreateICmpNE(test, __ getInt64(0));
      } else {
        cmp = __ CreateICmpEQ(test, __ getInt64(0));
      }
      DeoptimizeIf(cmp, Deoptimizer::kWrongInstanceType);
    } else {
      //TODO: not tested (fail form string-tagcloud.js in function ""
      //                  fail form date-format-tofte.js in arrayExists)
      llvm::Value* instance_offset = LoadFieldOperand(value,
                                                      Map::kInstanceTypeOffset);

      llvm::Value* casted_offset = __ CreatePtrToInt(instance_offset, Types::i64);
      llvm::Value* and_value = __ CreateAnd(casted_offset, __ getInt64(0x000000ff));
      llvm::Value* and_mask = __ CreateAnd(and_value, __ getInt64(mask));
      llvm::Value* cmp = __ CreateICmpEQ(and_mask, __ getInt64(tag));
      DeoptimizeIf(cmp, Deoptimizer::kWrongInstanceType, true);
    }
  }
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

void LLVMChunkBuilder::AddDeprecationDependency(Handle<Map> map) {
  if (map->is_deprecated()) return Retry(kMapBecameDeprecated);
  chunk()->AddDeprecationDependency(map);
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
    DeoptimizeIf(compare_result, Deoptimizer::kInstanceMigrationFailed,
                 deopt_on_equal, success);
    pending_pushed_args_.Clear();
    // Don't let the success BB go stray (__ SetInsertPoint).
    
  } else {
    bool deopt_on_not_equal = true;
    DeoptimizeIf(compare, Deoptimizer::kWrongMap, deopt_on_not_equal, success);
  }
}

void LLVMChunkBuilder::DoCheckMapValue(HCheckMapValue* instr) {
  llvm::Value* val = Use(instr->value());
  llvm::Value* tagged_val = FieldOperand(val, HeapObject::kMapOffset);
  llvm::Value* cmp = __ CreateICmpNE(Use(instr->map()), tagged_val);
  DeoptimizeIf(cmp, Deoptimizer::kWrongMap, true);
}

void LLVMChunkBuilder::DoCheckSmi(HCheckSmi* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCheckValue(HCheckValue* instr) {
  llvm::Value* reg = Use(instr->value());
  Handle<Object> source = instr->object().handle();
  llvm::Value* cmp = nullptr;
  if (source->IsSmi()) {
    Smi* smi = Smi::cast(*source);
    intptr_t intptr_value = reinterpret_cast<intptr_t>(smi);
    llvm::Value* value = __ getInt64(intptr_value);
    cmp = __ CreateICmpNE(reg, value);
  } else {
    auto obj = MoveHeapObject(instr->object().handle());
    cmp = __ CreateICmpNE(reg, obj);
  }
  DeoptimizeIf(cmp, Deoptimizer::kValueMismatch);
}

void LLVMChunkBuilder::DoClampToUint8(HClampToUint8* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoClassOfTestAndBranch(HClassOfTestAndBranch* instr) {
  UNIMPLEMENTED();
  // search test what it use this case
  // because I think what loop is not correctly
  llvm::Value* input = Use(instr->value());
  llvm::Value* temp = nullptr; 
  llvm::Value* temp2 = nullptr;
  Handle<String> class_name = instr->class_name();
  llvm::BasicBlock* input_not_smi = NewBlock("DoClassOfTestAndBranch"
                                             "input NotSmi");
  llvm::BasicBlock* continue_ = NewBlock("DoClassOfTestAndBranch Continue");
  llvm::BasicBlock* insert = __ GetInsertBlock();
  llvm::Value* smi_cond = SmiCheck(input);
  __ CreateCondBr(smi_cond, Use(instr->SuccessorAt(1)), input_not_smi);
  __ SetInsertPoint(input_not_smi);

  if (String::Equals(isolate()->factory()->Function_string(), class_name)) {
    STATIC_ASSERT(NUM_OF_CALLABLE_SPEC_OBJECT_TYPES == 2);
    STATIC_ASSERT(FIRST_NONCALLABLE_SPEC_OBJECT_TYPE ==
                  FIRST_SPEC_OBJECT_TYPE + 1);
    STATIC_ASSERT(LAST_NONCALLABLE_SPEC_OBJECT_TYPE ==
                  LAST_SPEC_OBJECT_TYPE - 1);
    STATIC_ASSERT(LAST_SPEC_OBJECT_TYPE == LAST_TYPE);
    UNIMPLEMENTED();
  } else {
    temp = LoadFieldOperand(input, HeapObject::kMapOffset);
    llvm::Value* instance_type = LoadFieldOperand(temp,
                                                  Map::kInstanceTypeOffset);
    temp2 = __ CreateAnd(instance_type, __ getInt64(0x000000ff));
    llvm::Value* obj_type = __ getInt64(FIRST_NONCALLABLE_SPEC_OBJECT_TYPE);
    llvm::Value* sub = __ CreateSub(temp2, obj_type);
    llvm::Value* imm = __ getInt64(LAST_NONCALLABLE_SPEC_OBJECT_TYPE - 
                                   FIRST_NONCALLABLE_SPEC_OBJECT_TYPE);
    llvm::Value* cmp = __ CreateICmpUGE(sub, imm);
    __ CreateCondBr(cmp, Use(instr->SuccessorAt(1)), continue_);
    __ SetInsertPoint(continue_);
  }
  
  llvm::BasicBlock* loop = NewBlock("DoClassOfTestAndBranch loop");
  llvm::BasicBlock* loop_not_smi = NewBlock("DoClassOfTestAndBranch "
                                            "loop not_smi");
  llvm::BasicBlock* equal = NewBlock("DoClassOfTestAndBranch loop type equal");
  llvm::BasicBlock* done = NewBlock("DoClassOfTestAndBranch done");
  llvm::Value* map = LoadFieldOperand(temp,
                                      Map::kConstructorOrBackPointerOffset);
  llvm::Value* new_map = nullptr;
  __ CreateBr(loop);

  __ SetInsertPoint(loop);
  llvm::PHINode* phi_map = __ CreatePHI(Types::i64, 2);
  phi_map->addIncoming(map, insert); 
  llvm::Value* map_is_smi = SmiCheck(phi_map);
  __ CreateCondBr(map_is_smi, done, loop_not_smi);

  __ SetInsertPoint(loop_not_smi);
  llvm::Value* scratch = LoadFieldOperand(phi_map, HeapObject::kMapOffset);
  llvm::Value* other_map = LoadFieldOperand(scratch, Map::kInstanceTypeOffset);
  llvm::Value* type_cmp = __ CreateICmpNE(other_map, __ getInt64(static_cast<int>(MAP_TYPE)));
  __ CreateCondBr(type_cmp, done, equal);

  __ SetInsertPoint(equal);
  new_map = LoadFieldOperand(phi_map, Map::kConstructorOrBackPointerOffset);
  phi_map->addIncoming(new_map, equal);
  __ CreateBr(loop);

  //TODO: need solv dominate all uses for other_map
  llvm::Value* zero = __ getInt64(0);
  __ SetInsertPoint(done);
  llvm::PHINode* phi_instance = __ CreatePHI(Types::i64, 2);
  phi_instance->addIncoming(zero, insert);
  phi_instance->addIncoming(other_map, loop_not_smi);

  llvm::Value* func_type = __ getInt64(static_cast<int>(JS_FUNCTION_TYPE));
  llvm::Value* CmpInstance = __ CreateICmpNE(phi_instance, func_type);
  llvm::BasicBlock* after_cmp_instance = NewBlock("DoClassOfTestAndBranch after "
                                                  "CmpInstance");
  if (String::Equals(class_name, isolate()->factory()->Object_string())) {
    __ CreateCondBr(CmpInstance, Use(instr->SuccessorAt(0)),
                                 after_cmp_instance);
    __ SetInsertPoint(after_cmp_instance);
  } else {
    __ CreateCondBr(CmpInstance, Use(instr->SuccessorAt(1)),
                                 after_cmp_instance);
    __ SetInsertPoint(after_cmp_instance);
  }
  
  llvm::Value* shared_info = LoadFieldOperand(phi_map, JSFunction::kSharedFunctionInfoOffset);
  llvm::Value* instance_class_name = LoadFieldOperand(shared_info,
                                     SharedFunctionInfo::kInstanceClassNameOffset);
  DCHECK(class_name->IsInternalizedString());
  llvm::Value* result = nullptr;
  AllowDeferredHandleDereference smi_check;
  if (class_name->IsSmi()) {
    UNIMPLEMENTED();
  } else {
    llvm::Value* name = MoveHeapObject(class_name);
    result =  __ CreateICmpEQ(instance_class_name, name);
  }
  __ CreateCondBr(result, Use(instr->SuccessorAt(0)), Use(instr->SuccessorAt(1)));
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
    llvm::Value* compare = __ CreateICmp(pred, Use(left), Use(right));
    llvm::Value* branch = __ CreateCondBr(compare,
        Use(instr->SuccessorAt(0)), Use(instr->SuccessorAt(1)));
    instr->set_llvm_value(branch);
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
  Handle<Code> ic = Handle<Code>::null();
  {
     AllowHandleAllocation allow_handles;
     AllowHeapAllocation allow_heap;
     ic = CodeFactory::CompareIC(isolate(), op, instr->strength()).code();
  }
  llvm::CmpInst::Predicate pred = TokenToPredicate(op, false, false);
  auto context = Use(instr->context());
  auto left = Use(instr->left());
  auto right = Use(instr->right());
  std::vector<llvm::Value*> params = { context, left, right };
  auto result = CallCode(ic, llvm::CallingConv::X86_64_V8_S10, params);
  result = __ CreateBitOrPointerCast(result, Types::i64);
  // Lithium comparison is a little strange, I think mine is all right.
  auto compare_result = __ CreateICmp(pred, result, __ getInt64(0));
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
  //UNIMPLEMENTED(); // calling convention should be v8_ic
}

void LLVMChunkBuilder::DoCompareMinusZeroAndBranch(HCompareMinusZeroAndBranch* instr) {
  //UNIMPLEMENTED();
  Representation rep = instr->value()->representation();

  if (rep.IsDouble()) {
    llvm::Value* zero = llvm::ConstantFP::get(Types::float64, 0);
    llvm::Value* not_zero = __ CreateFCmpONE(Use(instr->value()), zero);
    llvm::BasicBlock* is_zero = NewBlock("Instruction value is zero");
    __ CreateCondBr(not_zero, Use(instr->SuccessorAt(1)), is_zero);
    __ SetInsertPoint(is_zero);
    llvm::Value* cmp = __ CreateFCmpONE(Use(instr->value()), zero);
    llvm::BranchInst* branch = __ CreateCondBr(cmp, Use(instr->SuccessorAt(0)), Use(instr->SuccessorAt(1)));
    instr->set_llvm_value(branch);
  } else {
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoCompareObjectEqAndBranch(HCompareObjectEqAndBranch* instr) {
  //TODO: Test this case. charCodeAt function
  llvm::Value* cmp = nullptr;
  if (instr->right()->IsConstant()) {
    HConstant* constant = HConstant::cast(instr->right());
    Handle<Object> handle_value = constant->handle(isolate());
    llvm::Value* obj = MoveHeapObject(handle_value);
    cmp = __ CreateICmpEQ(Use(instr->left()), obj);
  } else {
    cmp = __ CreateICmpEQ(Use(instr->left()), Use(instr->right()));
  }
  __ CreateCondBr(cmp, Use(instr->SuccessorAt(0)), Use(instr->SuccessorAt(1)));
}

void LLVMChunkBuilder::DoCompareMap(HCompareMap* instr) {
   auto compare = CompareMap(Use(instr->value()), instr->map().handle());
   llvm::BranchInst* branch = __ CreateCondBr(compare,
         Use(instr->SuccessorAt(0)), Use(instr->SuccessorAt(1)));
   instr->set_llvm_value(branch);
}

void LLVMChunkBuilder::DoConstructDouble(HConstructDouble* instr) {
  //TODO Not tested.
  llvm::Value* hi = Use(instr->hi());
  llvm::Value* lo = Use(instr->lo());
  llvm::Value* hi_ext = __ CreateZExt(hi, Types::i64);
  llvm::Value* hi_shift = __ CreateShl(hi_ext, __ getInt64(32));
  llvm::Value* result = __ CreateOr(hi_shift, lo);
  llvm::Value* result_double = __ CreateSIToFP(result, Types::float64);
  instr->set_llvm_value(result_double);
}

int64_t LLVMChunkBuilder::RootRegisterDelta(ExternalReference other) {
  if (//predictable_code_size() &&
      (other.address() < reinterpret_cast<Address>(isolate()) ||
       other.address() >= reinterpret_cast<Address>(isolate() + 1))) {
    return -1;
  }
  Address roots_register_value = kRootRegisterBias +
      reinterpret_cast<Address>(isolate()->heap()->roots_array_start());

  int64_t delta = -1;
  if (kPointerSize == kInt64Size) {
    delta = other.address() - roots_register_value;
  } else {
    uint64_t o = static_cast<uint32_t>(
        reinterpret_cast<intptr_t>(other.address()));
    uint64_t r = static_cast<uint32_t>(
        reinterpret_cast<intptr_t>(roots_register_value));
    delta = o + r;
  }
  return delta;
}

llvm::Value* LLVMChunkBuilder::ExternalOperand(ExternalReference target) {
  //if (root_array_available_ && oserializer_enabled()) {
    int64_t delta = RootRegisterDelta(target);
    Address root_array_start_address =
          ExternalReference::roots_array_start(isolate()).address();
    auto int64_address =
        __ getInt64(reinterpret_cast<uint64_t>(root_array_start_address));
    auto load_address = ConstructAddress(int64_address, delta);
    auto casted_address = __ CreateBitCast(load_address, Types::ptr_i64);
    llvm::Value* object = __ CreateLoad(casted_address);
    return object;
  //}
}

void LLVMChunkBuilder::PrepareCallCFunction(int num_arguments) {
  int frame_alignment = base::OS::ActivationFrameAlignment();
  DCHECK(frame_alignment != 0);
  DCHECK(num_arguments >= 0);
  int argument_slots_on_stack =
      ArgumentStackSlotsForCFunctionCall(num_arguments);
  // Reading from rsp
  LLVMContext& llvm_context = LLVMGranularity::getInstance().context();
  llvm::Function* read_from_rsp = llvm::Intrinsic::getDeclaration(module_.get(),
      llvm::Intrinsic::read_register, { Types::i64 });
  auto metadata =
    llvm::MDNode::get(llvm_context, llvm::MDString::get(llvm_context, "rsp"));
  llvm::MetadataAsValue* val = llvm::MetadataAsValue::get(
      llvm_context, metadata);
  auto rsp_value = __ CreateCall(read_from_rsp, val);
  //TODO Try to move rsp value
  auto sub_v = __ CreateNSWSub(rsp_value, __ getInt64((argument_slots_on_stack + 1) * kRegisterSize));
  auto and_v = __ CreateAnd(sub_v, __ getInt64(-frame_alignment));
  auto address = ConstructAddress(and_v, argument_slots_on_stack * kRegisterSize);
  auto casted_address = __ CreateBitCast(address, Types::ptr_i64);
  __ CreateStore(rsp_value, casted_address);
}

int LLVMChunkBuilder::ArgumentStackSlotsForCFunctionCall(int num_arguments) {
  DCHECK(num_arguments >= 0);
  const int kRegisterPassedArguments = 6;
  if (num_arguments < kRegisterPassedArguments) return 0;
  return num_arguments - kRegisterPassedArguments;
}

llvm::Value* LLVMChunkBuilder::LoadAddress(ExternalReference source) {
  const int64_t kInvalidRootRegisterDelta = -1;
  llvm::Value* object = nullptr;
  //if (root_array_available_ && !serializer_enabled()) {
    int64_t delta = RootRegisterDelta(source);
    if (delta != kInvalidRootRegisterDelta && is_int32(delta)) {
      Address root_array_start_address =
          ExternalReference::roots_array_start(isolate()).address();
      auto int64_address =
          __ getInt64(reinterpret_cast<uint64_t>(root_array_start_address));
      object = LoadFieldOperand(int64_address, static_cast<int32_t>(delta));
      return object;
    } else {
    llvm::Value* address = __ getInt64(reinterpret_cast<uint64_t>(ExternalReference::get_date_field_function(isolate()).address()));
    auto constructed_address = ConstructAddress(address, 0);
    object = __ CreateLoad(constructed_address);
    return object;}
}

llvm::Value* LLVMChunkBuilder::CallCFunction(ExternalReference function,
                                             std::vector<llvm::Value*> params,
                                             int num_arguments) {
  if (emit_debug_code()) {
    UNIMPLEMENTED();
  }
  llvm::Value* obj = LoadAddress(function);
  llvm::Value* call = CallVal(obj, llvm::CallingConv::X86_64_V8_S3,
                              params, Types::tagged);
  DCHECK(base::OS::ActivationFrameAlignment() != 0);
  DCHECK(num_arguments >= 0);
  int argument_slots_on_stack = ArgumentStackSlotsForCFunctionCall(num_arguments);
  LLVMContext& llvm_context = LLVMGranularity::getInstance().context();
  llvm::Function* intrinsic_read = llvm::Intrinsic::getDeclaration(module_.get(),
      llvm::Intrinsic::read_register, { Types::i64 });
  auto metadata =
    llvm::MDNode::get(llvm_context, llvm::MDString::get(llvm_context, "rsp"));
  llvm::MetadataAsValue* val = llvm::MetadataAsValue::get(
      llvm_context, metadata);
  llvm::Value* rsp_value = __ CreateCall(intrinsic_read, val);
  llvm::Value* address = ConstructAddress(rsp_value, argument_slots_on_stack * kRegisterSize);
  llvm::Value* casted_address = __ CreateBitCast(address, Types::ptr_i64);
  llvm::Value* object = __ CreateLoad(casted_address);
  //Write into rsp
  std::vector<llvm::Value*> parameter = {val, object};
  llvm::Function* intrinsic_write = llvm::Intrinsic::getDeclaration(module_.get(),
      llvm::Intrinsic::write_register, { Types::i64 });
  __ CreateCall(intrinsic_write, parameter);
 return call;
}

void LLVMChunkBuilder::DoDateField(HDateField* instr) {
  llvm::BasicBlock* date_field_equal = nullptr;
  llvm::BasicBlock* date_field_runtime = NewBlock("Runtime");
  llvm::BasicBlock* DateFieldResult = NewBlock("Result block of DateField");
  llvm::Value* date_field_result_equal = nullptr;
  Smi* index = instr->index();

  if (FLAG_debug_code) {
    AssertNotSmi(Use(instr->value()));
    llvm::Value* map = FieldOperand(Use(instr->value()),
            HeapObject::kMapOffset);
    llvm::Value* cast_int = __ CreateBitCast(map, Types::ptr_i64);
    llvm::Value* address = __ CreateLoad(cast_int);
    llvm::Value* DateObject = LoadFieldOperand(address, Map::kMapOffset);
    llvm::Value* object_type_check = __ CreateICmpEQ(DateObject,
           __ getInt64(static_cast<int8_t>(JS_DATE_TYPE)));
    Assert(object_type_check);
  }

  if (index->value() == 0) {
    llvm::Value* map = LoadFieldOperand(Use(instr->value()), JSDate::kValueOffset);
    instr->set_llvm_value(map);
  } else {
    if (index->value() < JSDate::kFirstUncachedField) {
      date_field_equal = NewBlock("equal");
      ExternalReference stamp = ExternalReference::date_cache_stamp(isolate());
      llvm::Value* stamp_object = ExternalOperand(stamp);
      llvm::Value* object = LoadFieldOperand(Use(instr->value()), JSDate::kCacheStampOffset);
      llvm::Value* not_equal = __ CreateICmp(llvm::CmpInst::ICMP_NE, stamp_object, object);
      __ CreateCondBr(not_equal, date_field_runtime, date_field_equal);
      __ SetInsertPoint(date_field_equal);
      date_field_result_equal = LoadFieldOperand(Use(instr->value()), JSDate::kValueOffset +
                                                                   kPointerSize * index->value());
      __ CreateBr(DateFieldResult);
    }
    __ SetInsertPoint(date_field_runtime);
    PrepareCallCFunction(2);
    llvm::Value* param_one = Use(instr->value());
    intptr_t intptr_value = reinterpret_cast<intptr_t>(index);
    llvm::Value* param_two = __ getInt64(intptr_value);
    std::vector<llvm::Value*> params = { param_one, param_two };
    llvm::Value* result = CallCFunction(ExternalReference::get_date_field_function(isolate()), params, 2);
    llvm::Value* date_field_result_runtime = __ CreatePtrToInt(result, Types::i64);
    __ CreateBr(DateFieldResult);
    __ SetInsertPoint(DateFieldResult);
    if (date_field_equal) {
       llvm::PHINode* phi = __ CreatePHI(Types::i64, 2);
       phi->addIncoming(date_field_result_equal, date_field_equal);
       phi->addIncoming(date_field_result_runtime, date_field_runtime);
       instr->set_llvm_value(phi);
    } else {
       instr->set_llvm_value(date_field_result_runtime);
    }
  }
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
  // DCHECK(type == Deoptimizer::EAGER);
  auto reason = instr->reason();
  USE(reason);
  bool negate_condition = false;
  // It's unreacheable, but we don't care. We need it so that DeoptimizeIf()
  // does not create a new basic block which ends up unterminated.
  auto next_block = Use(instr->SuccessorAt(0));
  DeoptimizeIf(__ getTrue(), instr->reason(), negate_condition, next_block);
}

void LLVMChunkBuilder::DoDiv(HDiv* instr) {
  if(instr->representation().IsInteger32() || instr->representation().IsSmi()) {
    DCHECK(instr->left()->representation().Equals(instr->representation()));
    DCHECK(instr->right()->representation().Equals(instr->representation()));
    HValue* dividend = instr->left();
    HValue* divisor = instr->right();
    llvm::Value* Div = __ CreateSDiv(Use(dividend), Use(divisor),"");
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
  llvm::Value* value = Use(instr->value());
  if (instr->bits() == HDoubleBits::HIGH) {
    llvm::Value* tmp = __ CreateBitCast(value, Types::i64);
    value = __ CreateLShr(tmp, __ getInt64(32));
    value = __ CreateTrunc(value, Types::i32);
  } else {
    UNIMPLEMENTED();
  }
  instr->set_llvm_value(value);
}

void LLVMChunkBuilder::DoDummyUse(HDummyUse* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoEnterInlined(HEnterInlined* instr) {
  //UNIMPLEMENTED();
  HEnvironment* outer = current_block_->last_environment();
  outer->set_ast_id(instr->ReturnId());
  HConstant* undefined = graph()->GetConstantUndefined();
  HEnvironment* inner = outer->CopyForInlining(instr->closure(),
                                               instr->arguments_count(),
                                               instr->function(),
                                               undefined,
                                               instr->inlining_kind());
  // Only replay binding of arguments object if it wasn't removed from graph.
  if (instr->arguments_var() != NULL && instr->arguments_object()->IsLinked()) {
    inner->Bind(instr->arguments_var(), instr->arguments_object());
  }
  inner->BindContext(instr->closure_context());
  inner->set_entry(instr);
  current_block_->UpdateEnvironment(inner);
  chunk()->AddInlinedFunction(instr->shared());
}

void LLVMChunkBuilder::DoEnvironmentMarker(HEnvironmentMarker* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoForceRepresentation(HForceRepresentation* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoForInCacheArray(HForInCacheArray* instr) {
  llvm::Value* map_val = Use(instr->map());
  llvm::BasicBlock* load_cache = NewBlock("LOAD CACHE");
  llvm::BasicBlock* done_block1 = NewBlock("DONE1");
  llvm::BasicBlock* done_block = NewBlock("DONE");
  
  llvm::Value* result = EnumLength(map_val);
  llvm::Value* cmp_neq = __ CreateICmpNE(result, __ getInt64(0));
  __ CreateCondBr(cmp_neq, load_cache, done_block1);
  __ SetInsertPoint(done_block1);
  llvm::Value* result1 = LoadRoot(Heap::kEmptyFixedArrayRootIndex);
  __ CreateBr(done_block);
  __ SetInsertPoint(load_cache);
  result = LoadFieldOperand(map_val, Map::kDescriptorsOffset);
  result = LoadFieldOperand(result, DescriptorArray::kEnumCacheOffset);
  result = LoadFieldOperand(result, FixedArray::SizeFor(HForInCacheArray::cast(instr)->idx()));
  __ CreateBr(done_block);
  __ SetInsertPoint(done_block);
  llvm::PHINode* phi = __ CreatePHI(Types::tagged, 2);
  phi->addIncoming(result1, done_block1);
  phi->addIncoming(result, load_cache);
  llvm::Value* cond = SmiCheck(phi, true);
  DeoptimizeIf(cond,  Deoptimizer::kNoCache, true);
  instr->set_llvm_value(phi);
}

llvm::Value* LLVMChunkBuilder::EnumLength(llvm::Value* map) {
  STATIC_ASSERT(Map::EnumLengthBits::kShift == 0);
  llvm::Value* length = LoadFieldOperand(map, Map::kBitField3Offset);
  llvm::Value* tagged = __ CreatePtrToInt(length, Types::i64);
  llvm::Value* length32 = __ CreateIntCast(tagged, Types::i32, true);
  llvm::Value* imm = __ getInt32(Map::EnumLengthBits::kMask);
  llvm::Value* result = __ CreateAnd(length32, imm);
  return Integer32ToSmi(result);
}

void LLVMChunkBuilder::CheckEnumCache(HValue* enum_val, llvm::Value* val,
                                      llvm::BasicBlock* call_runtime) {
  llvm::BasicBlock* next = NewBlock("CheckEnumCache next");
  llvm::BasicBlock* start = NewBlock("CheckEnumCache start");
  llvm::Value* copy_enumerable = Use(enum_val);
  llvm::Value* empty_array = LoadRoot(Heap::kEmptyFixedArrayRootIndex);
  // Check if the enum length field is properly initialized, indicating that
  // there is an enum cache.
  llvm::Value* map = LoadFieldOperand(copy_enumerable, HeapObject::kMapOffset);
  llvm::Value* length = EnumLength(map);
  Smi* invalidEnum = Smi::FromInt(kInvalidEnumCacheSentinel);
  llvm::Value* enum_length = ValueFromSmi(invalidEnum);
  llvm::Value* cmp = __ CreateICmpEQ(length, enum_length);
  __ CreateCondBr(cmp, call_runtime, start);

  __ SetInsertPoint(next);
  map = LoadFieldOperand(copy_enumerable, HeapObject::kMapOffset);

  // For all objects but the receiver, check that the cache is empty.
  length = EnumLength(map);
  llvm::Value* test = __ CreateAnd(length, length);
  llvm::Value* zero = __ getInt64(0);
  llvm::Value* cmp_zero = __ CreateICmpNE(test, zero);
  __ CreateCondBr(cmp_zero, call_runtime, start);

  __ SetInsertPoint(start);
  // Check that there are no elements. Register rcx contains the current JS
  // object we've reached through the prototype chain.
  llvm::BasicBlock* no_elements = NewBlock("CheckEnumCache no_elements");
  llvm::BasicBlock* second_chance = NewBlock("CheckEnumCache Second chance");
  llvm::Value* object = LoadFieldOperand(copy_enumerable,
                                         JSObject::kElementsOffset);
  llvm::Value* cmp_obj = __ CreateICmpEQ(empty_array, object);
  __ CreateCondBr(cmp_obj, no_elements, second_chance);

  __ SetInsertPoint(second_chance);
  llvm::Value* slow = LoadRoot( Heap::kEmptySlowElementDictionaryRootIndex);
  llvm::Value* second_cmp_obj = __ CreateICmpNE(slow, object);
  __ CreateCondBr(second_cmp_obj, call_runtime, no_elements);
  
  __ SetInsertPoint(no_elements);
  copy_enumerable = LoadFieldOperand(map, Map::kPrototypeOffset);
  llvm::BasicBlock* final_block = NewBlock("CheckEnumCache end");
  llvm::Value* final_cmp = __ CreateICmpNE(copy_enumerable, val);
  __ CreateCondBr(final_cmp, next, final_block);
  
  __ SetInsertPoint(final_block);
}


void LLVMChunkBuilder::DoForInPrepareMap(HForInPrepareMap* instr) {
  llvm::Value* enumerable = Use(instr->enumerable());
  llvm::Value* smi_check = SmiCheck(enumerable);
  DeoptimizeIf(smi_check, Deoptimizer::kSmi);

  STATIC_ASSERT(FIRST_JS_PROXY_TYPE == FIRST_SPEC_OBJECT_TYPE);
  llvm::Value* cmp_type = CmpObjectType(enumerable, LAST_JS_PROXY_TYPE,
                                        llvm::CmpInst::ICMP_ULE);
  DeoptimizeIf(cmp_type, Deoptimizer::kWrongInstanceType);

  llvm::Value* root = LoadRoot(Heap::kNullValueRootIndex);
  llvm::BasicBlock* call_runtime = NewBlock("DoForInPrepareMap call runtime");
  llvm::BasicBlock* merge = NewBlock("DoForInPrepareMap use cache");
  CheckEnumCache(instr->enumerable(), root, call_runtime);
  llvm::BasicBlock* insert = __ GetInsertBlock();
  llvm::Value* map = LoadFieldOperand(enumerable, HeapObject::kMapOffset);
  __ CreateBr(merge);

  // Get the set of properties to enumerate.
  __ SetInsertPoint(call_runtime);
  std::vector<llvm::Value*> args;
  args.push_back(enumerable);
  llvm::Value* set =  CallRuntimeFromDeferred(Runtime::kGetPropertyNamesFast,
                                                Use(instr->context()), args);
  llvm::Value* runtime_map = LoadFieldOperand(set, HeapObject::kMapOffset);
  llvm::Value* cmp_root = CompareRoot(runtime_map, Heap::kMetaMapRootIndex,
                                      llvm::CmpInst::ICMP_NE);
  DeoptimizeIf(cmp_root, Deoptimizer::kWrongMap);
  __ CreateBr(merge);

  __ SetInsertPoint(merge);
  llvm::PHINode* phi = __ CreatePHI(Types::tagged, 2);
  phi->addIncoming(map, insert);
  phi->addIncoming(set, call_runtime);
  instr->set_llvm_value(phi);
}

void LLVMChunkBuilder::DoGetCachedArrayIndex(HGetCachedArrayIndex* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoHasCachedArrayIndexAndBranch(HHasCachedArrayIndexAndBranch* instr) {
  UNIMPLEMENTED();
}

static InstanceType TestType(HHasInstanceTypeAndBranch* instr) {
  InstanceType from = instr->from();
  InstanceType to = instr->to();
  if (from == FIRST_TYPE) return to;
  DCHECK(from == to || to == LAST_TYPE);
  return from;
}

void LLVMChunkBuilder::DoHasInstanceTypeAndBranch(HHasInstanceTypeAndBranch* instr) {
  llvm::Value* input = Use(instr->value());
  llvm::BasicBlock* near = NewBlock("HasInstanceTypeAndBranch Near");
  llvm::BranchInst* branch = nullptr;
 
  if (!instr->value()->type().IsHeapObject()) {
    llvm::Value* smi_cond = SmiCheck(input);
    branch = __ CreateCondBr(smi_cond, Use(instr->SuccessorAt(1)), near);
    __ SetInsertPoint(near);
  }


  auto cmp = CmpObjectType(input, TestType(instr));
  branch = __ CreateCondBr(cmp, Use(instr->SuccessorAt(0)), 
                           Use(instr->SuccessorAt(1)));
  instr->set_llvm_value(branch);
}

void LLVMChunkBuilder::DoInnerAllocatedObject(HInnerAllocatedObject* instr) {
  if(instr->offset()->IsConstant()) {
    uint32_t offset = (HConstant::cast(instr->offset()))->Integer32Value();
    llvm::Value* gep = ConstructAddress(Use(instr->base_object()), offset);
    instr->set_llvm_value(gep);
    //UNIMPLEMENTED();
  } else {
    UNIMPLEMENTED();
  }
  //UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoInstanceOf(HInstanceOf* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoHasInPrototypeChainAndBranch(
    HHasInPrototypeChainAndBranch* instr) {
  llvm::BasicBlock* insert = __ GetInsertBlock();
  llvm::Value* object = Use(instr->object());
  llvm::Value* prototype = Use(instr->prototype());
  if (instr->ObjectNeedsSmiCheck()) {
    llvm::BasicBlock* check_smi = NewBlock("DoHasInPrototypeChainAndBranch"
                                           " after check smi");
    llvm::Value* is_smi = SmiCheck(object, false);
    __ CreateCondBr(is_smi, Use(instr->SuccessorAt(1)), check_smi);

    __ SetInsertPoint(check_smi);
  }

  llvm::BasicBlock* loop = NewBlock("DoHasInPrototypeChainAndBranch loop");
  llvm::Value* object_map = LoadFieldOperand(object, HeapObject::kMapOffset);
  llvm::BasicBlock* after_compare_root = NewBlock("DoHasInPrototypeChainAndBranch"
                                                  " after compare root");
  llvm::Value* load_object_map;
  __ CreateBr(loop);
  __ SetInsertPoint(loop);
  llvm::PHINode* phi = __ CreatePHI(Types::tagged, 2);
  phi->addIncoming(object_map, insert);
  llvm::Value* object_prototype = LoadFieldOperand(phi,
                                                   Map::kPrototypeOffset);
  llvm::Value* cmp = __ CreateICmpEQ(object_prototype, prototype);
  llvm::BasicBlock* compare_root = NewBlock("DoHasInPrototypeChainAndBranch"
                                           " compare root");
  __ CreateCondBr(cmp, Use(instr->SuccessorAt(0)), compare_root);

  __ SetInsertPoint(compare_root);
  llvm::Value* cmp_root = CompareRoot(object_prototype,
                                      Heap::kNullValueRootIndex);
  __ CreateCondBr(cmp_root, Use(instr->SuccessorAt(1)), after_compare_root);

  __ SetInsertPoint(after_compare_root);
  load_object_map = LoadFieldOperand(object_prototype, HeapObject::kMapOffset);
  phi->addIncoming(load_object_map, after_compare_root);

  __ CreateBr(loop);
}

void LLVMChunkBuilder::DoInvokeFunction(HInvokeFunction* instr) {
  //TODO: Not tested
  Handle<JSFunction> known_function = instr->known_function();
  if (known_function.is_null()) {
    UNIMPLEMENTED();
  } else {
    bool dont_adapt_arguments =
        instr->formal_parameter_count() == SharedFunctionInfo::kDontAdaptArgumentsSentinel;
    bool can_invoke_directly =
        dont_adapt_arguments || instr->formal_parameter_count() == (instr->argument_count()-1);
    if (can_invoke_directly) {
      llvm::Value* context = LoadFieldOperand(Use(instr->function()), JSFunction::kContextOffset);

      if (dont_adapt_arguments) {
        UNIMPLEMENTED();
      }

      // InvokeF
      if (instr->known_function().is_identical_to(info()->closure())) {
        UNIMPLEMENTED();
      } else {
        std::vector<llvm::Value*> params;
        params.push_back(context);
        params.push_back(Use(instr->function()));
        for (int i = pending_pushed_args_.length()-1; i >=0; --i)
          params.push_back(pending_pushed_args_[i]);
        pending_pushed_args_.Clear();
        // callingConv 
        llvm::Value* call = CallVal(LoadFieldOperand(Use(instr->function()), JSFunction::kCodeEntryOffset),
                                    llvm::CallingConv::X86_64_V8_S4, params, Types::tagged);
        llvm::Value* return_val = __ CreatePtrToInt(call, Types::i64);
        instr->set_llvm_value(return_val);
      }
      //TODO: Implement SafePoint with lazy deopt
    } else {
      UNIMPLEMENTED();
    }
  }
}

void LLVMChunkBuilder::DoIsConstructCallAndBranch(
    HIsConstructCallAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoIsStringAndBranch(HIsStringAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoIsSmiAndBranch(HIsSmiAndBranch* instr) {
  //UNIMPLEMENTED();
  llvm::Value* input = Use(instr->value());
  llvm::Value* is_smi = SmiCheck(input);
  llvm::BranchInst* branch = __ CreateCondBr(is_smi,
         Use(instr->SuccessorAt(0)), Use(instr->SuccessorAt(1)));
  instr->set_llvm_value(branch);
}

void LLVMChunkBuilder::DoIsUndetectableAndBranch(HIsUndetectableAndBranch* instr) {
  if (!instr->value()->type().IsHeapObject()) {
    llvm::BasicBlock* after_check_smi = NewBlock("IsUndetectableAndBranch after check smi");
    llvm::Value* smi_cond = SmiCheck(Use(instr->value()), false); 
     __ CreateCondBr(smi_cond, Use(instr->SuccessorAt(1)), after_check_smi);
     __ SetInsertPoint(after_check_smi);
  }
  llvm::Value* map = LoadFieldOperand(Use(instr->value()), HeapObject::kMapOffset); 
  llvm::Value* bitFiledOffset = LoadFieldOperand(map, Map::kBitFieldOffset);
  llvm::Value* test = __ CreateICmpEQ(bitFiledOffset, __ getInt64(1 << Map::kIsUndetectable));
  llvm::Value* result = __ CreateCondBr(test, Use(instr->SuccessorAt(0)), Use(instr->SuccessorAt(1)));
  instr->set_llvm_value(result);
}

void LLVMChunkBuilder::DoLeaveInlined(HLeaveInlined* instr) {
  //UNIMPLEMENTED();
  HEnvironment* env = current_block_->last_environment();

  if (env->entry()->arguments_pushed()) {
    UNIMPLEMENTED();
    /*int argument_count = env->arguments_environment()->parameter_count();
    pop = new(zone()) LDrop(argument_count);
    DCHECK(instr->argument_delta() == -argument_count);*/
  }

  HEnvironment* outer = current_block_->last_environment()->
      DiscardInlined(false);
  current_block_->UpdateEnvironment(outer);
}

void LLVMChunkBuilder::DoLoadContextSlot(HLoadContextSlot* instr) {
  llvm::Value* context = Use(instr->value());
  llvm::BasicBlock* insert = __ GetInsertBlock(); 
  auto offset = Context::SlotOffset(instr->slot_index());
  llvm::Value* result_addr = ConstructAddress(context, offset);
  llvm::Value* result_casted = __ CreateBitCast(result_addr, Types::ptr_tagged);
  llvm::Value* result = __ CreateLoad(result_casted);
  llvm::Value* root = nullptr;
  llvm::BasicBlock* load_root = nullptr;
 
  int count = 1;
  if (instr->RequiresHoleCheck()) {
    llvm::Value* cmp_root = CompareRoot(result, Heap::kTheHoleValueRootIndex);
    if (instr->DeoptimizesOnHole()) {
      UNIMPLEMENTED();
    } else {
      load_root = NewBlock("DoLoadContextSlot load root");
      llvm::BasicBlock* is_not_hole = NewBlock("DoLoadContextSlot" 
                                               "is not hole");
      count = 2;
      __ CreateCondBr(cmp_root, load_root, is_not_hole);
      __ SetInsertPoint(load_root);
      root = LoadRoot(Heap::kUndefinedValueRootIndex);
      __ CreateBr(is_not_hole);

      __ SetInsertPoint(is_not_hole);
    }
  }
  
  if (count == 1) {
    instr->set_llvm_value(result);
  } else {
    llvm::PHINode* phi = __ CreatePHI(Types::tagged, 2);
    phi->addIncoming(result, insert);
    phi->addIncoming(root, load_root);
    instr->set_llvm_value(phi);
  }
}

void LLVMChunkBuilder::DoLoadFieldByIndex(HLoadFieldByIndex* instr) {
  llvm::Value* object = Use(instr->object());
  llvm::Value* index = nullptr;
  if (instr->index()->representation().IsTagged()) {
    llvm::Value* temp = Use(instr->index());
    index = __ CreatePtrToInt(temp, Types::i64);
  } else {
    index = Use(instr->index());
  }

  // DeferredLoadMutableDouble case does not implemented,
  llvm::BasicBlock* out_of_obj = NewBlock("OUT OF OBJECT");
  llvm::BasicBlock* done1 = NewBlock("DONE1");
  llvm::BasicBlock* done = NewBlock("DONE");
  /*llvm::Value* smi_tmp_val = __ CreateZExt(__ getInt64(1), Types::i64);
  llvm::Value* smi_val = __ CreateShl(smi_tmp_val, kSmiShift);*/
  /*llvm::Value* tmp_val = __ CreateAnd(val2, smi_val);
  llvm::Value* test = __ CreateICmpNE(tmp_val, __ getInt64(0));*/
  llvm::Value* smi_tmp = __ CreateAShr(index, __ getInt64(1));
  index = __ CreateLShr(smi_tmp, kSmiShift);
  index = __ CreateTrunc(index, Types::i32);
  llvm::Value* cmp_less = __ CreateICmpSLT(index, __ getInt32(0));
  __ CreateCondBr(cmp_less, out_of_obj, done1);
  __ SetInsertPoint(done1);
  //FIXME: Use BuildFastArrayOperand
  llvm::Value* scale = __ getInt32(8);
  llvm::Value* offset = __ getInt32(JSObject::kHeaderSize);
  llvm::Value* mul = __ CreateMul(index, scale);
  llvm::Value* add = __ CreateAdd(mul, offset);
  llvm::Value* int_ptr = __ CreateIntToPtr(object, Types::ptr_i8);
  llvm::Value* gep_0 = __ CreateGEP(int_ptr, add);
  llvm::Value* tmp1 = __ CreateBitCast(gep_0, Types::ptr_i64);
  llvm::Value* int64_val1 = __ CreateLoad(tmp1);
  __ CreateBr(done);
  __ SetInsertPoint(out_of_obj);
  scale = __ getInt64(8);
  offset = __ getInt64(JSObject::kHeaderSize-kPointerSize);
  llvm::Value* v2 = LoadFieldOperand(object, JSObject::kPropertiesOffset);
  llvm::Value* int64_val = __ CreatePtrToInt(v2, Types::i64);
  llvm::Value* neg_val1 = __ CreateNeg(int64_val);
  llvm::Value* mul1 = __ CreateMul(neg_val1, scale);
  llvm::Value* add1 = __ CreateAdd(mul1, offset);
  llvm::Value* int_ptr1 = __ CreateIntToPtr(v2, Types::ptr_i8);
  llvm::Value* v3 =  __ CreateGEP(int_ptr1, add1);
  llvm::Value* tmp2 = __ CreateBitCast(v3, Types::ptr_i64);
  llvm::Value* int64_val2 = __ CreateLoad(tmp2);
  __ CreateBr(done);
  __ SetInsertPoint(done);
  llvm::PHINode* phi = __ CreatePHI(Types::i64, 2);
  phi->addIncoming(int64_val1, done1);
  phi->addIncoming(int64_val2, out_of_obj);
  llvm::Value* phi_tagged = __ CreateIntToPtr(phi, Types::tagged);
  instr->set_llvm_value(phi_tagged);
}

void LLVMChunkBuilder::DoLoadFunctionPrototype(HLoadFunctionPrototype* instr) {
  llvm::BasicBlock* insert = __ GetInsertBlock();
  llvm::Value* function = Use(instr->function());
  llvm::BasicBlock* equal = NewBlock("LoadFunctionPrototype Equal");
  llvm::BasicBlock* done = NewBlock("LoadFunctionPrototype Done");

  // Get the prototype or initial map from the function.
  llvm::Value* load_func = LoadFieldOperand(function,
                           JSFunction::kPrototypeOrInitialMapOffset);

  // Check that the function has a prototype or an initial map.
  llvm::Value* cmp_root = CompareRoot(load_func, Heap::kTheHoleValueRootIndex);
  DeoptimizeIf(cmp_root, Deoptimizer::kHole);

  // If the function does not have an initial map, we're done.
  llvm::Value* result = LoadFieldOperand(load_func, HeapObject::kMapOffset);
  llvm::Value* map = LoadFieldOperand(result, Map::kInstanceTypeOffset);
  llvm::Value* map_type = __ getInt64(static_cast<int>(MAP_TYPE));
  llvm::Value* cmp_type = __ CreateICmpNE(map, map_type);
  __ CreateCondBr(cmp_type, done, equal);

  __ SetInsertPoint(equal);
  // Get the prototype from the initial map.
  llvm::Value* get_prototype = LoadFieldOperand(load_func,
                                               Map::kPrototypeOffset);

  __ CreateBr(done);
  __ SetInsertPoint(done);
  llvm::PHINode* phi = __ CreatePHI(Types::i64, 2);
  phi->addIncoming(load_func, insert);
  phi->addIncoming(get_prototype, equal);
  instr->set_llvm_value(phi);
}

void LLVMChunkBuilder::DoLoadGlobalGeneric(HLoadGlobalGeneric* instr) {
  //TODO: Not tested, test case string-base64.js in  base64ToString finction
  llvm::Value* context = Use(instr->context());
  llvm::Value* global_object = Use(instr->global_object());
  llvm::Value* name = MoveHeapObject(instr->name());
  //llvm::Value* vector = nullptr;
   
  AllowDeferredHandleDereference vector_structure_check;
  Handle<TypeFeedbackVector> feedback_vector = instr->feedback_vector();
  llvm::Value* vector =  MoveHeapObject(feedback_vector);
  FeedbackVectorSlot instr_slot = instr->slot();
  int index = feedback_vector->GetIndex(instr_slot);
  llvm::Value* slot = __ getInt64(index);

  AllowHandleAllocation allow_handles;
  AllowHeapAllocation allow_heap;
  Handle<Code> ic =
        CodeFactory::LoadICInOptimizedCode(isolate(), instr->typeof_mode(),
                                             SLOPPY, PREMONOMORPHIC).code();
  std::vector<llvm::Value*> params;
  params.push_back(context);
  params.push_back(global_object);
  params.push_back(name);
  params.push_back(vector);
  params.push_back(slot);
  auto result = CallCode(ic, llvm::CallingConv::X86_64_V8_S9, params);
  instr->set_llvm_value(result);
}

void LLVMChunkBuilder::DoLoadKeyed(HLoadKeyed* instr) {
  //UNIMPLEMENTED(); // FIXME(llvm): there's no more is_typed_elements()
  if (instr->is_fixed_typed_array()) {
    DoLoadKeyedExternalArray(instr);

  } else if (instr->representation().IsDouble()) {
    DoLoadKeyedFixedDoubleArray(instr);
  } else {
    DoLoadKeyedFixedArray(instr);
  }
}

void LLVMChunkBuilder::DoLoadKeyedExternalArray(HLoadKeyed* instr) {
  //  UNIMPLEMENTED();
  //TODO: not tested string-validate-input.js in doTest 
  //TODO: Compare generated asm while testing
  HValue* key = instr->key();
  ElementsKind kind = instr->elements_kind();
  int32_t base_offset = instr->base_offset();
  llvm::Value* casted_address = nullptr;
  llvm::Value* load = nullptr;

  if (kPointerSize == kInt32Size && !key->IsConstant()) {
    UNIMPLEMENTED();
  }

  llvm::Value* elements = Use(instr->elements());
  llvm::Value* address = BuildFastArrayOperand(key, elements, 
                                               kind, base_offset);
  if (kind == FLOAT32_ELEMENTS) {
    casted_address = __ CreateBitCast(address, Types::ptr_float32);
    load = __ CreateLoad(casted_address);
    auto result = __ CreateFPExt(load, Types::float64);
    instr->set_llvm_value(result);
    // UNIMPLEMENTED();
  } else if (kind == FLOAT64_ELEMENTS) {
    casted_address = __ CreateBitCast(address, Types::ptr_float64);
    load = __ CreateLoad(casted_address);
    instr->set_llvm_value(load);
  } else {
    switch (kind) {
      case INT8_ELEMENTS: {
        //movsxbl(result, operand)
        casted_address = __ CreateBitCast(address, Types::ptr_i8);
        load = __ CreateLoad(casted_address);
        llvm::Value* result = __ CreateSExt(load, Types::i32);
        instr->set_llvm_value(result);
        /*
        llvm::Value* check_sign = __ CreateAnd(load, __ getInt64(0x80));
        llvm::Value* cmp = __ CreateICmpSGE(check_sign, __ getInt64(0)); 
        llvm::BasicBlock* negative = NewBlock("DoLoadKeyedExternalArray" 
                                              " negative");
        llvm::BasicBlock* positive = NewBlock("DoLoadKeyedExternalArray"
                                              " positive");
        llvm::BasicBlock* merge = NewBlock("DoLoadKeyedExternalArray merge");
        __ CreateCondBr(cmp, positive, negative);
        
        __ SetInsertPoint(positive);
        llvm::Value* positiv_result = __ CreateAnd()
        */
        break;
      }
      case UINT8_ELEMENTS:
      case UINT8_CLAMPED_ELEMENTS:{
        //movzxbl(result, operand)
        casted_address = __ CreateBitCast(address, Types::ptr_i8);
        load = __ CreateLoad(casted_address);
        llvm::Value* result = __ CreateZExt(load, Types::i32);
        instr->set_llvm_value(result);
        break;
      }
      case INT16_ELEMENTS:
        UNIMPLEMENTED();
        break;
      case UINT16_ELEMENTS:
        UNIMPLEMENTED();
        break;
      case INT32_ELEMENTS:
        casted_address = __ CreateBitCast(address, Types::ptr_i32);
        load = __ CreateLoad(casted_address);
        instr->set_llvm_value(load);
        break;
      case UINT32_ELEMENTS:
        casted_address = __ CreateBitCast(address, Types::ptr_i32);
        load = __ CreateLoad(casted_address);
        instr->set_llvm_value(load);
        if (!instr->CheckFlag(HInstruction::kUint32)) {
          UNIMPLEMENTED();
          //__ testl(result, result);
          //DeoptimizeIf(negative, instr, Deoptimizer::kNegativeValue);
        }
        break;
      case FLOAT32_ELEMENTS:
      case FLOAT64_ELEMENTS:
      case FAST_ELEMENTS:
      case FAST_SMI_ELEMENTS:
      case FAST_DOUBLE_ELEMENTS:
      case FAST_HOLEY_ELEMENTS:
      case FAST_HOLEY_SMI_ELEMENTS:
      case FAST_HOLEY_DOUBLE_ELEMENTS:
      case DICTIONARY_ELEMENTS:
      case FAST_SLOPPY_ARGUMENTS_ELEMENTS:
      case SLOW_SLOPPY_ARGUMENTS_ELEMENTS:
        UNREACHABLE();
        break;
    }
  }
}

llvm::Value* LLVMChunkBuilder::BuildFastArrayOperand(HValue* key, 
                                                     llvm::Value* elements,
                                                     ElementsKind elements_kind,
                                                     uint32_t inst_offset) {
  llvm::Value* address = nullptr;
  int shift_size = ElementsKindToShiftSize(elements_kind);
  if (key->IsConstant()) {
    uint32_t const_val = (HConstant::cast(key))->Integer32Value();
    address = ConstructAddress(elements, (const_val << shift_size) + inst_offset);
  } else {
    llvm::Value* lkey = Use(key);
    llvm::Value* scale = nullptr;
    llvm::Value* offset = nullptr;
    int scale_factor;
    switch (shift_size) {
      case 0:
        scale_factor = 1;
        break;
      case 1:
        scale_factor = 2;
        break;
      case 2:
        scale_factor = 4;
        break;
      case 3:
        scale_factor = 8;
        break;
      default:
        UNIMPLEMENTED();
    }
    if (key->representation().IsInteger32()) {
       scale = __ getInt32(scale_factor);
       offset = __ getInt32(inst_offset);
     } else {
       scale = __ getInt64(scale_factor);
       offset = __ getInt64(inst_offset);
     }
     llvm::Value* mul = __ CreateMul(lkey, scale);
     llvm::Value* add = __ CreateAdd(mul, offset);
     llvm::Value* int_ptr = __ CreateIntToPtr(elements, Types::ptr_i8);
     address = __ CreateGEP(int_ptr, add);
  }
  return address;
}

void LLVMChunkBuilder::DoLoadKeyedFixedDoubleArray(HLoadKeyed* instr) {
  HValue* key = instr->key();
  uint32_t inst_offset = instr->base_offset();
  if (kPointerSize == kInt32Size && !key->IsConstant() &&
      instr->IsDehoisted()) {
    UNIMPLEMENTED();
  }
  if (instr->RequiresHoleCheck()) {
    UNIMPLEMENTED();
  }
  llvm::Value* address = BuildFastArrayOperand(key, Use(instr->elements()),
                                       FAST_DOUBLE_ELEMENTS, inst_offset);
  auto casted_address = __ CreateBitCast(address, Types::ptr_float64);
  llvm::Value* load = __ CreateLoad(casted_address);
  instr->set_llvm_value(load);
}

void LLVMChunkBuilder::DoLoadKeyedFixedArray(HLoadKeyed* instr) {
  HValue* key = instr->key();
  Representation representation = instr->representation();
  bool requires_hole_check = instr->RequiresHoleCheck();
  uint32_t inst_offset = instr->base_offset();
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
    DCHECK(kSmiTagSize + kSmiShiftSize == 32);
    inst_offset += kPointerSize / 2;
  }
  llvm::Value* address = BuildFastArrayOperand(key, Use(instr->elements()),
                                       FAST_DOUBLE_ELEMENTS, inst_offset);
  llvm::Value* casted_address = nullptr;
  auto pointer_type = GetLLVMType(instr->representation())->getPointerTo();
  casted_address = __ CreateBitCast(address, pointer_type);
  llvm::Value* load = __ CreateLoad(casted_address);

  if (requires_hole_check) {
    if (IsFastSmiElementsKind(instr->elements_kind())) {
      bool check_non_smi = true;
      llvm::Value* cmp = SmiCheck(load, check_non_smi);
      DeoptimizeIf(cmp, Deoptimizer::kNotASmi);
    } else {
      // FIXME(access-nsieve): not tested
      llvm::Value* cmp = CompareRoot(load, Heap::kTheHoleValueRootIndex);
      DeoptimizeIf(cmp, Deoptimizer::kHole);
    }
  } else if (instr->hole_mode() == CONVERT_HOLE_TO_UNDEFINED) {
    DCHECK(instr->elements_kind() == FAST_HOLEY_ELEMENTS);
    llvm::BasicBlock* merge = NewBlock("DoLoadKeyedFixedArray merge");
    llvm::BasicBlock* after_cmp = NewBlock("DoLoadKeyedFixedArray after cmp");
    llvm::BasicBlock* insert = __ GetInsertBlock();
    llvm::Value* cmp = CompareRoot(load, Heap::kTheHoleValueRootIndex,
                                   llvm::CmpInst::ICMP_NE);
    __ CreateCondBr(cmp, merge, after_cmp);

    __ SetInsertPoint(after_cmp);
    llvm::BasicBlock* check_info = NewBlock("DoLoadKeyedFixedArray check_info");
    if (info()->IsStub()) {
      // A stub can safely convert the hole to undefined only if the array
      // protector cell contains (Smi) Isolate::kArrayProtectorValid. Otherwise
      // it needs to bail out.

      //You should be jump to check_info block
      //DeoptimizeIf(cond, check_info);
      UNIMPLEMENTED();
    } else {
      __ CreateBr(check_info);
    }
    __ SetInsertPoint(check_info);
    auto undefined = MoveHeapObject(isolate()->factory()->undefined_value());
    __ CreateBr(merge);

    __ SetInsertPoint(merge);
    llvm::PHINode* phi = __ CreatePHI(Types::tagged, 2);
    phi->addIncoming(undefined, check_info);
    phi->addIncoming(load, insert);
    instr->set_llvm_value(phi);
    return;
  }
  instr->set_llvm_value(load);
}

void LLVMChunkBuilder::DoLoadKeyedGeneric(HLoadKeyedGeneric* instr) {
  llvm::Value* obj = Use(instr->object());
  llvm::Value* context = Use(instr->context());
  llvm::Value*  key = Use(instr->key());
  
  std::vector<llvm::Value*> params;
  params.push_back(context);
  params.push_back(obj);
  params.push_back(key);
  if (instr->HasVectorAndSlot()) {
    AllowDeferredHandleDereference vector_structure_check;
    Handle<TypeFeedbackVector> handle_vector = instr->feedback_vector();
    llvm::Value* vector = MoveHeapObject(handle_vector);
    FeedbackVectorSlot feedback_slot = instr->slot();
    int index = handle_vector->GetIndex(feedback_slot);
    Smi* smi = Smi::FromInt(index);
    llvm::Value* slot = ValueFromSmi(smi);
    params.push_back(vector);
    params.push_back(slot);
  }
 
  AllowHandleAllocation allow_handles;
  Handle<Code> ic = CodeFactory::KeyedLoadICInOptimizedCode(
                        isolate(), instr->language_mode(),
                        instr->initialization_state()).code();

  llvm::Value* result = nullptr;
  if (instr->HasVectorAndSlot()) {
    result = CallCode(ic, llvm::CallingConv::X86_64_V8_S9, params);
  } else {
    result = CallCode(ic, llvm::CallingConv::X86_64_V8_S5, params);
  }
  instr->set_llvm_value(result);
}

void LLVMChunkBuilder::DoLoadNamedField(HLoadNamedField* instr) {
  HObjectAccess access = instr->access();
  int offset = access.offset();
  if (access.IsExternalMemory()) {
    UNIMPLEMENTED();
  }

  if (instr->representation().IsDouble()) {
    llvm::Value* address = FieldOperand(Use(instr->object()), offset);
    llvm::Value* cast_double = __ CreateBitCast(address, Types::ptr_float64);
    llvm::Value* result = __ CreateLoad(cast_double);
    instr->set_llvm_value(result);
    return;
  }
  llvm::Value* obj_arg = Use(instr->object());
  if (!access.IsInobject()) {
    obj_arg = LoadFieldOperand(obj_arg, JSObject::kPropertiesOffset);
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
 
  llvm::Value* obj = FieldOperand(obj_arg, offset);
  if (instr->representation().IsInteger32()) {
    llvm::Value* casted_address = __ CreateBitCast(obj, Types::ptr_i32);
    llvm::Value* res = __ CreateLoad(casted_address);
    instr->set_llvm_value(res);
  } else {
    DCHECK_EQ(GetLLVMType(instr->representation()), Types::tagged);
    llvm::Value* casted_address = __ CreateBitCast(obj, Types::ptr_tagged);
    llvm::Value* res = __ CreateLoad(casted_address);
    instr->set_llvm_value(res);
  }
}

void LLVMChunkBuilder::DoLoadNamedGeneric(HLoadNamedGeneric* instr) {
  DCHECK(instr->object()->representation().IsTagged());

  llvm::Value* obj = Use(instr->object());
  llvm::Value* context = Use(instr->context());

  Handle<Object> handle_name = instr->name();
  llvm::Value*  name = MoveHeapObject(handle_name);

  AllowDeferredHandleDereference vector_structure_check;
  Handle<TypeFeedbackVector> feedback_vector = instr->feedback_vector();
  llvm::Value* vector =  MoveHeapObject(feedback_vector);
  FeedbackVectorSlot instr_slot = instr->slot();
  int index = feedback_vector->GetIndex(instr_slot);
  Smi* smi = Smi::FromInt(index);
  llvm::Value* slot = ValueFromSmi(smi);

  AllowHandleAllocation allow_handles;
  AllowHeapAllocation allow_heap;

  Handle<Code> ic = CodeFactory::LoadICInOptimizedCode(
                        isolate(), NOT_INSIDE_TYPEOF,
                        instr->language_mode(),
                        instr->initialization_state()).code();

  // TODO(llvm): RecordSafepointWithLazyDeopt (and reloc info) + MarkAsCall

  std::vector<llvm::Value*> params;
  params.push_back(context);
  params.push_back(obj);
  params.push_back(name);
  params.push_back(vector);
  params.push_back(slot);

  auto result = CallCode(ic, llvm::CallingConv::X86_64_V8_S9, params);
  instr->set_llvm_value(result);
}

void LLVMChunkBuilder::DoLoadRoot(HLoadRoot* instr) {
  llvm::Value* load_r = LoadRoot(instr->index());
  instr->set_llvm_value(load_r);
}

void LLVMChunkBuilder::DoMapEnumLength(HMapEnumLength* instr) {
  llvm::Value* val = EnumLength(Use(instr->value()));
  llvm::Value* smi_tmp_val = __ CreateZExt(val, Types::i64);
  llvm::Value* smi_val = __ CreateShl(smi_tmp_val, kSmiShift);
  instr->set_llvm_value(smi_val);
  //UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoMathFloorOfDiv(HMathFloorOfDiv* instr) {
  UNIMPLEMENTED();
}

/*void LLVMChunkBuilder::DoMathFloor(HUnaryMathOperation* instr) {
  llvm::Value* value = Use(instr->value());
  llvm::Value* output_reg = nullptr;
  llvm::Value* output_reg_s = __ getInt32(0);
  llvm::Value* output_reg_p = nullptr;
  // Let's assume we don't support this feature
  //if (CpuFeatures::IsSupported(SSE4_1))

  llvm::BasicBlock* negative_sign = NewBlock("Negative Sign");
  llvm::BasicBlock* non_negative_sign = NewBlock("Non Negative Sign");
  llvm::BasicBlock* positive_sign = NewBlock("Positive Sign");
  llvm::BasicBlock* sign = NewBlock("Probably zero sign");
  llvm::BasicBlock* check_overflow = NewBlock("Check Overflow");
  llvm::BasicBlock* done = NewBlock("Done");
  llvm::Value* l_zero = __ getInt64(0);
  llvm::Value* l_double_zero = __ CreateSIToFP(l_zero, Types::float64);
  llvm::Value* cmp = __ CreateFCmpOLT(value, l_double_zero);
  //To do deoptimize with condition parity_even
  //DeoptimizeIf()
  __ CreateCondBr(cmp, negative_sign, non_negative_sign);
  __ SetInsertPoint(non_negative_sign);
  if (instr->CheckFlag(HValue::kBailoutOnMinusZero)) {
    llvm::Value* cmp_gr = __ CreateFCmpOGT(value, l_double_zero);
    __ CreateCondBr(cmp_gr, positive_sign, sign);
    __ SetInsertPoint(sign);
    llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
        llvm::Intrinsic::x86_sse2_movmsk_pd);
    llvm::Value* param_vect = __ CreateVectorSplat(2, value);
    llvm::Value* movms = __ CreateCall(intrinsic, param_vect);
    llvm::Value* not_zero = __ CreateICmpNE(movms, __ getInt32(0));
    DeoptimizeIf(not_zero);
    output_reg_s = __ getInt32(0);
    __ CreateBr(done);
  } else __ CreateBr(positive_sign);
  __ SetInsertPoint(positive_sign);
  llvm::Value* floor_result = __ CreateFPToSI(value, Types::i32);
  output_reg_p = floor_result;
  auto type = instr->representation().IsSmi() ? Types::i64 : Types::i32;
  llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
        llvm::Intrinsic::ssub_with_overflow, type);

  llvm::Value* params[] = { floor_result, __ getInt32(0x1) };
  llvm::Value* call = __ CreateCall(intrinsic, params);

  llvm::Value* overflow = __ CreateExtractValue(call, 1);
  DeoptimizeIf(overflow);
  __ CreateBr(done);
  __ SetInsertPoint(negative_sign);
  llvm::Value* floor_result_int = __ CreateFPToSI(value, Types::i32);
  llvm::Value* floor_result_double = __ CreateSIToFP(floor_result_int, Types::float64);
  output_reg = floor_result_int;
  llvm::Value* cmp_eq = __ CreateFCmpOEQ(value, floor_result_double);
  __ CreateCondBr(cmp_eq, done, check_overflow);
  __ SetInsertPoint(check_overflow);

  llvm::Function* intrinsic_sub_overflow = llvm::Intrinsic::getDeclaration(module_.get(),
        llvm::Intrinsic::ssub_with_overflow, Types::i32);
  llvm::Value* par[] = { floor_result_int, __ getInt32(1) };
  llvm::Value* call_intrinsic = __ CreateCall(intrinsic_sub_overflow, par);
  overflow = __ CreateExtractValue(call_intrinsic, 1);
  DeoptimizeIf(overflow);
  llvm::Value* result = output_reg;
  __ CreateBr(done);

  __ SetInsertPoint(done);
  llvm::PHINode* phi = __ CreatePHI(Types::i32, 4);
  phi->addIncoming(output_reg, negative_sign);
  phi->addIncoming(output_reg_p, positive_sign);
  phi->addIncoming(result, check_overflow);
  phi->addIncoming(output_reg_s, sign);
  instr->set_llvm_value(phi);
}
*/

void LLVMChunkBuilder::DoMathFloor(HUnaryMathOperation* instr) {
  llvm::Function* floor_intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
         llvm::Intrinsic::floor, Types::float64);
  std::vector<llvm::Value*> params;
  params.push_back(Use(instr->value()));
  llvm::Value* floor = __ CreateCall(floor_intrinsic, params);
  //llvm::Value* casted_floor = __ CreateBitCast(floor, Types::i32); 
  llvm::Value* casted_int =  __ CreateFPToSI(floor, Types::i64);
     // FIXME: Figure out why we need this step. Fix for bitops-nsieve-bits
  auto result = __ CreateTruncOrBitCast(casted_int, Types::i32);
  instr->set_llvm_value(result);
}

void LLVMChunkBuilder::DoMathMinMax(HMathMinMax* instr) {
  llvm::Value* left = Use(instr->left());
  llvm::Value* right = Use(instr->right());
  llvm::Value* left_near;
  llvm::Value* cmpl_result;
  llvm::BasicBlock* near = NewBlock("NEAR");
  llvm::BasicBlock* return_block = NewBlock("MIN MAX RETURN");
  HMathMinMax::Operation operation = instr->operation();
  llvm::BasicBlock* insert_block = __ GetInsertBlock();
  bool cond_for_min = (operation == HMathMinMax::kMathMin);

  if (instr->representation().IsSmiOrInteger32()) {
    if (instr->right()->IsConstant()) {
      DCHECK(SmiValuesAre32Bits()
        ? !instr->representation().IsSmi()
        : SmiValuesAre31Bits());
      int32_t right_value = (HConstant::cast(instr->right()))->Integer32Value();
      llvm::Value* right_imm  = __ getInt32(right_value);

      if (cond_for_min) {
        cmpl_result = __ CreateICmpSLT(left, right_imm);
      } else {
        cmpl_result = __ CreateICmpSGT(left, right_imm);
      }
      __ CreateCondBr(cmpl_result, return_block, near);
      __ SetInsertPoint(near);
      left_near = right_imm;
    } else {
      if (cond_for_min) {
        cmpl_result = __ CreateICmpSLT(left, right);
      } else {
        cmpl_result = __ CreateICmpSGT(left, right);
      }
      __ CreateCondBr(cmpl_result, return_block, near);
      __ SetInsertPoint(near);
      left_near = right;
    }
    __ CreateBr(return_block);
    __ SetInsertPoint(return_block);

    llvm::PHINode* phi = __ CreatePHI(Types::i32, 2);
    phi->addIncoming(left_near, near);
    phi->addIncoming(left, insert_block);
    instr->set_llvm_value(phi);
  } else {
    if (cond_for_min) {
      llvm::Function* fmin_intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
          llvm::Intrinsic::minnum, Types::float64);
    std::vector<llvm::Value*> params;
    params.push_back(left);
    params.push_back(right);
    llvm::Value* fmin = __ CreateCall(fmin_intrinsic, params);
    instr->set_llvm_value(fmin);
    } else {
      llvm::Function* fmax_intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
          llvm::Intrinsic::maxnum, Types::float64);
    std::vector<llvm::Value*> params;
    params.push_back(left);
    params.push_back(right);
    llvm::Value* fmax = __ CreateCall(fmax_intrinsic, params);
    instr->set_llvm_value(fmax);
    }
  }
}

void LLVMChunkBuilder::DoMod(HMod* instr) {
  if (instr->representation().IsSmiOrInteger32()) {
    if (instr->RightIsPowerOf2()) {
      DoModByPowerOf2I(instr);
    } else if (instr->right()->IsConstant()) {
      DoModByConstI(instr);
    } else {
      DoModI(instr);
    }
  } else if (instr->representation().IsDouble()) {
    llvm::Value* left = Use(instr->left());
    llvm::Value* right = Use(instr->right());
    llvm::Value* result = __ CreateFRem(left, right);
    instr->set_llvm_value(result);
  } else {
    UNIMPLEMENTED();
    //return DoArithmeticT(Token::MOD, instr);
  }
}

void LLVMChunkBuilder::DoModByConstI(HMod* instr) {
  int32_t divisor_val = (HConstant::cast(instr->right()))->Integer32Value();
  if (divisor_val == 0) {
    UNIMPLEMENTED();
  }
  auto left = Use(instr->left());
  auto right = __ getInt32(divisor_val);
  auto result = __ CreateSRem(left, right);

  // Check for negative zero.
  if (instr->CheckFlag(HValue::kBailoutOnMinusZero)) {
     llvm::BasicBlock* remainder_not_zero = NewBlock("DoModByConstI"
                                                      " remainder_not_zero");
     llvm::BasicBlock* remainder_zero = NewBlock("DoModByConstI"
                                                  " remainder_zero");
     llvm::Value* zero = __ getInt32(0);
     llvm::Value* cmp = __ CreateICmpNE(result, zero);
     __ CreateCondBr(cmp, remainder_not_zero, remainder_zero);

     __ SetInsertPoint(remainder_zero);
     llvm::Value* cmp_divident = __ CreateICmpSLT(left, zero);
     DeoptimizeIf(cmp_divident, Deoptimizer::kMinusZero, false,
                  remainder_not_zero);
     __ SetInsertPoint(remainder_not_zero);
  }
  instr->set_llvm_value(result);

  // FIXME(llvm): say no commented code!
/*  HValue* dividend = instr->left();
  llvm::Value* l_dividend = Use(dividend);
  llvm::Value* l_rax = nullptr;
  llvm::Value* l_rdx = nullptr;
  int32_t divisor = instr->right()->GetInteger32Constant();

  if (divisor == 0) {
    UNIMPLEMENTED();
  }
  //__TruncatingDiv(dividend, divisor);
  int32_t abs_div = Abs(divisor);
  base::MagicNumbersForDivision<uint32_t> mag =
      base::SignedDivisionByConstant(static_cast<uint32_t>(abs_div));
  llvm::Value* l_mag = __ getInt32(mag.multiplier);
  l_rdx = __ CreateNSWMul(l_dividend, l_mag);
  bool neg = (mag.multiplier & (static_cast<uint32_t>(1) << 31)) != 0;
  if (abs_div > 0 && neg)
    l_rdx = __ CreateNSWAdd(l_rdx, l_dividend);
  if (abs_div < 0 && !neg && mag.multiplier > 0)
    l_rdx = __ CreateNSWSub(l_rdx, l_dividend);
  if (mag.shift > 0) {
    llvm::Value* shift = __ getInt32(mag.shift);
    l_rdx = __ CreateAShr(l_rdx, shift);
  }
  llvm::Value* shift = __ getInt32(31);
  l_rax = __ CreateLShr(l_dividend, shift);
  l_rdx = __ CreateAdd(l_rdx, l_dividend);
  //over
  llvm::Value* l_abs_div = __ getInt32(abs_div);
  l_rdx = __ CreateNSWMul(l_rdx, l_abs_div);
  l_rax = l_dividend;
  l_rax = __ CreateNSWSub(l_rax, l_rdx);

  if (instr->CheckFlag(HValue::kBailoutOnMinusZero)) {
    llvm::BasicBlock* remainder_not_zero = NewBlock("Remainder not zero");
    llvm::BasicBlock* near = NewBlock("Near");

    llvm::Value* zero = __ getInt32(0);
    llvm::Value* cmp_zero = __ CreateICmpNE(l_rax, zero);
    __ CreateCondBr(cmp_zero, remainder_not_zero, near);
    __ SetInsertPoint(near);
    DeoptimizeIf(cmp_zero, instr->block());
    __ CreateBr(remainder_not_zero);
    __ SetInsertPoint(remainder_not_zero);
  }
  instr->set_llvm_value(l_rax);*/
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

void LLVMChunkBuilder::DoModI(HMod* instr) {
  llvm::BasicBlock* insert = __ GetInsertBlock();
  //TODO: not tested, test case string-unpack-code.js in e finction
  llvm::Value* left = Use(instr->left());
  llvm::Value* right = Use(instr->right());
  llvm::Value* zero = __ getInt32(0);
  llvm::BasicBlock* done = NewBlock("DoModI done");
  llvm::Value* result = nullptr;
  llvm::Value* div_res = nullptr;
  if (instr->CheckFlag(HValue::kCanBeDivByZero)) {
    llvm::Value* is_zero  = __ CreateICmpEQ(right, zero);
    DeoptimizeIf(is_zero, Deoptimizer::kDivisionByZero);
  }

  int phi_in = 1;
  llvm::BasicBlock* after_cmp_one = nullptr;
  if (instr->CheckFlag(HValue::kCanOverflow)) {
    UNIMPLEMENTED(); // because not tested
    after_cmp_one = NewBlock("DoModI after cmpare minus one");
    llvm::BasicBlock* after_cmp_minInt = NewBlock("DoModI after cmpare MinInt");
    llvm::BasicBlock* no_overflow_possible = NewBlock("DoModI"
                                                      "no_overflow_possible");
    llvm::Value* min_int = __ getInt32(kMinInt);
    llvm::Value* cmp_min = __ CreateICmpEQ(left, min_int);
    __ CreateCondBr(cmp_min, no_overflow_possible, after_cmp_minInt);

    __ SetInsertPoint(after_cmp_minInt);
    llvm::Value* minus_one = __ getInt32(-1);
    llvm::Value* is_not_minus_one = __ CreateICmpNE(right, minus_one);
    if (instr->CheckFlag(HValue::kBailoutOnMinusZero)) {
      bool negate = true;
      DeoptimizeIf(is_not_minus_one, Deoptimizer::kMinusZero, negate);
    } else {
      phi_in++;
      __ CreateCondBr(is_not_minus_one, no_overflow_possible, after_cmp_one);
      __ SetInsertPoint(after_cmp_one);
      result = zero;
        __ CreateBr(done);
    }
    __ SetInsertPoint(no_overflow_possible);
    }

   llvm::BasicBlock* negative = nullptr;
   llvm::BasicBlock* positive = nullptr;

 if (instr->CheckFlag(HValue::kBailoutOnMinusZero)) {
    phi_in++;
    negative = NewBlock("DoModI left is negative");
    positive = NewBlock("DoModI left is positive");
    llvm::Value* cmp_sign = __ CreateICmpSGT(left, zero);
    __ CreateCondBr(cmp_sign, positive, negative);

    __ SetInsertPoint(negative);
    div_res = __ CreateSRem(left, right);
    llvm::Value* cmp_zero = __ CreateICmpEQ(div_res, zero);
    DeoptimizeIf(cmp_zero, Deoptimizer::kMinusZero);
    __ CreateBr(done);

    __ SetInsertPoint(positive);
  }

  llvm::Value* div = __ CreateSRem(left, right);
  __ CreateBr(done);

  __ SetInsertPoint(done);
  llvm::PHINode* phi = __ CreatePHI(Types::i32, phi_in);
  if (instr->CheckFlag(HValue::kBailoutOnMinusZero)) {
    phi->addIncoming(div_res, negative);
    phi->addIncoming(div, positive);
  } else {
    phi->addIncoming(div, insert);
  }
  if (instr->CheckFlag(HValue::kCanOverflow) &&
      !instr->CheckFlag(HValue::kBailoutOnMinusZero)) {
    phi->addIncoming(result, after_cmp_one);
  }
  instr->set_llvm_value(phi);
}

void LLVMChunkBuilder::DoMul(HMul* instr) {
  if(instr->representation().IsSmiOrInteger32()) {
    DCHECK(instr->left()->representation().Equals(instr->representation()));
    DCHECK(instr->right()->representation().Equals(instr->representation()));
    HValue* left = instr->left();
    HValue* right = instr->right();
    llvm::Value* llvm_left = Use(left);
    llvm::Value* llvm_right = Use(right);
    bool can_overflow = instr->CheckFlag(HValue::kCanOverflow);
    // TODO(llvm): use raw mul, not the intrinsic, if (!can_overflow).
    llvm::Value* overflow = nullptr;
    if (instr->representation().IsSmi()) {
      // FIXME (llvm):
      // 1) Minus Zero?? Important
      // 2) see if we can refactor using SmiToInteger32() or the like
      auto type = Types::i64;
      llvm::Value* shift = __ CreateAShr(llvm_left, 32);
      llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
                                     llvm::Intrinsic::smul_with_overflow, type);

      llvm::Value* params[] = { shift, llvm_right };
      llvm::Value* call = __ CreateCall(intrinsic, params);

      llvm::Value* mul = __ CreateExtractValue(call, 0);
      overflow = __ CreateExtractValue(call, 1);
      instr->set_llvm_value(mul);
    } else {
      auto type = Types::i32;
      llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
                                     llvm::Intrinsic::smul_with_overflow, type);

      llvm::Value* params[] = { llvm_left, llvm_right };
      llvm::Value* call = __ CreateCall(intrinsic, params);

      llvm::Value* mul = __ CreateExtractValue(call, 0);
      overflow = __ CreateExtractValue(call, 1);
      instr->set_llvm_value(mul);
    }
    if (can_overflow) DeoptimizeIf(overflow, Deoptimizer::kOverflow);
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
  int arg_count = graph()->osr()->UnoptimizedFrameSlots();
  std::string arg_offset = std::to_string(arg_count * kPointerSize);
  std::string asm_string1 = "add $$";
  std::string asm_string2 = ", %rsp";
  std::string final_strig = asm_string1 + arg_offset + asm_string2;
  auto inl_asm_f_type = llvm::FunctionType::get(__ getVoidTy(), false);
  llvm::InlineAsm* inline_asm = llvm::InlineAsm::get(
      inl_asm_f_type, final_strig, "~{dirflag},~{fpsr},~{flags}", true);
  __ CreateCall(inline_asm);

  //UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoPower(HPower* instr) {
  Representation exponent_type = instr->right()->representation();
  
  if (exponent_type.IsSmi()) {
    UNIMPLEMENTED();
  } else if (exponent_type.IsTagged()) {
    llvm::Value* tagged_exponent = Use(instr->right());
    llvm::Value* is_smi = SmiCheck(tagged_exponent, false);
    llvm::BasicBlock* deopt = NewBlock("DoPower CheckObjType");
    llvm::BasicBlock* no_deopt = NewBlock("DoPower No Deoptimize");
    __ CreateCondBr(is_smi, no_deopt, deopt);
    __ SetInsertPoint(deopt);
    llvm::Value* cmp = CmpObjectType(tagged_exponent,
                                     HEAP_NUMBER_TYPE, llvm::CmpInst::ICMP_NE);
    DeoptimizeIf(cmp, Deoptimizer::kNotAHeapNumber, false, no_deopt);

    __ SetInsertPoint(no_deopt);
    MathPowStub stub(isolate(), MathPowStub::TAGGED);
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
    llvm::Value* call = CallAddress(code->entry(),
                                    llvm::CallingConv::X86_64_V8_S2,
                                    params, Types::float64);
    instr->set_llvm_value(call);
  
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
    llvm::Value* call = CallAddress(code->entry(),
                                    llvm::CallingConv::X86_64_V8_S2,
                                    params, Types::float64);
    instr->set_llvm_value(call);
  } else {
    //UNIMPLEMENTED();
    MathPowStub stub(isolate(), MathPowStub::DOUBLE);
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
    llvm::Value* call = CallAddress(code->entry(),
                                    llvm::CallingConv::X86_64_V8_S2,
                                    params, Types::float64);
    instr->set_llvm_value(call);
  }
}

llvm::Value* LLVMChunkBuilder::RegExpLiteralSlow(HValue* instr,
                                                 llvm::Value* phi) {
  int size = JSRegExp::kSize + JSRegExp::kInObjectFieldCount * kPointerSize;

  Smi* smi_size = Smi::FromInt(size);
  DCHECK(pending_pushed_args_.is_empty());
  pending_pushed_args_.Add(ValueFromSmi(smi_size), info()->zone());
  pending_pushed_args_.Add(phi, info()->zone());
  llvm::Value* call_result = CallRuntimeViaId(Runtime::kAllocateInNewSpace);
  pending_pushed_args_.Clear();
  return call_result;
}

void LLVMChunkBuilder::DoRegExpLiteral(HRegExpLiteral* instr) {
  //TODO: not tested string-validate-input.js in doTest
  llvm::BasicBlock* materialized = NewBlock("DoRegExpLiteral materialized");
  llvm::BasicBlock* near = NewBlock("DoRegExpLiteral near");
  llvm::BasicBlock* input = __ GetInsertBlock();
  int literal_offset =
      FixedArray::OffsetOfElementAt(instr->literal_index());
  llvm::Value* literals = MoveHeapObject(instr->literals());
  llvm::Value* fild_literal = LoadFieldOperand(literals, literal_offset);
  auto cmp_root = CompareRoot(fild_literal, Heap::kUndefinedValueRootIndex,
                              llvm::CmpInst::ICMP_NE);
  __ CreateCondBr(cmp_root, materialized, near);
  __ SetInsertPoint(near);
  DCHECK(pending_pushed_args_.is_empty());
  Smi* index = Smi::FromInt(literal_offset);
  pending_pushed_args_.Add(literals, info()->zone());
  pending_pushed_args_.Add(ValueFromSmi(index), info()->zone());
  pending_pushed_args_.Add(MoveHeapObject(instr->pattern()), info()->zone());
  pending_pushed_args_.Add(MoveHeapObject(instr->flags()), info()->zone());
  llvm::Value* call_result = CallRuntimeViaId(Runtime::kMaterializeRegExpLiteral);
  pending_pushed_args_.Clear();
  __ CreateBr(materialized);

  __ SetInsertPoint(materialized);
  int size = JSRegExp::kSize + JSRegExp::kInObjectFieldCount * kPointerSize;
  llvm::Value* l_size = __ getInt32(size);
  //TODO(llvm) impement Allocate(size, rax, rcx, rdx, &runtime_allocate, TAG_OBJECT);
  //                    jmp(&allocated, Label::kNear);
  llvm::PHINode* phi = __ CreatePHI(Types::tagged, 2);
  phi->addIncoming(call_result, near);
  phi->addIncoming(fild_literal, input);
  llvm::Value* (LLVMChunkBuilder::*fptr)(HValue*, llvm::Value*);
  fptr = &LLVMChunkBuilder::RegExpLiteralSlow;
  llvm::Value* value = Allocate(l_size, fptr, TAG_OBJECT, nullptr, phi);

  llvm::Value* temp = nullptr;    //rdx
  llvm::Value* temp2 = nullptr;     //rcx
  for (int i = 0; i < size - kPointerSize; i += 2 * kPointerSize) {
    temp = LoadFieldOperand(value, i);
    temp2 = LoadFieldOperand(value, i + kPointerSize);
    llvm::Value* ptr = __ CreateIntToPtr(phi, Types::ptr_i8);
    llvm::Value* address =  __ CreateGEP(ptr, __ getInt32(i));
    address = __ CreateBitCast(address, Types::ptr_tagged);
    __ CreateStore(temp, address);
    llvm::Value* address2 =  __ CreateGEP(ptr, __ getInt32(i + kPointerSize));
    address2 = __ CreateBitCast(address2, Types::ptr_tagged);
    __ CreateStore(temp2, address2);
  }
  if ((size % (2 * kPointerSize)) != 0) {
   temp = LoadFieldOperand(value, size - kPointerSize);  // rdx
   llvm::Value* ptr = __ CreateIntToPtr(phi, Types::ptr_i8);
   llvm::Value* address =  __ CreateGEP(ptr, __ getInt32(size - kPointerSize));
   llvm::Value* casted_address = __ CreateBitCast(address, Types::ptr_tagged);
   __ CreateStore(temp, casted_address);
  }
  instr->set_llvm_value(value);
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
    llvm::Value* AShr = __ CreateAShr(Use(left), Use(right), "Sar");
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
  //TODO: not tested
  llvm::Value* context = Use(instr->context());
  llvm::Value* value = Use(instr->value());
  int offset = Context::SlotOffset(instr->slot_index()); 

  if (instr->RequiresHoleCheck()) {
    UNIMPLEMENTED();
  }

  llvm::Value* target = ConstructAddress(context, offset);
  llvm::Value* casted_address = nullptr;
  if (instr->value()->representation().IsTagged())
    casted_address = __ CreateBitCast(target, Types::ptr_tagged);
  else
    casted_address = __ CreateBitCast(target, Types::ptr_i64);
  __ CreateStore(value, casted_address);
  if (instr->NeedsWriteBarrier()) {
    int slot_offset = Context::SlotOffset(instr->slot_index());
    enum SmiCheck check_needed = instr->value()->type().IsHeapObject()
          ? OMIT_SMI_CHECK : INLINE_SMI_CHECK;
    RecordWriteField(context,
                     value,
                     slot_offset + kHeapObjectTag,
                     check_needed,
                     kPointersToHereMaybeInteresting,
                     EMIT_REMEMBERED_SET);
  }
}

void LLVMChunkBuilder::DoStoreFrameContext(HStoreFrameContext* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoStoreKeyed(HStoreKeyed* instr) {
  if (instr->is_fixed_typed_array()) {
    DoStoreKeyedExternalArray(instr);
  } else if (instr->value()->representation().IsDouble()) {
    DoStoreKeyedFixedDoubleArray(instr);
  } else {
    DoStoreKeyedFixedArray(instr);
  }
}

void LLVMChunkBuilder::DoStoreKeyedExternalArray(HStoreKeyed* instr) {
  //UNIMPLEMENTED();
  //TODO: not tested string-validate-input.js in doTest
  ElementsKind elements_kind = instr->elements_kind();
  uint32_t offset = instr->base_offset();
  llvm::Value* casted_address = nullptr;
  llvm::Value* store = nullptr;
  HValue* key = instr->key();

  if (kPointerSize == kInt32Size && !key->IsConstant()) {
    Representation key_representation =
        instr->key()->representation();
    if (ExternalArrayOpRequiresTemp(key_representation, elements_kind)) {
      UNIMPLEMENTED();
    } else if (instr->IsDehoisted()) {
      UNIMPLEMENTED();
    }
  }
  llvm::Value* address = BuildFastArrayOperand(key, Use(instr->elements()),
                                               elements_kind, offset);
  if (elements_kind == FLOAT32_ELEMENTS) {
    casted_address = __ CreateBitCast(address, Types::ptr_float32);
    auto result = __ CreateFPTrunc(Use(instr->value()), Types::float32);
    store = __ CreateStore(result, casted_address);
    instr->set_llvm_value(store);
    //UNIMPLEMENTED();
  } else if (elements_kind == FLOAT64_ELEMENTS) {
    UNIMPLEMENTED();
  } else {
    switch (elements_kind) {
      case INT8_ELEMENTS:
      case UINT8_ELEMENTS:
      case UINT8_CLAMPED_ELEMENTS: {
        casted_address = __ CreateBitCast(address, Types::ptr_i8);
        auto result = __ CreateTruncOrBitCast(Use(instr->value()), Types::i8);
        store = __ CreateStore(result, casted_address);
        instr->set_llvm_value(store);
        break;
      }
      case INT16_ELEMENTS:
      case UINT16_ELEMENTS:
        UNIMPLEMENTED();
        break;
      case INT32_ELEMENTS:
      case UINT32_ELEMENTS:
        casted_address = __ CreateBitCast(address, Types::ptr_i32);
        store = __ CreateStore(Use(instr->value()), casted_address);
        instr->set_llvm_value(store);
        break;
      case FLOAT32_ELEMENTS:
      case FLOAT64_ELEMENTS:
      case FAST_ELEMENTS:
      case FAST_SMI_ELEMENTS:
      case FAST_DOUBLE_ELEMENTS:
      case FAST_HOLEY_ELEMENTS:
      case FAST_HOLEY_SMI_ELEMENTS:
      case FAST_HOLEY_DOUBLE_ELEMENTS:
      case DICTIONARY_ELEMENTS:
      case FAST_SLOPPY_ARGUMENTS_ELEMENTS:
      case SLOW_SLOPPY_ARGUMENTS_ELEMENTS:
        UNREACHABLE();
        break;
    }
  }
}

void LLVMChunkBuilder::DoStoreKeyedFixedDoubleArray(HStoreKeyed* instr) {
  HValue* key = instr->key();
  llvm::Value* value = Use(instr->value());
  uint32_t inst_offset = instr->base_offset();
  ElementsKind elements_kind = instr->elements_kind();
  if (kPointerSize == kInt32Size && !key->IsConstant()
      && instr->IsDehoisted()) {
    UNIMPLEMENTED();
  }
  llvm::Value* canonical_value = value;
  if (instr->NeedsCanonicalization()) {
    UNIMPLEMENTED();
    llvm::Function* canonicalize = llvm::Intrinsic::getDeclaration(module_.get(),
          llvm::Intrinsic::canonicalize, Types::float64);
    llvm::Value* params[] = { value };
    canonical_value = __ CreateCall(canonicalize, params);

  }
  llvm::Value* address = BuildFastArrayOperand(key, Use(instr->elements()),
                                               elements_kind, inst_offset);
  llvm::Value* casted_address = __ CreateBitCast(address, Types::ptr_float64);
  llvm::Value* Store = __ CreateStore(canonical_value, casted_address);
  instr->set_llvm_value(Store);
}

void LLVMChunkBuilder::DoStoreKeyedFixedArray(HStoreKeyed* instr) {
  HValue* key = instr->key();
  Representation representation = instr->value()->representation();
  ElementsKind elements_kind = instr->elements_kind();
  uint32_t inst_offset = instr->base_offset();
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
  llvm::Value* address = BuildFastArrayOperand(key, Use(instr->elements()),
                                               elements_kind, inst_offset); 
  HValue* hValue = instr->value();
  llvm::Value* store = nullptr;
  llvm::Value* casted_address = nullptr;

  if (!hValue->IsConstant() || hValue->representation().IsSmi() ||
      hValue->representation().IsInteger32()) {
    auto pointer_type = GetLLVMType(hValue->representation())->getPointerTo();
    casted_address = __ CreateBitOrPointerCast(address, pointer_type);
    store = __ CreateStore(Use(hValue), casted_address);
  } else {
    DCHECK(hValue->IsConstant());
    HConstant* constant = HConstant::cast(instr->value());
    Handle<Object> handle_value = constant->handle(isolate());
    casted_address = __ CreateBitOrPointerCast(address, Types::ptr_tagged);
    auto llvm_val = MoveHeapObject(handle_value);
    store = __ CreateStore(llvm_val, casted_address);
  } 
  instr->set_llvm_value(store);
  if (instr->NeedsWriteBarrier()) {
    llvm::Value* elements = Use(instr->elements());
    llvm::Value* value = Use(instr->value());
    enum SmiCheck check_needed = instr->value()->type().IsHeapObject()
            ? OMIT_SMI_CHECK : INLINE_SMI_CHECK;
    // FIXME(llvm): kSaveFPRegs
    RecordWrite(elements,
                casted_address,
                value,
                instr->PointersToHereCheckForValue(),
                EMIT_REMEMBERED_SET,
                check_needed);
  }
}

void LLVMChunkBuilder::RecordWriteField(llvm::Value* object,
                                        llvm::Value* value,
                                        int offset,
                                        enum SmiCheck smi_check,
                                        PointersToHereCheck ptr_check,
                                        RememberedSetAction remembered_set) {
  //FIXME: Not sure this is right
  //TODO: Find a way to test this function
  llvm::BasicBlock* done = NewBlock("RecordWriteField done");
  if (smi_check == INLINE_SMI_CHECK) {
    llvm::BasicBlock* current_block = NewBlock("RecordWriteField Smi checked");
    // Skip barrier if writing a smi.
    llvm::Value* smi_cond = SmiCheck(value, false);//JumpIfSmi(value, &done);
    __ CreateCondBr(smi_cond, done, current_block);
    __ SetInsertPoint(current_block);
  }

  DCHECK(IsAligned(offset, kPointerSize));
  auto map_address = FieldOperand(object, offset);
  map_address = __ CreateBitOrPointerCast(map_address, Types::tagged);

  if (emit_debug_code()) {
    UNIMPLEMENTED();
  }

  RecordWrite(object, map_address, value, ptr_check, remembered_set,
              OMIT_SMI_CHECK);
  __ CreateBr(done);
  __ SetInsertPoint(done);

  if (emit_debug_code()) {
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::RecordWrite(llvm::Value* object,
                                   llvm::Value* address,
                                   llvm::Value* value,
                                   PointersToHereCheck ptr_check,
                                   RememberedSetAction remembered_set_action,
                                   enum SmiCheck smi_check) {
  AssertNotSmi(object);

  if (remembered_set_action == OMIT_REMEMBERED_SET &&
      !FLAG_incremental_marking) {
    return;
  }

  if (emit_debug_code()) {
    // WRONG: LoadFieldOperand (FieldOperand) subtracts kHeapObject tag,
    // Operand does not
//    Assert(Compare(value, LoadFieldOperand(key_reg, 0)));
    UNIMPLEMENTED();
  }
  auto stub_block = NewBlock("RecordWrite after checked page flag");
  llvm::BasicBlock* done = NewBlock("RecordWrite done");

  if (smi_check == INLINE_SMI_CHECK) {
    llvm::BasicBlock* current_block = NewBlock("RecordWrite Smi checked");
    // Skip barrier if writing a smi.
    llvm::Value* smi_cond = SmiCheck(value, false);//JumpIfSmi(value, &done);
    __ CreateCondBr(smi_cond, done, current_block);
    __ SetInsertPoint(current_block);
  }

  if (ptr_check != kPointersToHereAreAlwaysInteresting) {
    auto equal = CheckPageFlag(value,
                               MemoryChunk::kPointersToHereAreInterestingMask);
    llvm::BasicBlock* after_page_check = NewBlock("RecordWrite page check");
    __ CreateCondBr(equal, done, after_page_check);
    __ SetInsertPoint(after_page_check);
  }

  auto equal = CheckPageFlag(object,
                             MemoryChunk::kPointersFromHereAreInterestingMask);
  __ CreateCondBr(equal, done, stub_block);

  __ SetInsertPoint(stub_block);
  Register object_reg = rbx;
  Register map_reg = rcx;
  Register dst_reg = rdx;
  RecordWriteStub stub(isolate(), object_reg, map_reg, dst_reg,
                       remembered_set_action, kSaveFPRegs);
  Handle<Code> code = Handle<Code>::null();
  {
    AllowHandleAllocation allow_handles;
    AllowHeapAllocation allow_heap_alloc;
    code = stub.GetCode();
    // FIXME(llvm,gc): respect reloc info mode...
  }
  std::vector<llvm::Value*> params = { object, value, address };
  CallCode(code, llvm::CallingConv::X86_64_V8_RWS, params, true);
  __ CreateBr(done);

  __ SetInsertPoint(done);

  // Count number of write barriers in generated code.
  isolate()->counters()->write_barriers_static()->Increment();
  IncrementCounter(isolate()->counters()->write_barriers_dynamic(), 1);
}

void LLVMChunkBuilder::DoStoreKeyedGeneric(HStoreKeyedGeneric* instr) {
  DCHECK(instr->object()->representation().IsTagged());
  DCHECK(instr->key()->representation().IsTagged());
  DCHECK(instr->value()->representation().IsTagged());
  if (instr->HasVectorAndSlot()) {
    UNIMPLEMENTED();
  }

  Handle<Code> ic = CodeFactory::KeyedStoreICInOptimizedCode(
                        isolate(), instr->language_mode(),
                        instr->initialization_state()).code();


  std::vector<llvm::Value*> params;
  params.push_back(Use(instr->context()));
  params.push_back(Use(instr->object()));
  params.push_back(Use(instr->value()));
  params.push_back(Use(instr->key()));
  auto result = CallCode(ic, llvm::CallingConv::X86_64_V8_S7, params);
  instr->set_llvm_value(result);
}

void LLVMChunkBuilder::DoStoreNamedField(HStoreNamedField* instr) {
  Representation representation = instr->field_representation();

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
    Handle<Map> transition = instr->transition_map();
    AddDeprecationDependency(transition);
    if (!instr->NeedsWriteBarrierForMap()) {
      llvm::Value* heap_transition = MoveHeapObject(transition);
      llvm::Value* ptr = __ CreateIntToPtr(Use(instr->object()), Types::ptr_i8);
      llvm::Value* address = FieldOperand(ptr, HeapObject::kMapOffset);
      llvm::Value* casted_address = __ CreateBitCast(address, Types::ptr_tagged);
      __ CreateStore(heap_transition, casted_address);
    } else {
      llvm::Value* scratch = MoveHeapObject(transition);
      llvm::Value* obj_addr = FieldOperand(Use(instr->object()),
                                           HeapObject::kMapOffset);
      auto casted_address = __ CreateBitCast(obj_addr, Types::ptr_tagged);
      __ CreateStore(scratch, casted_address);
      RecordWriteForMap(Use(instr->object()), scratch);
    }
  }

  // Do the store.
  llvm::Value* obj_arg = Use(instr->object());
  if (!access.IsInobject()) {
    obj_arg = LoadFieldOperand(obj_arg, JSObject::kPropertiesOffset);
  }

  if (representation.IsSmi() && SmiValuesAre32Bits() &&
      instr->value()->representation().IsInteger32()) {
    DCHECK(instr->store_mode() == STORE_TO_INITIALIZED_ENTRY);
    if (FLAG_debug_code) {
      UNIMPLEMENTED();
    }
    // Store int value directly to upper half of the smi.
    STATIC_ASSERT(kSmiTag == 0);
    DCHECK(kSmiTagSize + kSmiShiftSize == 32);
    offset += kPointerSize / 2;
    representation = Representation::Integer32();
  }

  //Operand operand = FieldOperand(write_register, offset);

  if (FLAG_unbox_double_fields && representation.IsDouble()) {
    UNIMPLEMENTED();
    DCHECK(access.IsInobject());
    llvm::Value* obj_address = ConstructAddress(Use(instr->object()), offset);
    llvm::Value* casted_obj_add =  __ CreateBitCast(obj_address,
                                                    Types::ptr_float64);
    llvm::Value* value = Use(instr->value());
    __ CreateStore(value, casted_obj_add);
    return;
  } else {
    HValue* hValue = instr->value();
    if (hValue->representation().IsInteger32()) {
      llvm::Value* store_address = ConstructAddress(obj_arg, offset);
      llvm::Value* casted_adderss = __ CreateBitCast(store_address,
                                                     Types::ptr_i32);
      llvm::Value* casted_value = __ CreateBitCast(Use(hValue), Types::i32);
      __ CreateStore(casted_value, casted_adderss);
    } else if (hValue->representation().IsSmi() || !hValue->IsConstant()){
      llvm::Value* store_address = ConstructAddress(obj_arg, offset);
      auto pointer_type = GetLLVMType(hValue->representation())->getPointerTo();
      llvm::Value* casted_adderss = __ CreateBitCast(store_address,
                                                     pointer_type);
      __ CreateStore(Use(hValue), casted_adderss);
    } else {
      DCHECK(hValue->IsConstant());
      {
        AllowHandleAllocation allow_handles; //TODO: Why we need this?
        HConstant* constant = HConstant::cast(instr->value());
        Handle<Object> handle_value = constant->handle(isolate());
        llvm::Value* store_address = ConstructAddress(obj_arg,
                                                      offset);
        llvm::Value* casted_adderss = __ CreateBitCast(store_address,
                                                       Types::ptr_tagged);
        auto llvm_val = MoveHeapObject(handle_value);
        __ CreateStore(llvm_val, casted_adderss);
      }

    }
  }

  if (instr->NeedsWriteBarrier()) {
    //FIXME: Not sure this is right
    //TODO: Find a way to test this case
    RecordWriteField(obj_arg,
                     Use(instr->value()),
                     offset + 1,
                     instr->SmiCheckForWriteBarrier(),
                     instr->PointersToHereCheckForValue(),
                     EMIT_REMEMBERED_SET);
  }
}

void LLVMChunkBuilder::DoStoreNamedGeneric(HStoreNamedGeneric* instr) {
  llvm::Value* context = Use(instr->context());
  llvm::Value* object = Use(instr->object());
  llvm::Value* value = Use(instr->value());
  llvm::Value* name_reg = MoveHeapObject(instr->name());
  AllowHandleAllocation allow_handles_allocation;
  Handle<Code> ic =
      StoreIC::initialize_stub(isolate(), instr->language_mode(),
                               instr->initialization_state());
  std::vector<llvm::Value*> params;
  params.push_back(context);
  params.push_back(object);
  params.push_back(value);
  params.push_back(name_reg);
  for (int i = pending_pushed_args_.length() - 1; i >= 0; i--)
    params.push_back(pending_pushed_args_[i]);
  pending_pushed_args_.Clear();
  llvm::Value* call = CallCode(ic, llvm::CallingConv::X86_64_V8_S7, params);
  llvm::Value* return_val = __ CreatePtrToInt(call,Types::i64);
  instr->set_llvm_value(return_val);
}

void LLVMChunkBuilder::DoStringAdd(HStringAdd* instr) {
  StringAddStub stub(isolate(),
                     instr->flags(),
                     instr->pretenure_flag());
  
  Handle<Code> code = Handle<Code>::null();
  {
    AllowHandleAllocation allow_handles;
    AllowHeapAllocation allow_heap;
    code = stub.GetCode();
    // FIXME(llvm,gc): respect reloc info mode...
  }
  std::vector<llvm::Value*> params;
  params.push_back(Use(instr->context()));
  for (int i = 1; i < instr->OperandCount() ; ++i)
    params.push_back(Use(instr->OperandAt(i)));
  llvm::Value* call = CallCode(code, llvm::CallingConv::X86_64_V8_S10, params);
  instr->set_llvm_value(call);
}

// TODO(llvm): this is a horrible function.
// At least do something about this mess with types.
// And test it thoroughly...
void LLVMChunkBuilder::DoStringCharCodeAt(HStringCharCodeAt* instr) {
  //Only one path is tested
  //IsSeqString->OneByte->Done.
  //TODO: Find scripts to tests other paths
  llvm::BasicBlock* insert = __ GetInsertBlock();
  llvm::Value* str = Use(instr->string());
  llvm::Value* index = Use(instr->index());
  llvm::BasicBlock* deferred = NewBlock("StringCharCodeAt Deferred");
  llvm::BasicBlock* set_value = NewBlock("StringCharCodeAt End");
  llvm::Value* map_offset = LoadFieldOperand(str, HeapObject::kMapOffset);
  llvm::Value* instance_type = LoadFieldOperand(map_offset,
                                                Map::kInstanceTypeOffset);
  instance_type = __ CreateBitOrPointerCast(instance_type, Types::i64);
  //movzxbl
  llvm::Value* result_type = __ CreateAnd(instance_type,
                                          __ getInt64(0x000000ff));
  llvm::BasicBlock* check_sequental = NewBlock("StringCharCodeAt"
                                                "CheckSequental");
  llvm::BasicBlock* check_seq_cont = NewBlock("StringCharCodeAt"
                                                "CheckSequental Cont");
  llvm::Value* and_IndirectStringMask = __ CreateAnd(result_type,
                                            __ getInt64(kIsIndirectStringMask));
  llvm::Value* cmp_IndirectStringMask = __ CreateICmpEQ(and_IndirectStringMask,
                                                        __ getInt64(0));
  __ CreateCondBr(cmp_IndirectStringMask, check_sequental, check_seq_cont);

  __ SetInsertPoint(check_seq_cont);
  llvm::BasicBlock* cons_str = NewBlock("StringCharCodeAt IsConsString");
  llvm::BasicBlock* cons_str_cont =  NewBlock("StringCharCodeAt NotConsString");
  llvm::Value* and_NotConsMask = __ CreateAnd(result_type,
                                             __ getInt64(kSlicedNotConsMask));
  llvm::Value* cmp_NotConsMask = __ CreateICmpEQ(and_NotConsMask, __ getInt64(0));
  __ CreateCondBr(cmp_NotConsMask, cons_str, cons_str_cont);

  __ SetInsertPoint(cons_str_cont);
  llvm::BasicBlock* indirect_string_loaded = NewBlock("StringCharCodeAt Indirect String");
  llvm::Value* load = LoadFieldOperand(str, SlicedString::kOffsetOffset);
  llvm::Value* address = FieldOperand(load, kSmiShift / kBitsPerByte);
  llvm::Value* casted_address = __ CreatePointerCast(address,
                                                     Types::ptr_i32);
//  llvm::Value* ptr_string = __ CreateIntToPtr(str, Types::ptr_i8);
//  llvm::Value* gep_string = __ CreateGEP(ptr_string,
//                                        __ getInt64(kSmiShift / kBitsPerByte));
  // TODO(Jivan) //Do wee need ptr_i32 here?
//  llvm::Value* casted_cons = __ CreateBitCast(gep_string, Types::ptr_i32);
//  llvm::Value* cons_load = __ CreateLoad(casted_cons);
  llvm::Value* cons_load = __ CreateLoad(casted_address);
  llvm::Value* cons_index = __ CreateAdd(index, cons_load);
  llvm::Value* cons_string = LoadFieldOperand(str, SlicedString::kParentOffset);
  __ CreateBr(indirect_string_loaded);

  __ SetInsertPoint(cons_str);
  llvm::BasicBlock* cmp_root_cont = NewBlock("StringCharCodeAt"
                                              "ConsStr CompareRoot Cont");
  llvm::Value* string_second_offset = LoadFieldOperand(str,
                                                       ConsString::kSecondOffset);
  llvm::Value* cmp_root = CompareRoot(string_second_offset,
                                      Heap::kempty_stringRootIndex);
  __ CreateCondBr(cmp_root, cmp_root_cont, deferred);

  __ SetInsertPoint(cmp_root_cont);
  llvm::Value* after_cmp_root_str = LoadFieldOperand(str,
                                                     ConsString::kFirstOffset);
  __ CreateBr(indirect_string_loaded);

  __ SetInsertPoint(indirect_string_loaded);
  llvm::PHINode* phi_string = __ CreatePHI(Types::tagged, 2);
  phi_string->addIncoming(after_cmp_root_str, cmp_root_cont);
  phi_string->addIncoming(cons_string, cons_str_cont);

  llvm::PHINode* index_indirect = __ CreatePHI(Types::i32, 2);
  index_indirect->addIncoming(cons_index, cons_str_cont);
  index_indirect->addIncoming(index, insert);

  llvm::Value* indirect_map = LoadFieldOperand(phi_string,
                                               HeapObject::kMapOffset);
  llvm::Value* indirect_instance = LoadFieldOperand(indirect_map,
                                                    Map::kInstanceTypeOffset);
  indirect_instance = __ CreateBitOrPointerCast(indirect_instance, Types::i64);
  llvm::Value* indirect_result_type = __ CreateAnd(indirect_instance,
                                                  __ getInt64(0x000000ff));
  __ CreateBr(check_sequental);

  __ SetInsertPoint(check_sequental);
  STATIC_ASSERT(kSeqStringTag == 0);
  llvm::BasicBlock* seq_string = NewBlock("StringCharCodeAt SeqString");
  llvm::BasicBlock* cont_inside_seq = NewBlock("StringCharCodeAt SeqString cont");
  llvm::PHINode* phi_result_type = __ CreatePHI(Types::i64, 2);
  phi_result_type->addIncoming(indirect_result_type, indirect_string_loaded);
  phi_result_type->addIncoming(result_type, insert);

  llvm::PHINode* phi_index = __ CreatePHI(Types::i32, 2);
  phi_index->addIncoming(index_indirect, indirect_string_loaded);
  phi_index->addIncoming(index, insert);

  llvm::PHINode* phi_str = __ CreatePHI(Types::tagged, 2);
  phi_str->addIncoming(str, insert);
  phi_str->addIncoming(phi_string, indirect_string_loaded);

  llvm::Value* and_representation =  __ CreateAnd(phi_result_type,
                                        __ getInt64(kStringRepresentationMask));
  llvm::Value* cmp_representation = __ CreateICmpEQ(and_representation,
                                                        __ getInt64(0));
   __ CreateCondBr(cmp_representation, seq_string, cont_inside_seq);

  __ SetInsertPoint(cont_inside_seq);
  llvm::BasicBlock* extern_string = NewBlock("StringCharCodeAt"
                                             "CheckShortExternelString");
  if (FLAG_debug_code) {
    UNIMPLEMENTED();
  }
  STATIC_ASSERT(kShortExternalStringTag != 0);
  llvm::Value* and_short_tag = __ CreateAnd(phi_result_type,
                                          __ getInt64(kShortExternalStringTag));
  llvm::Value* cmp_short_tag = __ CreateICmpNE(and_short_tag,
                                                             __ getInt64(0));
  __ CreateCondBr(cmp_short_tag, deferred, extern_string);

  __ SetInsertPoint(extern_string);
  STATIC_ASSERT(kTwoByteStringTag == 0);
  llvm::BasicBlock* one_byte_external = NewBlock("StringCharCodeAt"
                                                 "OneByteExternal");
  llvm::BasicBlock* two_byte_external = NewBlock("StringCharCodeAt"
                                                 "OneByteExternal Cont");

  llvm::Value* and_encoding_mask = __ CreateAnd(phi_result_type,
                                              __ getInt64(kStringEncodingMask));
  llvm::Value* not_encoding_mask = __ CreateICmpNE(and_encoding_mask,
                                                       __ getInt64(0));
  llvm::Value* external_string = LoadFieldOperand(phi_str,
                                           ExternalString::kResourceDataOffset);
  __ CreateCondBr(not_encoding_mask, one_byte_external, two_byte_external);

  __ SetInsertPoint(two_byte_external);
  llvm::BasicBlock* done = NewBlock("StringCharCodeAt Done");
  llvm::Value* two_byte_offset = __ CreateMul(phi_index, __ getInt32(2));
  llvm::Value* base_casted_two_ext = __ CreateBitOrPointerCast(external_string,
                                                               Types::ptr_i8);
  llvm::Value* two_byte_address = __ CreateGEP(base_casted_two_ext,
                                               two_byte_offset);
  llvm::Value* casted_addr_two_ext = __ CreatePointerCast(two_byte_address,
                                                          Types::ptr_tagged);
  llvm::Value* two_byte_ex_load = __ CreateLoad(casted_addr_two_ext);
  two_byte_ex_load = __ CreateBitOrPointerCast(two_byte_ex_load, Types::i64);
  llvm::Value* two_byte_external_result = __ CreateAnd(two_byte_ex_load,
                                                      __ getInt64(0x0000ffff));
  __ CreateBr(done);

  __ SetInsertPoint(one_byte_external);
  llvm::Value* one_byte_offset = __ CreateAdd(phi_index,
                                              __ getInt32(kHeapObjectTag));
  llvm::Value* base_casted_one_ext = __ CreateIntToPtr(external_string,
                                                       Types::ptr_i8);
  llvm::Value* one_byte_addr_ext = __ CreateGEP(base_casted_one_ext,
                                           one_byte_offset);
  llvm::Value* casted_addr_one_ext = __ CreatePointerCast(one_byte_addr_ext,
                                                  Types::ptr_tagged);
  llvm::Value* add_result_one =  __ CreateLoad(casted_addr_one_ext);
  add_result_one = __ CreateBitOrPointerCast(add_result_one, Types::i64);
  llvm::Value* one_byte_external_result = __ CreateAnd(add_result_one,
                                                      __ getInt64(0x000000ff));
  __ CreateBr(done);

  __ SetInsertPoint(seq_string);
  llvm::BasicBlock* one_byte = NewBlock("StringCharCodeAt OneByte");
  llvm::BasicBlock* two_byte = NewBlock("StringCharCodeAt OneByteCont");
  STATIC_ASSERT((kStringEncodingMask & kOneByteStringTag) != 0);
  STATIC_ASSERT((kStringEncodingMask & kTwoByteStringTag) == 0);
  llvm::Value* and_seq_str = __ CreateAnd(phi_result_type,
                                        __ getInt64(kStringEncodingMask));
  llvm::Value* seq_not_zero = __ CreateICmpNE(and_seq_str, __ getInt64(0));
  __ CreateCondBr(seq_not_zero, one_byte, two_byte);

  __ SetInsertPoint(two_byte);
  STATIC_ASSERT(kSmiTag == 0 && kSmiTagSize == 1);

  llvm::Value* two_byte_index = __ CreateMul(phi_index, __ getInt32(2));
  llvm::Value* two_byte_add_index = __ CreateAdd(two_byte_index,
                                    __ getInt32(SeqTwoByteString::kHeaderSize));
  llvm::Value* base_casted_two = __ CreateIntToPtr(phi_str, Types::ptr_i8);
  llvm::Value* address_two = __ CreateGEP(base_casted_two, two_byte_add_index);
  llvm::Value* casted_adds_two = __ CreatePointerCast(address_two,
                                                     Types::ptr_tagged);
  llvm::Value* two_byte_load = __ CreateLoad(casted_adds_two);
  two_byte_load = __ CreateBitOrPointerCast(two_byte_load, Types::i64);
  llvm::Value* two_byte_result = __ CreateAnd(two_byte_load,
                                             __ getInt64(0x0000ffff));
  __ CreateBr(done);

  __ SetInsertPoint(one_byte);
  llvm::Value* one_byte_add_index = __ CreateAdd(phi_index,
                               __ getInt32(SeqTwoByteString::kHeaderSize - 1));
  llvm::Value* base_casted_one = __ CreateIntToPtr(phi_str, Types::ptr_i8);
  llvm::Value* addr_one = __ CreateGEP(base_casted_one, one_byte_add_index);
  llvm::Value* casted_adds_one = __ CreatePointerCast(addr_one, Types::ptr_tagged);
  llvm::Value* one_byte_load = __ CreateLoad(casted_adds_one);
  one_byte_load = __ CreateBitOrPointerCast(one_byte_load, Types::i64);
  llvm::Value* one_byte_result = __ CreateAnd(one_byte_load, __ getInt64(0x000000ff));
  __ CreateBr(done);

  __ SetInsertPoint(done);
  llvm::PHINode* result_gen = __ CreatePHI(Types::i64, 4);
  result_gen->addIncoming(one_byte_external_result, one_byte_external);
  result_gen->addIncoming(two_byte_external_result, two_byte_external);
  result_gen->addIncoming(one_byte_result, one_byte);
  result_gen->addIncoming(two_byte_result, two_byte);
  __ CreateBr(set_value);

  __ SetInsertPoint(deferred);
  llvm::PHINode* str_phi_deferred = __ CreatePHI(Types::tagged, 2);
  str_phi_deferred->addIncoming(str, insert);
  str_phi_deferred->addIncoming(phi_str, cont_inside_seq);

  std::vector<llvm::Value*> params;
  //TODO : implement non constant case
  STATIC_ASSERT(String::kMaxLength <= Smi::kMaxValue);
  if (instr->index()->IsConstant()) {
    UNIMPLEMENTED();
  } else {
    llvm::Value* const_index = Integer32ToSmi(instr->index());
    params.push_back(const_index);
  }
  params.push_back(str_phi_deferred);
  llvm::Value* call = CallRuntimeFromDeferred(Runtime::kStringCharCodeAtRT,
                                               Use(instr->context()),
                                               params);
   llvm::Value* call_casted = __ CreatePtrToInt(call, Types::i64);
  __ CreateBr(set_value);

  __ SetInsertPoint(set_value);
  llvm::PHINode* phi = __ CreatePHI(Types::i64, 2);
  phi->addIncoming(result_gen, insert);
  phi->addIncoming(call_casted, deferred);
  auto result = __ CreateTruncOrBitCast(phi, Types::i32);
  instr->set_llvm_value(result);
}

void LLVMChunkBuilder::DoStringCharFromCode(HStringCharFromCode* instr) {
  //TODO:Fast case implementation
  std::vector<llvm::Value*> args;
  llvm::Value* arg1 = Integer32ToSmi(instr->value());
  args.push_back(arg1);
  llvm::Value* result =  CallRuntimeFromDeferred(Runtime::kCharFromCode, Use(instr->context()), args);
  instr->set_llvm_value(result);
}

void LLVMChunkBuilder::DoStringCompareAndBranch(HStringCompareAndBranch* instr) {
  //TODO: not tested string-validate-input.js in doTest
  llvm::Value* context = Use(instr->context());
  llvm::Value* left = Use(instr->left());
  llvm::Value* right = Use(instr->right());
  Token::Value op = instr->token();
  AllowHandleAllocation allow_handles;
  AllowHeapAllocation allow_heap;
  Handle<Code> ic = CodeFactory::StringCompare(isolate()).code();
  std::vector<llvm::Value*> params;
  params.push_back(context);
  params.push_back(left);
  params.push_back(right);
  llvm::Value* result =  CallCode(ic, llvm::CallingConv::X86_64_V8_S10, params);
  llvm::Value* return_val = __ CreatePtrToInt(result, Types::i64);
  //TODO (Jivan) It seems redudant
  llvm::Value* test = __ CreateAnd(return_val, return_val);
  llvm::CmpInst::Predicate pred = TokenToPredicate(op, false, false);
  llvm::Value* cmp = __ CreateICmp(pred, test, __ getInt64(0));
  llvm::BranchInst* branch = __ CreateCondBr(cmp, Use(instr->SuccessorAt(0)),
                                             Use(instr->SuccessorAt(1)));
  instr->set_llvm_value(branch);
}

void LLVMChunkBuilder::DoSub(HSub* instr) {
  if(instr->representation().IsInteger32() || instr->representation().IsSmi()) {
    DCHECK(instr->left()->representation().Equals(instr->representation()));
    DCHECK(instr->right()->representation().Equals(instr->representation()));
    HValue* left = instr->left();
    HValue* right = instr->right();
    if (!instr->CheckFlag(HValue::kCanOverflow)) {
      llvm::Value* sub = __ CreateSub(Use(left), Use(right), "");
      instr->set_llvm_value(sub);
    } else {
      auto type = instr->representation().IsSmi() ? Types::i64 : Types::i32;
      llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(
          module_.get(), llvm::Intrinsic::ssub_with_overflow, type);
      llvm::Value* params[] = { Use(left), Use(right) };
      llvm::Value* call = __ CreateCall(intrinsic, params);
      llvm::Value* sub = __ CreateExtractValue(call, 0);
      llvm::Value* overflow = __ CreateExtractValue(call, 1);
      DeoptimizeIf(overflow, Deoptimizer::kOverflow);
      instr->set_llvm_value(sub);
    }
  } else if (instr->representation().IsDouble()) {
    DCHECK(instr->representation().IsDouble());
    DCHECK(instr->left()->representation().IsDouble());
    DCHECK(instr->right()->representation().IsDouble());
    HValue* left = instr->left();
    HValue* right = instr->right();
    llvm::Value* fSub =  __ CreateFSub(Use(left), Use(right), "");
    instr->set_llvm_value(fSub);  
  } else if(instr->representation().IsTagged()) {
    llvm::Value* context = Use(instr->context());
    llvm::Value* left = Use(instr->left());
    llvm::Value* right = Use(instr->right());
    std::vector<llvm::Value*> params;
    params.push_back(context);
    params.push_back(left);
    params.push_back(right);
    AllowHandleAllocation allow_handles;
    Handle<Code> code =
        CodeFactory::BinaryOpIC(isolate(), Token::SUB,
                                instr->strength()).code();
    llvm::Value* sub = CallCode(code, llvm::CallingConv::X86_64_V8_S10, params);
    instr->set_llvm_value(sub);
  } else {
    UNIMPLEMENTED();
  }
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
    // map is a tagged value.
    auto new_map = Move(to_map, RelocInfo::EMBEDDED_OBJECT);
    auto store_addr = FieldOperand(object, HeapObject::kMapOffset);
    auto casted_store_addr = __ CreateBitCast(store_addr, Types::ptr_tagged);
    __ CreateStore(new_map, casted_store_addr);
    // Write barrier. TODO(llvm): give llvm.gcwrite and company a thought.
    RecordWriteForMap(object, new_map);
    __ CreateBr(end);
  } else {


   AllowHeapAllocation allow_heap;
   bool is_js_array = from_map->instance_type() == JS_ARRAY_TYPE;
   llvm::Value* map = MoveHeapObject(to_map);
   TransitionElementsKindStub stub(isolate(), from_kind, to_kind, is_js_array);
   std::vector<llvm::Value*> params;
   params.push_back(object);
   params.push_back(map);
   params.push_back(GetContext());
   AllowHandleAllocation allow_handles; 
   CallCode(stub.GetCode(), llvm::CallingConv::X86_64_V8_CES, params);
   //RecordSafepointWithLazyDeopt(instr, RECORD_SAFEPOINT_WITH_REGISTERS, 0);
   __ CreateBr(end);
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
  CallCode(code, llvm::CallingConv::X86_64_V8_RWS, params, false);
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
  llvm::Value* context = Use(instr->context());
  llvm::Value* value = Use(instr->value());
  llvm::BasicBlock* do_call = NewBlock("DoTypeof Call Stub");
  llvm::BasicBlock* no_call = NewBlock("DoTypeof Fast");
  llvm::BasicBlock* end = NewBlock("DoTypeof Merge");
  llvm::Value* not_smi = SmiCheck(value, true);
  __ CreateCondBr(not_smi, do_call, no_call);

  __ SetInsertPoint(no_call);
  Factory* factory = isolate()->factory();
  Handle<String> string = factory->number_string();
  llvm::Value* val = MoveHeapObject(string);
  __ CreateBr(end);

  __ SetInsertPoint(do_call);

  AllowHandleAllocation allow_handles;
  std::vector<llvm::Value*> params;
  params.push_back(context);
  params.push_back(value);
  TypeofStub stub(isolate());
  Handle<Code> code = stub.GetCode();
  llvm::Value* call = CallCode(code, llvm::CallingConv::X86_64_V8_S8, params);
  __ CreateBr(end);
  __ SetInsertPoint(end);
  llvm::PHINode* phi = __ CreatePHI(Types::tagged, 2);
  phi->addIncoming(val, no_call);
  phi->addIncoming(call, do_call);
  instr->set_llvm_value(phi);
}

void LLVMChunkBuilder::DoTypeofIsAndBranch(HTypeofIsAndBranch* instr) {
  llvm::Value* input = Use(instr->value());
  Factory* factory = isolate()->factory();
  llvm::BasicBlock* not_smi = NewBlock("DoTypeofIsAndBranch NotSmi");
//  llvm::BranchInst* branch = nullptr;
  Handle<String> type_name = instr->type_literal();
  if (String::Equals(type_name, factory->number_string())) {
    llvm::Value* smi_cond = SmiCheck(input);
    __ CreateCondBr(smi_cond, Use(instr->SuccessorAt(1)), not_smi);
    __ SetInsertPoint(not_smi);

    llvm::Value* root = LoadFieldOperand(input, HeapObject::kMapOffset);
    llvm::Value* cmp_root = CompareRoot(root, Heap::kHeapNumberMapRootIndex);
     __ CreateCondBr(cmp_root, Use(instr->SuccessorAt(0)),
                     Use(instr->SuccessorAt(1)));
  } else if (String::Equals(type_name, factory->string_string())) {
    UNIMPLEMENTED();
    //TODO: find test
    llvm::Value* smi_cond = SmiCheck(input);
    __ CreateCondBr(smi_cond, Use(instr->SuccessorAt(1)), not_smi);

    __ SetInsertPoint(not_smi);
    llvm::Value* map = LoadFieldOperand(input, HeapObject::kMapOffset);
    auto imm = static_cast<int8_t>(FIRST_NONSTRING_TYPE);
    llvm::Value* type_offset = LoadFieldOperand(map, Map::kInstanceTypeOffset);
    llvm::Value* cond = __ CreateICmpUGE(type_offset, __ getInt64(imm));
    __ CreateCondBr(cond, Use(instr->SuccessorAt(0)),
                    Use(instr->SuccessorAt(1)));
  } else if (String::Equals(type_name, factory->symbol_string())) {
    UNIMPLEMENTED();
  } else if (String::Equals(type_name, factory->boolean_string())) {
    UNIMPLEMENTED();
  } else if (String::Equals(type_name, factory->undefined_string())) {
    //TODO: not tested
    llvm::BasicBlock* after_cmp_root = NewBlock("DoTypeofIsAndBranch "
                                          "after compare root");
    llvm::Value* cmp_root = CompareRoot(input, Heap::kUndefinedValueRootIndex);
    __ CreateCondBr(cmp_root, Use(instr->SuccessorAt(0)), after_cmp_root);

    __ SetInsertPoint(after_cmp_root);
    llvm::BasicBlock* after_check_smi = NewBlock("DoTypeofIsAndBranch "
                                                 "after check smi");
    llvm::Value* smi_cond = SmiCheck(input, false);
    __ CreateCondBr(smi_cond, Use(instr->SuccessorAt(1)), after_check_smi);

    __ SetInsertPoint(after_check_smi);
    llvm::Value* map_offset = LoadFieldOperand(input, HeapObject::kMapOffset);
    llvm::Value* is_undetectable = __ getInt64(1 << Map::kIsUndetectable);
    llvm::Value* result = LoadFieldOperand(map_offset, Map::kBitFieldOffset);
    llvm::Value* test = __ CreateAnd(result, is_undetectable);
    llvm::Value* cmp = __ CreateICmpNE(test, __ getInt64(0));
    __ CreateCondBr(cmp, Use(instr->SuccessorAt(0)), Use(instr->SuccessorAt(1)));
  } else if (String::Equals(type_name, factory->function_string())) {
    llvm::Value* smi_cond = SmiCheck(input);
    __ CreateCondBr(smi_cond, Use(instr->SuccessorAt(1)), not_smi);

    __ SetInsertPoint(not_smi);
    llvm::Value* map_offset = LoadFieldOperand(input, HeapObject::kMapOffset);
    llvm::Value* bit_field = LoadFieldOperand(map_offset, Map::kBitFieldOffset);
    llvm::Value* input_chenged = __ CreateAnd(bit_field, __ getInt64(0x000000ff));     //movzxbl
    llvm::Value* imm = __ getInt64((1 << Map::kIsCallable) | 
                                   (1 << Map::kIsUndetectable));
    llvm::Value* result = __ CreateAnd(input_chenged,imm);
    llvm::Value* cmp = __ CreateICmpEQ(result,
                                       __ getInt64(1 << Map::kIsCallable));
    __ CreateCondBr(cmp, Use(instr->SuccessorAt(0)),
                    Use(instr->SuccessorAt(1)));
  } else if (String::Equals(type_name, factory->object_string())) {
    llvm::Value* smi_cond = SmiCheck(input);
    __ CreateCondBr(smi_cond, Use(instr->SuccessorAt(1)), not_smi);
    __ SetInsertPoint(not_smi);
    llvm::BasicBlock* after_cmp_root = NewBlock("DoTypeofIsAndBranch "
                                          "after compare root");
    llvm::Value* cmp_root = CompareRoot(input, Heap::kNullValueRootIndex);
    __ CreateCondBr(cmp_root, Use(instr->SuccessorAt(0)), after_cmp_root);
    
    __ SetInsertPoint(after_cmp_root);
    llvm::BasicBlock* after_cmp_type = NewBlock("DoTypeofIsAndBranch "
                                                "after cmpare type");
    STATIC_ASSERT(LAST_SPEC_OBJECT_TYPE == LAST_TYPE);
    llvm::Value* map = LoadFieldOperand(input, HeapObject::kMapOffset);
    llvm::Value* result  = LoadFieldOperand(map, Map::kInstanceTypeOffset);
    llvm::Value* type = __ getInt64(static_cast<int>(FIRST_SPEC_OBJECT_TYPE));
    llvm::Value* cmp = __ CreateICmpULT(result, type);
    __ CreateCondBr(cmp, Use(instr->SuccessorAt(1)), after_cmp_type);
    
    __ SetInsertPoint(after_cmp_type);
    llvm::Value* bit_field = LoadFieldOperand(map, Map::kBitFieldOffset);
    llvm::Value* imm = __ getInt64((1 << Map::kIsCallable) |
                                   (1 << Map::kIsUndetectable));
    llvm::Value* test = __ CreateAnd(bit_field, imm);
    llvm::Value* cmp_result = __ CreateICmpNE(test, __ getInt64(0));
    __ CreateCondBr(cmp_result, Use(instr->SuccessorAt(0)),
                                Use(instr->SuccessorAt(1))); 
    
    // clang-format off
#define SIMD128_TYPE(TYPE, Type, type, lane_count, lane_type)         \
  } else if (String::Equals(type_name, factory->type##_string())) {   \
    llvm::Value* smi_cond = SmiCheck(input);                          \
    __ CreateCondBr(smi_cond, Use(instr->SuccessorAt(1)), not_smi);   \
    __ SetInsertPoint(not_smi);                                       \
    llvm::Value* value = LoadFieldOperand(input,                      \
                                          HeapObject::kMapOffset);    \
    llvm::Value* cmp_root =  CompareRoot(value,                       \
                                    Heap::k##Type##MapRootIndex);     \
    __ CreateCondBr(cmp_root, Use(instr->SuccessorAt(0)),             \
                              Use(instr->SuccessorAt(1)));            \

  SIMD128_TYPES(SIMD128_TYPE)
#undef SIMD128_TYPE
    // clang-format on   

  } else {
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoIntegerMathAbs(HUnaryMathOperation* instr) {
  llvm::BasicBlock* is_negative = NewBlock("ABS INTEGER CANDIDATE IS NEGATIVE");
  llvm::BasicBlock* is_positive = NewBlock("ABS INTEGER CANDIDATE IS POSITIVE");

  llvm::Value* zero = __ getInt32(0);
  llvm::Value* cmp =  __ CreateICmpSLT(Use(instr->value()), zero);
  __ CreateCondBr(cmp, is_negative, is_positive);
  __ SetInsertPoint(is_negative);
  llvm::Value* neg_val =  __ CreateNeg(Use(instr->value()));
  llvm::Value* is_neg = __ CreateICmpSLT(neg_val, zero);
  DeoptimizeIf(is_neg, Deoptimizer::kOverflow, false, is_positive);
  __ SetInsertPoint(is_positive);
  llvm::Value* val = Use(instr->value());
  llvm::PHINode* phi = __ CreatePHI(Types::i32, 2);
  phi->addIncoming(neg_val, is_negative);
  phi->addIncoming(val, is_positive);
  instr->set_llvm_value(phi);
}

void LLVMChunkBuilder::DoSmiMathAbs(HUnaryMathOperation* instr) {
  llvm::BasicBlock* is_negative = NewBlock("ABS SMI CANDIDATE IS NEGATIVE");
  llvm::BasicBlock* is_positive = NewBlock("ABS SMI CANDIDATE IS POSITIVE");

  llvm::BasicBlock* insert_block = __ GetInsertBlock();
  llvm::Value* value = Use(instr->value());
  llvm::Value* cmp =  __ CreateICmpSLT(Use(instr->value()), __ getInt64(0));
  __ CreateCondBr(cmp, is_negative, is_positive);
  __ SetInsertPoint(is_negative);
  llvm::Value* neg_val =  __ CreateNeg(Use(instr->value()));
  llvm::Value* is_neg = __ CreateICmpSLT(neg_val, __ getInt64(0));
  DeoptimizeIf(is_neg, Deoptimizer::kOverflow, false, is_positive);
  __ SetInsertPoint(is_positive);
  llvm::PHINode* phi = __ CreatePHI(Types::smi, 2);
  phi->addIncoming(neg_val, is_negative);
  phi->addIncoming(value, insert_block);
  instr->set_llvm_value(phi);
}

void LLVMChunkBuilder::DoMathAbs(HUnaryMathOperation* instr) {
  Representation r = instr->representation();
  if (r.IsDouble()) {
    llvm::Function* fabs_intrinsic = llvm::Intrinsic::
          getDeclaration(module_.get(), llvm::Intrinsic::fabs, Types::float64);
    std::vector<llvm::Value*> params;
    params.push_back(Use(instr->value()));
    llvm::Value* f_abs = __ CreateCall(fabs_intrinsic, params);
    instr->set_llvm_value(f_abs);
  } else if (r.IsInteger32()) {
    DoIntegerMathAbs(instr);
  } else if (r.IsSmi()) {
    DoSmiMathAbs(instr);
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

void LLVMChunkBuilder::DoMathRound(HUnaryMathOperation* instr) {
  llvm::Value* llvm_double_one_half = llvm::ConstantFP::get(Types::float64, 0.5);
  llvm::Value* llvm_double_minus_one_half = llvm::ConstantFP::get(Types::float64, -0.5);
  llvm::Value* input_reg = Use(instr->value());
  llvm::Value* input_temp = nullptr;
  llvm::Value* xmm_scratch = nullptr;
  llvm::BasicBlock* round_to_zero = NewBlock("Round to zero");
  llvm::BasicBlock* round_to_one = NewBlock("Round to one");
  llvm::BasicBlock* below_one_half = NewBlock("Below one half");
  llvm::BasicBlock* above_one_half = NewBlock("Above one half");
  llvm::BasicBlock* not_equal = NewBlock("Not equal");
  llvm::BasicBlock* round_result = NewBlock("Jump to final Round result block");
  /*if (DeoptEveryNTimes()){
    UNIMPLEMENTED();
  }*/
  llvm::Value* cmp = __ CreateFCmpOGT(llvm_double_one_half, input_reg);
  __ CreateCondBr(cmp, below_one_half, above_one_half);

  __ SetInsertPoint(above_one_half);
  xmm_scratch = __ CreateFAdd(llvm_double_one_half, input_reg);
  llvm::Value* output_reg1 = __ CreateFPToSI(xmm_scratch, Types::i32);
  //DeoptimizeIF
  auto type = instr->representation().IsSmi() ? Types::i64 : Types::i32;
  llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
        llvm::Intrinsic::ssub_with_overflow, type);
  llvm::Value* params[] = { output_reg1, __ getInt32(0x1) };
  llvm::Value* call = __ CreateCall(intrinsic, params);
  llvm::Value* overflow = __ CreateExtractValue(call, 1);
  DeoptimizeIf(overflow, Deoptimizer::kOverflow);
  __ CreateBr(round_result);

  __ SetInsertPoint(below_one_half);
  cmp = __ CreateFCmpOLE(llvm_double_minus_one_half, input_reg);
  __ CreateCondBr(cmp, round_to_zero, round_to_one);

  __ SetInsertPoint(round_to_one);
  input_temp = __ CreateFSub(input_reg, llvm_double_minus_one_half);
  llvm::Value* output_reg2 = __ CreateFPToSI(input_temp, Types::i32);
  auto instr_type = instr->representation().IsSmi() ? Types::i64 : Types::i32;
  llvm::Function* ssub_intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
        llvm::Intrinsic::ssub_with_overflow, instr_type);
  llvm::Value* parameters[] = { output_reg2, __ getInt32(0x1) };
  llvm::Value* call_intrinsic = __ CreateCall(ssub_intrinsic, parameters);
  llvm::Value* cmp_overflow = __ CreateExtractValue(call_intrinsic, 1);
  DeoptimizeIf(cmp_overflow, Deoptimizer::kOverflow);
  xmm_scratch = __ CreateSIToFP(output_reg2, Types::float64);
  cmp = __ CreateFCmpOEQ(xmm_scratch, input_reg);
  __ CreateCondBr(cmp, round_result, not_equal);

  __ SetInsertPoint(not_equal);
  llvm::Value* output_reg3 = __ CreateNSWSub(output_reg2, __ getInt32(1));
  __ CreateBr(round_result);

  __ SetInsertPoint(round_to_zero);
  if (instr->CheckFlag(HValue::kBailoutOnMinusZero)) {
    //UNIMPLEMENTED();
    llvm::Value* cmp_zero = __ CreateFCmpOLT(input_reg, __ CreateSIToFP(__ getInt64(0), Types::float64));
    DeoptimizeIf(cmp_zero, Deoptimizer::kMinusZero);
  }
  llvm::Value* output_reg4 = __ getInt32(6);
  __ CreateBr(round_result);

  __ SetInsertPoint(round_result);
  llvm::PHINode* phi = __ CreatePHI(Types::i32, 4);
  phi->addIncoming(output_reg1, above_one_half);
  phi->addIncoming(output_reg2, round_to_one);
  phi->addIncoming(output_reg3, not_equal);
  phi->addIncoming(output_reg4, round_to_zero);
  instr->set_llvm_value(phi);
}

void LLVMChunkBuilder::DoMathLog(HUnaryMathOperation* instr) {
  llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
          llvm::Intrinsic::log, Types::float64);
  std::vector<llvm::Value*> params;
  params.push_back(Use(instr->value()));
  llvm::Value* log = __ CreateCall(intrinsic, params);
  instr->set_llvm_value(log);
}

void LLVMChunkBuilder::DoMathExp(HUnaryMathOperation* instr) {
  llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(module_.get(),
          llvm::Intrinsic::exp, Types::float64);
  std::vector<llvm::Value*> params;
  params.push_back(Use(instr->value()));
  llvm::Value* exp = __ CreateCall(intrinsic, params);
  instr->set_llvm_value(exp);
}

void LLVMChunkBuilder::DoUnaryMathOperation(HUnaryMathOperation* instr) {
  switch (instr->op()) {
    case kMathAbs:
      DoMathAbs(instr);
      break;
    case kMathPowHalf:
      DoMathPowHalf(instr);
      break;
    case kMathFloor: {
      DoMathFloor(instr);
      break;
    }
    case kMathRound: {
      DoMathRound(instr);
      break;
    }
    case kMathFround: {
        //FIXME(llvm): Is this right?
        llvm::Value* value = Use(instr->value());
        llvm::Value* trunc_fp = __ CreateFPTrunc(value, Types::float32);
        llvm::Value* result = __ CreateFPExt(trunc_fp, Types::float64);
        instr->set_llvm_value(result);
       break;
      }
    case kMathLog: {
      //UNIMPLEMENTED();
      DoMathLog(instr);
      break;
    }
    case kMathExp: {
      //UNIMPLEMENTED();
      DoMathExp(instr);
      break;
    }
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
  int env_index = instr->index();
  int spill_index = 0;
  if (instr->environment()->is_parameter_index(env_index)) {
    spill_index = chunk()->GetParameterStackSlot(env_index);
    spill_index = -spill_index;
    llvm::Function::arg_iterator it = function_->arg_begin();
    int i = 0;
    while (++i < 3 + spill_index) ++it;
    llvm::Value* result = it;
    instr->set_llvm_value(result);
  } else {
    spill_index = env_index - instr->environment()->first_local_index();
    if (spill_index > LUnallocated::kMaxFixedSlotIndex) {
      UNIMPLEMENTED();
    }
    if (spill_index >=0) {
      bool is_volatile = true;
      llvm::Value* result = __ CreateLoad(osr_preserved_values_[spill_index], is_volatile);
      result = __ CreateBitOrPointerCast(result, GetLLVMType(instr->representation()));
      instr->set_llvm_value(result);
    } else {
      //TODO: Check this case
      DCHECK(spill_index == -1);
      spill_index = 1;
      llvm::Function::arg_iterator it = function_->arg_begin();
      int i = 0;
      while (++i < 3 + spill_index) ++it;
      llvm::Value* result = it;
      instr->set_llvm_value(result);
    }
    
  }
}

void LLVMChunkBuilder::DoUseConst(HUseConst* instr) {
  //UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoWrapReceiver(HWrapReceiver* instr) {
  llvm::Value* receiver = Use(instr->receiver());
  llvm::Value* function = Use(instr->function());
  llvm::BasicBlock* insert_block = __ GetInsertBlock();
  llvm::BasicBlock* global_object = NewBlock("DoWrapReceiver Global object");
  llvm::BasicBlock* not_global_object = NewBlock("DoWrapReceiver "
                                                 "Not global object");
  llvm::BasicBlock* not_equal = NewBlock("DoWrapReceiver not_equal");
  llvm::BasicBlock* receiver_ok = NewBlock("DoWrapReceiver Receiver ok block");
  llvm::BasicBlock* dist = NewBlock("DoWrapReceiver Distance");
  llvm::BasicBlock* receiver_fail = NewBlock("DoWrapReceiver Receiver"
                                             " fail block");

  if (!instr->known_function()){
    llvm::Value* op = LoadFieldOperand(function,
                                       JSFunction::kSharedFunctionInfoOffset);
    llvm::Value* bit_with_byte =
                __ getInt64(1 << SharedFunctionInfo::kStrictModeBitWithinByte);
    llvm::Value* byte_offset = LoadFieldOperand(op,
                                 SharedFunctionInfo::kStrictModeByteOffset);
    llvm::Value* casted_offset = __ CreatePtrToInt(byte_offset, Types::i64);
    llvm::Value* cmp = __ CreateICmpNE(casted_offset, bit_with_byte);
    __ CreateCondBr(cmp, receiver_ok, receiver_fail);
    __ SetInsertPoint(receiver_fail);
    llvm::Value* native_byte_offset = LoadFieldOperand(op,
                                 SharedFunctionInfo::kNativeByteOffset);
    llvm::Value* native_bit_with_byte =
                 __ getInt64(1 << SharedFunctionInfo::kNativeBitWithinByte);
    llvm::Value* casted_native_offset = __ CreatePtrToInt(native_byte_offset,
                                                          Types::i64);
    llvm::Value* compare = __ CreateICmpNE(casted_native_offset,
                                          native_bit_with_byte);
    __ CreateCondBr(compare, receiver_ok, dist);
  } else
    __ CreateBr(dist);
  __ SetInsertPoint(dist);

  // Normal function. Replace undefined or null with global receiver.
  llvm::Value* compare_root = CompareRoot(receiver, Heap::kNullValueRootIndex);
  __ CreateCondBr(compare_root, global_object, not_global_object);
  __ SetInsertPoint(not_global_object);
  llvm::Value* comp_root = CompareRoot(receiver,
                                      Heap::kUndefinedValueRootIndex);
  __ CreateCondBr(comp_root, global_object, not_equal);

  // The receiver should be a JS object
  __ SetInsertPoint(not_equal);
  llvm::Value* is_smi = SmiCheck(receiver);
  DeoptimizeIf(is_smi, Deoptimizer::kSmi);
  llvm::Value* compare_obj = CmpObjectType(receiver,
                                          FIRST_SPEC_OBJECT_TYPE,
                                          llvm::CmpInst::ICMP_ULT);
  DeoptimizeIf(compare_obj, Deoptimizer::kNotAJavaScriptObject);
  __ CreateBr(receiver_ok);

  __ SetInsertPoint(global_object);
  llvm::Value* global_receiver = LoadFieldOperand(function,
                                                 JSFunction::kContextOffset);
  global_receiver = LoadFieldOperand(global_receiver,
                             Context::SlotOffset(Context::GLOBAL_OBJECT_INDEX));
  global_receiver = LoadFieldOperand(global_receiver,
                                    GlobalObject::kGlobalProxyOffset);
  __ CreateBr(receiver_ok);

  __ SetInsertPoint(receiver_ok);
  llvm::PHINode* phi = __ CreatePHI(Types::tagged, 2);
  phi->addIncoming(global_receiver, global_object);
  phi->addIncoming(receiver, insert_block);
  instr->set_llvm_value(phi);
}


llvm::Value* LLVMChunkBuilder::CmpObjectType(llvm::Value* heap_object,
                                             InstanceType type,
                                             llvm::CmpInst::Predicate predicate) {
  llvm::Value* map = LoadFieldOperand(heap_object, HeapObject::kMapOffset);
  llvm::Value* map_as_ptr_to_i8 = __ CreateBitOrPointerCast(map, Types::ptr_i8);
  llvm::Value* object_type_addr = ConstructAddress(
      map_as_ptr_to_i8, Map::kInstanceTypeOffset - kHeapObjectTag);
  llvm::Value* object_type = __ CreateLoad(object_type_addr);
  llvm::Value* expected_type = __ getInt8(static_cast<int8_t>(type));
  return __ CreateICmp(predicate, object_type, expected_type);
}

void LLVMChunkBuilder::DoCheckArrayBufferNotNeutered(
    HCheckArrayBufferNotNeutered* instr) {
  llvm::Value* view = Use(instr->value());
  llvm::Value* array_offset = LoadFieldOperand(view,
                                               JSArrayBufferView::kBufferOffset);
  llvm::Value* bit_field_offset = LoadFieldOperand(array_offset,
                                                   JSArrayBuffer::kBitFieldOffset);
  bit_field_offset = __ CreateBitOrPointerCast(bit_field_offset, Types::i64);
  llvm::Value* shift = __ getInt64(1 << JSArrayBuffer::WasNeutered::kShift);
  llvm::Value* test = __ CreateAnd(bit_field_offset, shift);
  llvm::Value* cmp = __ CreateICmpNE(test, __ getInt64(0));
  DeoptimizeIf(cmp, Deoptimizer::kOutOfBounds);
}

void LLVMChunkBuilder::DoLoadGlobalViaContext(HLoadGlobalViaContext* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoMaybeGrowElements(HMaybeGrowElements* instr) {
  //TODO: not tested, 3d-cube.js in functions MMulti
  llvm::BasicBlock* insert = __ GetInsertBlock();
  llvm::BasicBlock* deferred = NewBlock("MaybeGrowElements deferred");
  llvm::BasicBlock* done = NewBlock("MaybeGrowElements done");
  llvm::Value* result = Use(instr->elements());
  llvm::Value* result_from_deferred = nullptr;
  HValue* key = instr->key();
  HValue* current_capacity = instr->current_capacity();
  DCHECK(instr->key()->representation().IsInteger32());
  DCHECK(instr->current_capacity()->representation().IsInteger32());
  if (key->IsConstant() && current_capacity->IsConstant()) {
    UNIMPLEMENTED();
  } else if (key->IsConstant()) {
    int32_t constant_key = (HConstant::cast(key))->Integer32Value();
    llvm::Value* capacity = Use(instr->current_capacity());
    llvm::Value* cmp = __ CreateICmpSLE(capacity, __ getInt32(constant_key));
    __ CreateCondBr(cmp, deferred, done);
  } else if (current_capacity->IsConstant()) {
    UNIMPLEMENTED();
  } else {
    llvm::Value* cmp = __ CreateICmpSGE(Use(key), Use(current_capacity));
    __ CreateCondBr(cmp, deferred, done);
  }

  __ SetInsertPoint(deferred);
  std::vector<llvm::Value*> params;
  //PushSafepointRegistersScope scope(this);
  if (instr->object()->IsConstant()) {
    HConstant* constant_object = HConstant::cast(instr->object());
    if (instr->object()->representation().IsSmi()) {
      UNIMPLEMENTED();
    } else {
      Handle<Object> handle_value = constant_object->handle(isolate());
      llvm::Value* object = MoveHeapObject(handle_value);
      params.push_back(object);
    }
  } else {
    llvm::Value* object = Use(instr->object());
    params.push_back(object);
  }

  if (key->IsConstant()) {
    HConstant* constant = HConstant::cast(key);
    Smi* smi_key = Smi::FromInt(constant->Integer32Value());
    llvm::Value* smi = ValueFromSmi(smi_key);
    params.push_back(smi);
  } else {
    llvm::Value* smi_key = Integer32ToSmi(key);
    params.push_back(smi_key);
  }

  GrowArrayElementsStub stub(isolate(), instr->is_js_array(),
                             instr->kind());

  AllowHandleAllocation allow_handle;
  AllowHeapAllocation allow_heap;
  result_from_deferred = CallCode(stub.GetCode(),
                                  llvm::CallingConv::X86_64_V8_S12,
                                  params);
//  RecordSafepointWithLazyDeopt(instr, RECORD_SAFEPOINT_WITH_REGISTERS, 0);
  // __ StoreToSafepointRegisterSlot(result, result);
  llvm::Value* is_smi = SmiCheck(result_from_deferred);
  DeoptimizeIf(is_smi, Deoptimizer::kSmi);
  __ CreateBr(done);

  __ SetInsertPoint(done);
  DCHECK(result->getType() == result_from_deferred->getType());
  llvm::PHINode* phi = __ CreatePHI(result->getType(), 2);
  phi->addIncoming(result, insert);
  phi->addIncoming(result_from_deferred, deferred);
  instr->set_llvm_value(phi);
}

void LLVMChunkBuilder::DoPrologue(HPrologue* instr) {
  if (info_->num_heap_slots() > 0) {
    UNIMPLEMENTED();
  }
}

void LLVMChunkBuilder::DoStoreGlobalViaContext(HStoreGlobalViaContext* instr) {
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

  if (representation.IsDouble()) {
    is_double_.Add(values_.length() - 1, zone());
  }
}

void LLVMRelocationData::DumpSafepointIds() {
#ifdef DEBUG
  std::cerr << "< Safepoint ids begin: \n";
  for (GrowableBitVector::Iterator it(&is_safepoint_, zone_);
       !it.Done();
       it.Advance()) {
    std::cerr << it.Current() << " ";
  }
  std::cerr << "Safepoint ids end >\n";
#endif
}

#undef __

} }  // namespace v8::internal
