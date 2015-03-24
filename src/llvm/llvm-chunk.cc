// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <cstdio>
#include "llvm-chunk.h"

namespace v8 {
namespace internal {

LLVMChunk::~LLVMChunk() {}

Handle<Code> LLVMChunk::Codegen() {
  UNIMPLEMENTED();
  return Handle<Code>();
}

LLVMChunk* LLVMChunk::NewChunk(HGraph *graph) {
  DisallowHandleAllocation no_handles;
  DisallowHeapAllocation no_gc;
  graph->DisallowAddingNewValues();
//  int values = graph->GetMaximumValueID();
  CompilationInfo* info = graph->info();

//  if (values > LUnallocated::kMaxVirtualRegisters) {
//    info->AbortOptimization(kNotEnoughVirtualRegistersForValues);
//    return NULL;
//  }
//  LAllocator allocator(values, graph);
  LLVMChunkBuilder builder(info, graph);
  LLVMChunk* chunk = builder.Build();
  if (chunk == NULL) return NULL;

//  if (!allocator.Allocate(chunk)) {
//    info->AbortOptimization(kNotEnoughVirtualRegistersRegalloc);
//    return NULL;
//  }

//  chunk->set_allocated_double_registers(
//      allocator.assigned_double_registers());

  return chunk;
}

void LLVMChunkBuilder::CreateBasicBlock(HBasicBlock* block) {
  if (!block->llvm_basic_block()) {
    llvm::BasicBlock *llvm_block = llvm::BasicBlock::Create(
        LLVMGranularity::getInstance().context(),
        "BlockEntry", function_.get());
    block->set_llvm_basic_block(llvm_block);
  }
}

LLVMChunk* LLVMChunkBuilder::Build() {
  chunk_ = new(zone()) LLVMChunk(info(), graph());
  module_ = LLVMGranularity::getInstance().CreateModule();
  status_ = BUILDING;

//  // If compiling for OSR, reserve space for the unoptimized frame,
//  // which will be subsumed into this frame.
//  if (graph()->has_osr()) {
//    for (int i = graph()->osr()->UnoptimizedFrameSlots(); i > 0; i--) {
//      chunk()->GetNextSpillIndex(GENERAL_REGISTERS);
//    }
//  }

  LLVMContext& context = LLVMGranularity::getInstance().context();

  // First param is context (v8, js context) which goes to esi,
  // second param is the callee's JSFunction object (edi),
  // third param is Parameter 0 which is I am not sure what
  int num_parameters = info()->num_parameters() + 3;

  std::vector<llvm::Type*> params(num_parameters, nullptr);
  for (auto i = 0; i < num_parameters; i++) {
    // For now everything is Int64. Probably it is even right for x64.
    // So in that case we are going to do come casts AFAIK
    params[i] = llvm::Type::getInt64Ty(context);
  }
  llvm::ArrayRef<llvm::Type*> paramsRef(params);
  llvm::FunctionType *function_type = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(context), params, false);

  // TODO(llvm): return type for void JS functions?
  // I think it's all right, because undefined is a tagged value
  llvm::Function* raw_function_ptr = llvm::cast<llvm::Function>(
      module_->getOrInsertFunction("", function_type));

  function_ = std::unique_ptr<llvm::Function>(raw_function_ptr);
  function_->setCallingConv(llvm::CallingConv::X86_64_V8);

  const ZoneList<HBasicBlock*>* blocks = graph()->blocks();
  for (int i = 0; i < blocks->length(); i++) {
    HBasicBlock* next = NULL;
    if (i < blocks->length() - 1) next = blocks->at(i + 1);
    DoBasicBlock(blocks->at(i), next);
    if (is_aborted()) return NULL;
  }
  DCHECK(module_);
  llvm::outs() << *(module_.get());
  LLVMGranularity::getInstance().AddModule(std::move(module_));
  status_ = DONE;
  return chunk();
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
      CreateBasicBlock(successor);
      llvm_ir_builder_->CreateBr(successor->llvm_basic_block());
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

llvm::Value* LLVMChunkBuilder::Use(HValue* value) {
  if (value->EmitAtUses() && !value->llvm_value()) {
    HInstruction* instr = HInstruction::cast(value);
    VisitInstruction(instr);
  }
  return value->llvm_value();
}

void LLVMChunkBuilder::DoBasicBlock(HBasicBlock* block,
                                    HBasicBlock* next_block) {
  std::cerr << __FUNCTION__ << std::endl;
  DCHECK(is_building());
  CreateBasicBlock(block);
  llvm_ir_builder_ = llvm::make_unique<llvm::IRBuilder<>>(
      block->llvm_basic_block());
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

void LLVMChunkBuilder::DoBlockEntry(HBlockEntry* instr) {
  CreateBasicBlock(instr->block());
  // TODO(llvm): LGap & parallel moves (OSR support)
  // return new(zone()) LLabel(instr->block());
}

void LLVMChunkBuilder::DoContext(HContext* instr) {
  if (instr->HasNoUses()) return;
//  std::cerr << "#Uses == " << instr->UseCount() << std::endl;
  if (info()->IsStub()) {
    UNIMPLEMENTED();
  }
//  FIXME(llvm): we need it for bailouts and such
//  UNIMPLEMENTED();
//  return DefineAsRegister(new(zone()) LContext);
}

void LLVMChunkBuilder::DoParameter(HParameter* instr) {
// TODO(llvm)
//  for functions w/o parameters there is nonetheless
//  always a parameter
//  Parameter 0 type:Tagged
//  which is not important for us right now
//  since all it's usages are ArgumentsObject and Simulate
//  which also are not implemented at the moment
  int index = instr->index();
  std::cerr << "Parameter #" << index << std::endl;

  llvm::Function::arg_iterator it = function_->arg_begin();
  while (index-- > 0) ++it;
  instr->set_llvm_value(it);
//  UNIMPLEMENTED();
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

  // TODO(llvm): we need to implement this for deoptimization support.
  // seems to be the right implementation (same as for Lithium)
  instr->ReplayEnvironment(current_block_->last_environment());
}

void LLVMChunkBuilder::DoStackCheck(HStackCheck* instr) {
//  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoConstant(HConstant* instr) {
  // Note: constants might have EmitAtUses() == true
  Representation r = instr->representation();
  if (r.IsSmi()) {
    // TODO(llvm): use/write a function for that
    // FIXME(llvm): this block was not tested
    int64_t int32_value = instr->Integer32Value();
    llvm::Value* value = llvm_ir_builder_->getInt64(int32_value << (kSmiTagSize + kSmiShiftSize));
    instr->set_llvm_value(value);
  } else if (r.IsInteger32()) {
    UNIMPLEMENTED();
  } else if (r.IsDouble()) {
    UNIMPLEMENTED();
  } else if (r.IsExternal()) {
    UNIMPLEMENTED();
  } else if (r.IsTagged()) {
    Handle<Object> object = instr->handle(isolate());
    if (object->IsSmi()) {
      // TODO(llvm): use/write a function for that
      Smi* smi = Smi::cast(*object);
      intptr_t intptr_value = reinterpret_cast<intptr_t>(smi);
      llvm::Value* value = llvm_ir_builder_->getInt64(intptr_value);
      instr->set_llvm_value(value);
    } else {
      UNIMPLEMENTED();
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
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoAllocateBlockContext(HAllocateBlockContext* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoAllocate(HAllocate* instr) {
  UNIMPLEMENTED();
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
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoBoundsCheck(HBoundsCheck* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoBoundsCheckBaseIndexInformation(HBoundsCheckBaseIndexInformation* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoBranch(HBranch* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCallWithDescriptor(HCallWithDescriptor* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCallJSFunction(HCallJSFunction* instr) {
  UNIMPLEMENTED();
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

void LLVMChunkBuilder::DoChange(HChange* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCheckHeapObject(HCheckHeapObject* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCheckInstanceType(HCheckInstanceType* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoCheckMaps(HCheckMaps* instr) {
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
  UNIMPLEMENTED();
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
  UNIMPLEMENTED();
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
  UNIMPLEMENTED();
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
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoLoadGlobalGeneric(HLoadGlobalGeneric* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoLoadKeyed(HLoadKeyed* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoLoadKeyedGeneric(HLoadKeyedGeneric* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoLoadNamedField(HLoadNamedField* instr) {
  UNIMPLEMENTED();
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
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoOsrEntry(HOsrEntry* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoPower(HPower* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoPushArguments(HPushArguments* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoRegExpLiteral(HRegExpLiteral* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoRor(HRor* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoSar(HSar* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoSeqStringGetChar(HSeqStringGetChar* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoSeqStringSetChar(HSeqStringSetChar* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoShl(HShl* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoShr(HShr* instr) {
  UNIMPLEMENTED();
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
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoStoreKeyed(HStoreKeyed* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoStoreKeyedGeneric(HStoreKeyedGeneric* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoStoreNamedField(HStoreNamedField* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoStoreNamedGeneric(HStoreNamedGeneric* instr) {
  UNIMPLEMENTED();
}

void LLVMChunkBuilder::DoStringAdd(HStringAdd* instr) {
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
  UNIMPLEMENTED();
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
