// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <cstdio>
#include "llvm-chunk.h"

namespace v8 {
namespace internal {

LLVMChunk* LLVMChunk::NewChunk(HGraph *graph) {
  DisallowHandleAllocation no_handles;
  DisallowHeapAllocation no_gc;
  graph->DisallowAddingNewValues();
  int values = graph->GetMaximumValueID();
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

LLVMChunk* LLVMChunkBuilder::Build() {
  chunk_ = new(zone()) LLVMChunk(info(), graph());
  module_ = chunk()->module();
  status_ = BUILDING;

//  // If compiling for OSR, reserve space for the unoptimized frame,
//  // which will be subsumed into this frame.
//  if (graph()->has_osr()) {
//    for (int i = graph()->osr()->UnoptimizedFrameSlots(); i > 0; i--) {
//      chunk()->GetNextSpillIndex(GENERAL_REGISTERS);
//    }
//  }

  // here goes module_->AddFunction or so
//  llvm::Function raw_function_ptr =
//    cast<Function>(module_->getOrInsertFunction("", Type::getInt32Ty(Context),
//                                          Type::getInt32Ty(Context),
//                                          (Type *)0));
// function_ = std::unique_ptr<llvm::Function>(raw_function_ptr);
  // now, the problem is: get the parameters...
  // but what are they? Let's take a look at Hydrogen nodes.
  // (and also see the IRs in c1visualiser)

  // We can skip params and consider only funtions
  // with no arguments for now.
  // And come back later when it's figured out.

  const ZoneList<HBasicBlock*>* blocks = graph()->blocks();
  for (int i = 0; i < blocks->length(); i++) {
    HBasicBlock* next = NULL;
    if (i < blocks->length() - 1) next = blocks->at(i + 1);
    DoBasicBlock(blocks->at(i), next);
    if (is_aborted()) return NULL;
  }
  status_ = DONE;
  return chunk();
}

void LLVMChunkBuilder::VisitInstruction(HInstruction* current) {
  HInstruction* old_current = current_instruction_;
  current_instruction_ = current;

  LInstruction* instr = NULL;
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
//      instr = new(zone()) LGoto(successor);
      UNIMPLEMENTED();
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
//  current_instruction_ = old_current;
}

void LLVMChunkBuilder::DoBasicBlock(HBasicBlock* block,
                                    HBasicBlock* next_block) {
  DCHECK(is_building());
  current_block_ = block;
  next_block_ = next_block;
  if (block->IsStartBlock()) {
    block->UpdateEnvironment(graph_->start_environment());
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
  }
  HInstruction* current = block->first();
  int start = chunk()->instructions()->length();
  while (current != NULL && !is_aborted()) {
    // Code for constants in registers is generated lazily.
    if (!current->EmitAtUses()) {
      VisitInstruction(current);
    }
    current = current->next();
  }
  int end = chunk()->instructions()->length() - 1;
  if (end >= start) {
    block->set_first_instruction_index(start);
    block->set_last_instruction_index(end);
  }
  next_block_ = NULL;
  current_block_ = NULL;
}

void LLLVMChunkBuilder::DoBlockEntry(HBlockEntry* instr) {
//  return new(zone()) LLabel(instr->block());
}

void LLLVMChunkBuilder::DoContext(HContext* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoParameter(HParameter* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoArgumentsObject(HArgumentsObject* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoGoto(HGoto* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoSimulate(HSimulate* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoStackCheck(HStackCheck* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoConstant(HConstant* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoReturn(HReturn* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoAbnormalExit(HAbnormalExit* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoAccessArgumentsAt(HAccessArgumentsAt* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoAdd(HAdd* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoAllocateBlockContext(HAllocateBlockContext* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoAllocate(HAllocate* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoApplyArguments(HApplyArguments* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoArgumentsElements(HArgumentsElements* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoArgumentsLength(HArgumentsLength* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoBitwise(HBitwise* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoBoundsCheck(HBoundsCheck* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoBoundsCheckBaseIndexInformation(HBoundsCheckBaseIndexInformation* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoBranch(HBranch* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCallWithDescriptor(HCallWithDescriptor* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCallJSFunction(HCallJSFunction* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCallFunction(HCallFunction* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCallNew(HCallNew* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCallNewArray(HCallNewArray* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCallRuntime(HCallRuntime* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCallStub(HCallStub* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCapturedObject(HCapturedObject* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoChange(HChange* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCheckHeapObject(HCheckHeapObject* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCheckInstanceType(HCheckInstanceType* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCheckMaps(HCheckMaps* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCheckMapValue(HCheckMapValue* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCheckSmi(HCheckSmi* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCheckValue(HCheckValue* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoClampToUint8(HClampToUint8* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoClassOfTestAndBranch(HClassOfTestAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCompareNumericAndBranch(HCompareNumericAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCompareHoleAndBranch(HCompareHoleAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCompareGeneric(HCompareGeneric* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCompareMinusZeroAndBranch(HCompareMinusZeroAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCompareObjectEqAndBranch(HCompareObjectEqAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoCompareMap(HCompareMap* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoConstructDouble(HConstructDouble* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoDateField(HDateField* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoDebugBreak(HDebugBreak* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoDeclareGlobals(HDeclareGlobals* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoDeoptimize(HDeoptimize* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoDiv(HDiv* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoDoubleBits(HDoubleBits* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoDummyUse(HDummyUse* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoEnterInlined(HEnterInlined* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoEnvironmentMarker(HEnvironmentMarker* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoForceRepresentation(HForceRepresentation* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoForInCacheArray(HForInCacheArray* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoForInPrepareMap(HForInPrepareMap* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoFunctionLiteral(HFunctionLiteral* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoGetCachedArrayIndex(HGetCachedArrayIndex* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoHasCachedArrayIndexAndBranch(HHasCachedArrayIndexAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoHasInstanceTypeAndBranch(HHasInstanceTypeAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoInnerAllocatedObject(HInnerAllocatedObject* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoInstanceOf(HInstanceOf* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoInstanceOfKnownGlobal(HInstanceOfKnownGlobal* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoInvokeFunction(HInvokeFunction* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoIsConstructCallAndBranch(HIsConstructCallAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoIsObjectAndBranch(HIsObjectAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoIsStringAndBranch(HIsStringAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoIsSmiAndBranch(HIsSmiAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoIsUndetectableAndBranch(HIsUndetectableAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoLeaveInlined(HLeaveInlined* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoLoadContextSlot(HLoadContextSlot* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoLoadFieldByIndex(HLoadFieldByIndex* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoLoadFunctionPrototype(HLoadFunctionPrototype* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoLoadGlobalCell(HLoadGlobalCell* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoLoadGlobalGeneric(HLoadGlobalGeneric* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoLoadKeyed(HLoadKeyed* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoLoadKeyedGeneric(HLoadKeyedGeneric* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoLoadNamedField(HLoadNamedField* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoLoadNamedGeneric(HLoadNamedGeneric* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoLoadRoot(HLoadRoot* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoMapEnumLength(HMapEnumLength* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoMathFloorOfDiv(HMathFloorOfDiv* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoMathMinMax(HMathMinMax* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoMod(HMod* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoMul(HMul* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoOsrEntry(HOsrEntry* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoPower(HPower* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoPushArguments(HPushArguments* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoRegExpLiteral(HRegExpLiteral* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoRor(HRor* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoSar(HSar* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoSeqStringGetChar(HSeqStringGetChar* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoSeqStringSetChar(HSeqStringSetChar* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoShl(HShl* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoShr(HShr* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoStoreCodeEntry(HStoreCodeEntry* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoStoreContextSlot(HStoreContextSlot* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoStoreFrameContext(HStoreFrameContext* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoStoreGlobalCell(HStoreGlobalCell* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoStoreKeyed(HStoreKeyed* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoStoreKeyedGeneric(HStoreKeyedGeneric* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoStoreNamedField(HStoreNamedField* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoStoreNamedGeneric(HStoreNamedGeneric* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoStringAdd(HStringAdd* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoStringCharCodeAt(HStringCharCodeAt* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoStringCharFromCode(HStringCharFromCode* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoStringCompareAndBranch(HStringCompareAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoSub(HSub* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoTailCallThroughMegamorphicCache(HTailCallThroughMegamorphicCache* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoThisFunction(HThisFunction* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoToFastProperties(HToFastProperties* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoTransitionElementsKind(HTransitionElementsKind* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoTrapAllocationMemento(HTrapAllocationMemento* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoTypeof(HTypeof* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoTypeofIsAndBranch(HTypeofIsAndBranch* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoUnaryMathOperation(HUnaryMathOperation* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoUnknownOSRValue(HUnknownOSRValue* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoUseConst(HUseConst* instr) {
  UNIMPLEMENTED();
}

void LLLVMChunkBuilder::DoWrapReceiver(HWrapReceiver* instr) {
  UNIMPLEMENTED();
}

}  // namespace internal
}  // namespace v8
