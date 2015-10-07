// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "pass-rewrite-safepoints.h"

#include "src/base/macros.h"
#include <map>

namespace v8 {
namespace internal {

// FunctionPasses may overload three virtual methods to do their work.
// All of these methods should return true if they modified the program,
// or false if they didn’t.

class RewriteSafepointsPass : public llvm::FunctionPass {
 public:
  RewriteSafepointsPass(std::set<llvm::Value*>&);
  bool runOnFunction(llvm::Function& function) override;
  void getAnalysisUsage(llvm::AnalysisUsage& analysis_usage) const override;

//  bool doInitialization(Module& module) override { return false; };
  static char ID;

 private:
  ValueSet& gc_collected_pointers_;
};

char RewriteSafepointsPass::ID = 0;

RewriteSafepointsPass::RewriteSafepointsPass(ValueSet& pointers)
    : FunctionPass(ID),
      gc_collected_pointers_(pointers) {}

static void computeLiveInValues(llvm::BasicBlock::reverse_iterator rbegin,
                                llvm::BasicBlock::reverse_iterator rend,
                                llvm::DenseSet<llvm::Value*>& LiveTmp,
                                ValueSet& gc_collected_pointers) {

  for (llvm::BasicBlock::reverse_iterator ritr = rbegin; ritr != rend; ritr++) {
    llvm::Instruction* I = &*ritr;

    // KILL/Def - Remove this definition from LiveIn
    LiveTmp.erase(I);

    // Don't consider *uses* in PHI nodes, we handle their contribution to
    // predecessor blocks when we seed the LiveOut sets
    if (llvm::isa<llvm::PHINode>(I))
      continue;

    // USE - Add to the LiveIn set for this instruction
    for (llvm::Value* V : I->operands()) {
      if (gc_collected_pointers.count(V))
        LiveTmp.insert(V);
    }
  }
}

static void findLiveSetAtInst(llvm::Instruction* inst,
                              GCPtrLivenessData& data,
                              StatepointLiveSetTy& out_liveset,
                              ValueSet& gc_collected_pointers) {

  llvm::BasicBlock* block = inst->getParent();

  // Note: The copy is intentional and required
  assert(data.LiveOut.count(block));
  llvm::DenseSet<llvm::Value*> LiveOut = data.LiveOut[block];

  // We want to handle the statepoint itself oddly.  It's
  // call result is not live (normal), nor are it's arguments
  // (unless they're used again later).  This adjustment is
  // specifically what we need to relocate
  llvm::BasicBlock::reverse_iterator rend(inst);
  computeLiveInValues(block->rbegin(), rend, LiveOut, gc_collected_pointers);
  LiveOut.erase(inst);
  out_liveset.insert(LiveOut.begin(), LiveOut.end());
}

static void computeLiveOutSeed(llvm::BasicBlock* BB,
                               llvm::DenseSet<llvm::Value*>& LiveTmp,
                               ValueSet& gc_collected_pointers) {
  for (llvm::BasicBlock* Succ : successors(BB)) {
    const llvm::BasicBlock::iterator E(Succ->getFirstNonPHI());
    for (llvm::BasicBlock::iterator I = Succ->begin(); I != E; I++) {
      llvm::PHINode *Phi = llvm::cast<llvm::PHINode>(&*I);
      llvm::Value *V = Phi->getIncomingValueForBlock(BB);
      if (gc_collected_pointers.count(V))
        LiveTmp.insert(V);
    }
  }
}

static llvm::DenseSet<llvm::Value*> computeKillSet(
    llvm::BasicBlock *BB, ValueSet& gc_collected_pointers) {
  llvm::DenseSet<llvm::Value*> KillSet;
  for (llvm::Instruction& I : *BB)
    if (gc_collected_pointers.count(&I)) KillSet.insert(&I);
  return KillSet;
}

#ifdef DEBUG
/// Check that the items in 'Live' dominate 'TI'.  This is used as a basic
/// sanity check for the liveness computation.
static void checkBasicSSA(llvm::DominatorTree &DT,
                          llvm::DenseSet<llvm::Value*> &Live,
                          llvm::TerminatorInst *TI,
                          bool TermOkay = false) {
  for (llvm::Value* V : Live) {
    if (auto *I = llvm::dyn_cast<llvm::Instruction>(V)) {
      // The terminator can be a member of the LiveOut set.  LLVM's definition
      // of instruction dominance states that V does not dominate itself.  As
      // such, we need to special case this to allow it.
      if (TermOkay && TI == I)
        continue;
      assert(DT.dominates(I, TI) &&
             "basic SSA liveness expectation violated by liveness analysis");
    }
  }
}

/// Check that all the liveness sets used during the computation of liveness
/// obey basic SSA properties.  This is useful for finding cases where we miss
/// a def.
static void checkBasicSSA(llvm::DominatorTree &DT,
                          GCPtrLivenessData &Data,
                          llvm::BasicBlock& BB) {
  checkBasicSSA(DT, Data.LiveSet[&BB], BB.getTerminator());
  checkBasicSSA(DT, Data.LiveOut[&BB], BB.getTerminator(), true);
  checkBasicSSA(DT, Data.LiveIn[&BB], BB.getTerminator());
}
#endif

static void computeLiveInValues(llvm::DominatorTree& DT,
                                llvm::Function& F,
                                GCPtrLivenessData& Data,
                                ValueSet& gc_collected_pointers) {
  llvm::SmallSetVector<llvm::BasicBlock*, 200> Worklist;
  auto AddPredsToWorklist = [&](llvm::BasicBlock* BB) {
    // We use a SetVector so that we don't have duplicates in the worklist.
    Worklist.insert(pred_begin(BB), pred_end(BB));
  };
  auto NextItem = [&]() {
    llvm::BasicBlock* BB = Worklist.back();
    Worklist.pop_back();
    return BB;
  };

  // Seed the liveness for each individual block
  for (llvm::BasicBlock& BB : F) {
    Data.KillSet[&BB] = computeKillSet(&BB, gc_collected_pointers);
    Data.LiveSet[&BB].clear();
    computeLiveInValues(BB.rbegin(), BB.rend(), Data.LiveSet[&BB],
                        gc_collected_pointers);

#ifdef DEBUG
    for (llvm::Value* Kill : Data.KillSet[&BB])
      assert(!Data.LiveSet[&BB].count(Kill) && "live set contains kill");
#endif

    Data.LiveOut[&BB] = llvm::DenseSet<llvm::Value*>();
    computeLiveOutSeed(&BB, Data.LiveOut[&BB], gc_collected_pointers);
    Data.LiveIn[&BB] = Data.LiveSet[&BB];
    set_union(Data.LiveIn[&BB], Data.LiveOut[&BB]);
    set_subtract(Data.LiveIn[&BB], Data.KillSet[&BB]);
    if (!Data.LiveIn[&BB].empty())
      AddPredsToWorklist(&BB);
  }

  // Propagate that liveness until stable
  while (!Worklist.empty()) {
    llvm::BasicBlock *BB = NextItem();

    // Compute our new liveout set, then exit early if it hasn't changed
    // despite the contribution of our successor.
    llvm::DenseSet<llvm::Value*> LiveOut = Data.LiveOut[BB];
    const auto OldLiveOutSize = LiveOut.size();
    for (llvm::BasicBlock *Succ : successors(BB)) {
      assert(Data.LiveIn.count(Succ));
      set_union(LiveOut, Data.LiveIn[Succ]);
    }
    // assert OutLiveOut is a subset of LiveOut
    if (OldLiveOutSize == LiveOut.size()) {
      // If the sets are the same size, then we didn't actually add anything
      // when unioning our successors LiveIn  Thus, the LiveIn of this block
      // hasn't changed.
      continue;
    }
    Data.LiveOut[BB] = LiveOut;

    // Apply the effects of this basic block
    llvm::DenseSet<llvm::Value*> LiveTmp = LiveOut;
    set_union(LiveTmp, Data.LiveSet[BB]);
    set_subtract(LiveTmp, Data.KillSet[BB]);

    assert(Data.LiveIn.count(BB));
    const llvm::DenseSet<llvm::Value *> &OldLiveIn = Data.LiveIn[BB];
    // assert: OldLiveIn is a subset of LiveTmp
    if (OldLiveIn.size() != LiveTmp.size()) {
      Data.LiveIn[BB] = LiveTmp;
      AddPredsToWorklist(BB);
    }
  } // while( !worklist.empty() )

#ifdef DEBUG
  // Sanity check our output against SSA properties.  This helps catch any
  // missing kills during the above iteration.
  for (llvm::BasicBlock &BB : F) {
    checkBasicSSA(DT, Data, BB);
  }
#endif
}

// Conservatively identifies any definitions which might be live at the
// given instruction. The  analysis is performed immediately before the
// given instruction. Values defined by that instruction are not considered
// live.  Values used by that instruction are considered live.
static void AnalyzeParsePointLiveness(
    llvm::DominatorTree& DT, GCPtrLivenessData& OriginalLivenessData,
    const llvm::CallSite& CS, PartiallyConstructedSafepointRecord& result,
    ValueSet& gc_collected_pointers) {

  llvm::Instruction *inst = CS.getInstruction();

  StatepointLiveSetTy liveset;
  findLiveSetAtInst(inst, OriginalLivenessData, liveset, gc_collected_pointers);

//  if (PrintLiveSet) {
//    // Note: This output is used by several of the test cases
//    // The order of elements in a set is not stable, put them in a vec and sort
//    // by name
//    SmallVector<Value *, 64> Temp;
//    Temp.insert(Temp.end(), liveset.begin(), liveset.end());
//    std::sort(Temp.begin(), Temp.end(), order_by_name);
//    errs() << "Live Variables:\n";
//    for (Value *V : Temp)
//      dbgs() << " " << V->getName() << " " << *V << "\n";
//  }
//  if (PrintLiveSetSize) {
//    errs() << "Safepoint For: " << CS.getCalledValue()->getName() << "\n";
//    errs() << "Number live values: " << liveset.size() << "\n";
//  }
  result.liveset = liveset;
}

static void FindLiveReferences(
    llvm::Function &function, llvm::DominatorTree& dom_tree,
    llvm::ArrayRef<llvm::CallSite> to_update,
    llvm::MutableArrayRef<struct PartiallyConstructedSafepointRecord> records,
    ValueSet& gc_collected_pointers) {

  GCPtrLivenessData original_liveness_data;
  computeLiveInValues(dom_tree, function, original_liveness_data,
                      gc_collected_pointers);
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    const llvm::CallSite &CS = to_update[i];
    AnalyzeParsePointLiveness(dom_tree, original_liveness_data, CS, info,
                              gc_collected_pointers);
  }
}

// Create new attribute set containing only attributes which can be transferred
// from original call to the safepoint.
static llvm::AttributeSet legalizeCallAttributes(llvm::AttributeSet AS) {
  llvm::AttributeSet ret;

  for (unsigned Slot = 0; Slot < AS.getNumSlots(); Slot++) {
    unsigned index = AS.getSlotIndex(Slot);

    if (index == llvm::AttributeSet::ReturnIndex ||
        index == llvm::AttributeSet::FunctionIndex) {

      for (auto it = AS.begin(Slot), it_end = AS.end(Slot); it != it_end;
           ++it) {
        llvm::Attribute attr = *it;

        // Do not allow certain attributes - just skip them
        // Safepoint can not be read only or read none.
        if (attr.hasAttribute(llvm::Attribute::ReadNone) ||
            attr.hasAttribute(llvm::Attribute::ReadOnly))
          continue;

        auto attr_set = llvm::AttributeSet::get(AS.getContext(), index,
                                                llvm::AttrBuilder(attr));
        ret = ret.addAttributes(AS.getContext(), index, attr_set);
      }
    }

    // Just skip parameter attributes for now
  }

  return ret;
}

// Add new statepoint with deopt parameters right after the original statepoint
// and remove it (the original).
static void makeStatepointExplicit(const llvm::CallSite& CS, // to replace
                                   PartiallyConstructedSafepointRecord& info) {
  DCHECK(llvm::isStatepoint(CS));

  auto live_set = info.liveset;

  llvm::BasicBlock* block = CS.getInstruction()->getParent();
  DCHECK(block);
  llvm::Function* function = block->getParent();
  DCHECK(function);
  llvm::Module* module = function->getParent();
  DCHECK(module);
  USE(module);

  // We're not changing the function signature of the statepoint since the gc
  // arguments go into the var args section.
  llvm::Function* gc_statepoint_decl = CS.getCalledFunction();

  // Then go ahead and use the builder do actually do the inserts.  We insert
  // immediately before the previous instruction under the assumption that all
  // arguments will be available here.  We can't insert afterwards since we may
  // be replacing a terminator.
  llvm::Instruction* insertBefore = CS.getInstruction();
  llvm::IRBuilder<> Builder(insertBefore);

  // Copy all of the arguments from the original statepoint - this includes the
  // target, call args, and transition args (which we don't have).
  llvm::SmallVector<llvm::Value*, 64> args;
  args.insert(args.end(), CS.arg_begin(), CS.arg_end());
  llvm::Value* last = args.pop_back_val();
  // Actually, it's not just a constant, it's a zero meaning
  // there are no deopt parameters. TODO(llvm): check for int zero here.
  DCHECK(llvm::isa<llvm::Constant>(last));
  args.push_back(Builder.getInt32(live_set.size())); // Why not int64?
  args.insert(args.end(), live_set.begin(), live_set.end());

  // Create the statepoint given all the arguments
  llvm::Instruction* token = nullptr;
  llvm::AttributeSet return_attributes;
  {
    DCHECK(CS.isCall()); // We don't need invoke and thus do not support it.

    llvm::CallInst *toReplace = llvm::cast<llvm::CallInst>(CS.getInstruction());
    llvm::CallInst *call =
        Builder.CreateCall(gc_statepoint_decl, args, "safepoint_token");
    call->setTailCall(toReplace->isTailCall());
    call->setCallingConv(toReplace->getCallingConv());

    // Currently we will fail on parameter attributes and on certain
    // function attributes.
    llvm::AttributeSet old_attrs = toReplace->getAttributes();
    llvm::AttributeSet new_attrs = legalizeCallAttributes(old_attrs);
    // In case if we can handle this set of attributes - set up function attrs
    // directly on statepoint and return attrs later for gc_result intrinsic.
    call->setAttributes(new_attrs.getFnAttributes());
    return_attributes = new_attrs.getRetAttributes();

    token = call;

    // Put the following gc_result and gc_relocate calls immediately after the
    // the old call (which we're about to delete)
    llvm::BasicBlock::iterator next(toReplace);
    DCHECK(block->end() != next); // Must have next.
    next++;
    llvm::Instruction* IP = &*(next);
    Builder.SetInsertPoint(IP);
    Builder.SetCurrentDebugLocation(IP->getDebugLoc());
  }
  DCHECK(token);

  // Take the name of the original value call if it had one.
  token->takeName(CS.getInstruction());

  // Update the gc.result of the original statepoint (if any) to use the newly
  // inserted statepoint.  This is safe to do here since the token can't be
  // considered a live reference.
  CS.getInstruction()->replaceAllUsesWith(token);

  info.StatepointToken = token;

  CS.getInstruction()->eraseFromParent();
}

static bool InsertParsePoints(
    llvm::Function& function, llvm::DominatorTree& dom_tree,
    llvm::SmallVectorImpl<llvm::CallSite>& to_update,
    ValueSet& gc_collected_pointers) {

  for (llvm::CallSite callsite : to_update) {
    // We generate only calls, so we expect nothing but calls.
    DCHECK(!callsite.isInvoke());
    auto vm_state_iterator = llvm::Statepoint(callsite).vm_state_args();
    // We expect 'deopt args' of the safepoint instructions to be empty. Because
    // if it wasn't we'd have to ensure their liveness across the safepoint.
    DCHECK(vm_state_iterator.begin() == vm_state_iterator.end());
  }

  llvm::SmallVector<struct PartiallyConstructedSafepointRecord, 64> records;
  records.reserve(to_update.size());
  for (size_t i = 0; i < to_update.size(); i++) {
    struct PartiallyConstructedSafepointRecord info;
    records.push_back(info);
  }

  FindLiveReferences(function, dom_tree, to_update, records, gc_collected_pointers);

  //update the safepoint instructions

  // Now run through and replace the existing statepoints with new ones with
  // the live variables listed.  We do not yet update uses of the values being
  // relocated. We have references to live variables that need to
  // survive to the last iteration of this loop.  (By construction, the
  // previous statepoint can not be a live variable, thus we can and remove
  // the old statepoint calls as we go.)
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord& info = records[i];
    llvm::CallSite& callsite = to_update[i];
    makeStatepointExplicit(callsite, info);
  }
  to_update.clear(); // prevent accident use of invalid CallSites
  return !records.empty();
}

bool RewriteSafepointsPass::runOnFunction(llvm::Function& function) {
  // FIXME(llvm):
  // 1. First make sure there are no unreachable safepoints.
  // We should either run DCE before this pass or delete them ourselves
  // (as it is done in RewriteStatepointsForGC).
  // 2. Same for single entry phi nodes.

  llvm::DominatorTree& dom_tree =
      getAnalysis<llvm::DominatorTreeWrapperPass>(function).getDomTree();

  // Gather all safepoints [which need to be rewritten].
  llvm::SmallVector<llvm::CallSite, 64> parse_point_needed;
  for (llvm::Instruction& instr : llvm::instructions(function)) {
    DCHECK(dom_tree.isReachableFromEntry(instr.getParent()));
    parse_point_needed.push_back(llvm::CallSite(&instr));
  }

  return InsertParsePoints(function, dom_tree, parse_point_needed,
                           gc_collected_pointers_);
}

void RewriteSafepointsPass::getAnalysisUsage(
    llvm::AnalysisUsage& analysis_usage) const {
  analysis_usage.addRequired<llvm::DominatorTreeWrapperPass>();
//  analysis_usage.addRequired<llvm::TargetTransformInfoWrapperPass>();
}

llvm::FunctionPass* createRewriteSafepointsPass(ValueSet& pointers) {
  llvm::initializeDominatorTreeWrapperPassPass(
      *llvm::PassRegistry::getPassRegistry());
  return new RewriteSafepointsPass(pointers);
}

} } // v8::internal
