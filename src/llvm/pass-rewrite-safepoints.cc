// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//===----------------------------------------------------------------------===//
//
// Rewrite an existing set of gc.statepoints such that they make potential
// relocations performed by the garbage collector explicit in the IR.
//
//===----------------------------------------------------------------------===//

// TODO(llvm): edit the LICENSE file so that it is clear this file is derived
// from the LLVM source.

#include "llvm-chunk.h" // TODO(llvm): we only use IntHelper from here (move it)
#include "pass-rewrite-safepoints.h"

#include "src/base/macros.h"
#include <map>


using v8::internal::IntHelper;
using v8::internal::ValueSet;

using namespace llvm;

static bool ClobberNonLive = false;

#ifdef DEBUG
// Print the liveset found at the insert location
static bool PrintLiveSet = true;
static bool PrintLiveSetSize = true;
#else
static bool PrintLiveSet = false;
static bool PrintLiveSetSize = false;
#endif




namespace {

struct RewriteStatepointsForGC : public ModulePass {
  static char ID; // Pass identification, replacement for typeid

  RewriteStatepointsForGC(ValueSet& pointers)
    : ModulePass(ID),
      gc_collected_pointers_(pointers) {
    initializeDominatorTreeWrapperPassPass(*PassRegistry::getPassRegistry());
    initializeTargetTransformInfoWrapperPassPass(*PassRegistry::getPassRegistry());
  }
  bool runOnFunction(Function &F);
  bool runOnModule(Module &M) override {
    bool Changed = false;
    for (Function &F : M)
      Changed |= runOnFunction(F);

    if (Changed) {
      // stripDereferenceabilityInfo asserts that shouldRewriteStatepointsIn
      // returns true for at least one function in the module.  Since at least
      // one function changed, we know that the precondition is satisfied.
      stripDereferenceabilityInfo(M);
    }

    return Changed;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // We add and rewrite a bunch of instructions, but don't really do much
    // else.  We could in theory preserve a lot more analyses here.
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }

  /// The IR fed into RewriteStatepointsForGC may have had attributes implying
  /// dereferenceability that are no longer valid/correct after
  /// RewriteStatepointsForGC has run.  This is because semantically, after
  /// RewriteStatepointsForGC runs, all calls to gc.statepoint "free" the entire
  /// heap.  stripDereferenceabilityInfo (conservatively) restores correctness
  /// by erasing all attributes in the module that externally imply
  /// dereferenceability.
  ///
  void stripDereferenceabilityInfo(Module &M);

  // Helpers for stripDereferenceabilityInfo
  void stripDereferenceabilityInfoFromBody(Function &F);
  void stripDereferenceabilityInfoFromPrototype(Function &F);

 private:
  ValueSet& gc_collected_pointers_;
};
} // namespace

namespace v8 {
namespace internal {

llvm::ModulePass* createRewriteStatepointsForGCPass(ValueSet& pointers) {
  return new RewriteStatepointsForGC(pointers);
}

} } // v8::internal

char RewriteStatepointsForGC::ID = 0;

namespace {
struct GCPtrLivenessData {
  /// Values defined in this block.
  DenseMap<BasicBlock *, DenseSet<Value *>> KillSet;
  /// Values used in this block (and thus live); does not included values
  /// killed within this block.
  DenseMap<BasicBlock *, DenseSet<Value *>> LiveSet;

  /// Values live into this basic block (i.e. used by any
  /// instruction in this basic block or ones reachable from here)
  DenseMap<BasicBlock *, DenseSet<Value *>> LiveIn;

  /// Values live out of this basic block (i.e. live into
  /// any successor block)
  DenseMap<BasicBlock *, DenseSet<Value *>> LiveOut;
};

// The type of the internal cache used inside the findBasePointers family
// of functions.  From the callers perspective, this is an opaque type and
// should not be inspected.
//
// In the actual implementation this caches two relations:
// - The base relation itself (i.e. this pointer is based on that one)
// - The base defining value relation (i.e. before base_phi insertion)
// Generally, after the execution of a full findBasePointer call, only the
// base relation will remain.  Internally, we add a mixture of the two
// types, then update all the second type to the first type
typedef DenseMap<Value *, Value *> DefiningValueMapTy;
typedef DenseSet<llvm::Value *> StatepointLiveSetTy;
typedef DenseMap<Instruction *, Value *> RematerializedValueMapTy;

struct PartiallyConstructedSafepointRecord {
  /// The set of values known to be live across this safepoint
  StatepointLiveSetTy liveset;

  /// The *new* gc.statepoint instruction itself.  This produces the token
  /// that normal path gc.relocates and the gc.result are tied to.
  Instruction *StatepointToken;

  /// Instruction to which exceptional gc relocates are attached
  /// Makes it easier to iterate through them during relocationViaAlloca.
  Instruction *UnwindToken;

  /// Record live values we are rematerialized instead of relocating.
  /// They are not included into 'liveset' field.
  /// Maps rematerialized copy to it's original value.
  RematerializedValueMapTy RematerializedValues;
};
}

/// Compute the live-in set for every basic block in the function
static void computeLiveInValues(DominatorTree &DT, Function &F,
                                GCPtrLivenessData &Data,
                                ValueSet& gc_collected_pointers);

/// Given results from the dataflow liveness computation, find the set of live
/// Values at a particular instruction.
static void findLiveSetAtInst(Instruction *inst, GCPtrLivenessData &Data,
                              StatepointLiveSetTy &out,
                              ValueSet& gc_collected_pointers);

// TODO: Once we can get to the GCStrategy, this becomes
// Optional<bool> isGCManagedPointer(const Value *V) const override {

static bool isGCPointerType(Type *T) {
  if (auto *PT = dyn_cast<PointerType>(T))
    // For the sake of this example GC, we arbitrarily pick addrspace(1) as our
    // GC managed heap.  We know that a pointer into this heap needs to be
    // updated and that no other pointer does.
    return (1 == PT->getAddressSpace());
  return false;
}

// Return true if this type is one which a) is a gc pointer or contains a GC
// pointer and b) is of a type this code expects to encounter as a live value.
// (The insertion code will DCHECK that a type which matches (a) and not (b)
// is not encountered.)
static bool isHandledGCPointerType(Type *T) {
  // We fully support gc pointers
  if (isGCPointerType(T))
    return true;
  // We partially support vectors of gc pointers. The code will DCHECK if it
  // can't handle something.
  if (auto VT = dyn_cast<VectorType>(T))
    if (isGCPointerType(VT->getElementType()))
      return true;
  return false;
}

static bool order_by_name(llvm::Value *a, llvm::Value *b) {
  if (a->hasName() && b->hasName()) {
    return -1 == a->getName().compare(b->getName());
  } else if (a->hasName() && !b->hasName()) {
    return true;
  } else if (!a->hasName() && b->hasName()) {
    return false;
  } else {
    // Better than nothing, but not stable
    return a < b;
  }
}

// Return the name of the value suffixed with the provided value, or if the
// value didn't have a name, the default value specified.
static std::string suffixed_name_or(Value *V, StringRef Suffix,
                                    StringRef DefaultName) {
  return V->hasName() ? (V->getName() + Suffix).str() : DefaultName.str();
}

// Conservatively identifies any definitions which might be live at the
// given instruction. The  analysis is performed immediately before the
// given instruction. Values defined by that instruction are not considered
// live.  Values used by that instruction are considered live.
static void analyzeParsePointLiveness(
    DominatorTree &DT, GCPtrLivenessData &OriginalLivenessData,
    const CallSite &CS, PartiallyConstructedSafepointRecord &result,
    ValueSet& gc_collected_pointers) {
  Instruction *inst = CS.getInstruction();

  StatepointLiveSetTy liveset;
  findLiveSetAtInst(inst, OriginalLivenessData, liveset, gc_collected_pointers);

  if (PrintLiveSet) {
    // Note: This output is used by several of the test cases
    // The order of elements in a set is not stable, put them in a vec and sort
    // by name
    SmallVector<Value *, 64> Temp;
    Temp.insert(Temp.end(), liveset.begin(), liveset.end());
    std::sort(Temp.begin(), Temp.end(), order_by_name);
    errs() << "Live Variables:\n";
    for (Value *V : Temp)
      dbgs() << " " << V->getName() << " " << *V << "\n";
  }
  if (PrintLiveSetSize) {
    errs() << "Safepoint For: " << CS.getCalledValue()->getName() << "\n";
    errs() << "Number live values: " << liveset.size() << "\n";
  }
  result.liveset = liveset;
}

static bool isKnownBaseResult(Value *V);
namespace {
/// A single base defining value - An immediate base defining value for an
/// instruction 'Def' is an input to 'Def' whose base is also a base of 'Def'.
/// For instructions which have multiple pointer [vector] inputs or that
/// transition between vector and scalar types, there is no immediate base
/// defining value.  The 'base defining value' for 'Def' is the transitive
/// closure of this relation stopping at the first instruction which has no
/// immediate base defining value.  The b.d.v. might itself be a base pointer,
/// but it can also be an arbitrary derived pointer.
struct BaseDefiningValueResult {
  /// Contains the value which is the base defining value.
  Value * const BDV;
  /// True if the base defining value is also known to be an actual base
  /// pointer.
  const bool IsKnownBase;
  BaseDefiningValueResult(Value *BDV, bool IsKnownBase)
    : BDV(BDV), IsKnownBase(IsKnownBase) {
#ifndef NDEBUG
    // Check consistency between new and old means of checking whether a BDV is
    // a base.
    bool MustBeBase = isKnownBaseResult(BDV);
    USE(MustBeBase);
    DCHECK(!MustBeBase || MustBeBase == IsKnownBase);
#endif
  }
};
}

/// Given the result of a call to findBaseDefiningValue, or findBaseOrBDV,
/// is it known to be a base pointer?  Or do we need to continue searching.
static bool isKnownBaseResult(Value *V) {
  if (!isa<PHINode>(V) && !isa<SelectInst>(V) &&
      !isa<ExtractElementInst>(V) && !isa<InsertElementInst>(V) &&
      !isa<ShuffleVectorInst>(V)) {
    // no recursion possible
    return true;
  }
  if (isa<Instruction>(V) &&
      cast<Instruction>(V)->getMetadata("is_base_value")) {
    // This is a previously inserted base phi or select.  We know
    // that this is a base value.
    return true;
  }

  // We need to keep searching
  return false;
}

namespace {
/// Models the state of a single base defining value in the findBasePointer
/// algorithm for determining where a new instruction is needed to propagate
/// the base of this BDV.
class BDVState {
public:
  enum Status { Unknown, Base, Conflict };

  BDVState(Status s, Value *b = nullptr) : status(s), base(b) {
    DCHECK(status != Base || b);
  }
  explicit BDVState(Value *b) : status(Base), base(b) {}
  BDVState() : status(Unknown), base(nullptr) {}

  Status getStatus() const { return status; }
  Value *getBase() const { return base; }

  bool isBase() const { return getStatus() == Base; }
  bool isUnknown() const { return getStatus() == Unknown; }
  bool isConflict() const { return getStatus() == Conflict; }

  bool operator==(const BDVState &other) const {
    return base == other.base && status == other.status;
  }

  bool operator!=(const BDVState &other) const { return !(*this == other); }

  LLVM_DUMP_METHOD
  void dump() const { print(dbgs()); dbgs() << '\n'; }

  void print(raw_ostream &OS) const {
    switch (status) {
    case Unknown:
      OS << "U";
      break;
    case Base:
      OS << "B";
      break;
    case Conflict:
      OS << "C";
      break;
    };
    OS << " (" << base << " - "
       << (base ? base->getName() : "nullptr") << "): ";
  }

private:
  Status status;
  Value *base; // non null only if status == base
};
}

#ifndef NDEBUG
static raw_ostream &operator<<(raw_ostream &OS, const BDVState &State) {
  State.print(OS);
  return OS;
}
#endif

namespace {
// Values of type BDVState form a lattice, and this is a helper
// class that implementes the meet operation.  The meat of the meet
// operation is implemented in MeetBDVStates::pureMeet
class MeetBDVStates {
public:
  /// Initializes the currentResult to the TOP state so that if can be met with
  /// any other state to produce that state.
  MeetBDVStates() {}

  // Destructively meet the current result with the given BDVState
  void meetWith(BDVState otherState) {
    currentResult = meet(otherState, currentResult);
  }

  BDVState getResult() const { return currentResult; }

private:
  BDVState currentResult;

  /// Perform a meet operation on two elements of the BDVState lattice.
  static BDVState meet(BDVState LHS, BDVState RHS) {
    DCHECK((pureMeet(LHS, RHS) == pureMeet(RHS, LHS)) &&
           "math is wrong: meet does not commute!");
    BDVState Result = pureMeet(LHS, RHS);
#ifdef DEBUG
    dbgs() << "meet of " << LHS << " with " << RHS << " produced "
        << Result << "\n";
#endif
    return Result;
  }

  static BDVState pureMeet(const BDVState &stateA, const BDVState &stateB) {
    switch (stateA.getStatus()) {
    case BDVState::Unknown:
      return stateB;

    case BDVState::Base:
      DCHECK(stateA.getBase() && "can't be null");
      if (stateB.isUnknown())
        return stateA;

      if (stateB.isBase()) {
        if (stateA.getBase() == stateB.getBase()) {
          DCHECK(stateA == stateB && "equality broken!");
          return stateA;
        }
        return BDVState(BDVState::Conflict);
      }
      DCHECK(stateB.isConflict() && "only three states!");
      return BDVState(BDVState::Conflict);

    case BDVState::Conflict:
      return stateA;
    }
    llvm_unreachable("only three states!");
  }
};
}

/// Given an updated version of the dataflow liveness results, update the
/// liveset and base pointer maps for the call site CS.
static void recomputeLiveInValues(GCPtrLivenessData &RevisedLivenessData,
                                  const CallSite &CS,
                                  PartiallyConstructedSafepointRecord &result,
                                  ValueSet& gc_collected_pointers);

static void recomputeLiveInValues(
    Function &F, DominatorTree &DT, Pass *P, ArrayRef<CallSite> toUpdate,
    MutableArrayRef<struct PartiallyConstructedSafepointRecord> records,
    ValueSet& gc_collected_pointers) {
  // TODO-PERF: reuse the original liveness, then simply run the dataflow
  // again.  The old values are still live and will help it stabilize quickly.
  GCPtrLivenessData RevisedLivenessData;
  computeLiveInValues(DT, F, RevisedLivenessData, gc_collected_pointers);
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    const CallSite &CS = toUpdate[i];
    recomputeLiveInValues(RevisedLivenessData, CS, info, gc_collected_pointers);
  }
}

// When inserting gc.relocate calls, we need to ensure there are no uses
// of the original value between the gc.statepoint and the gc.relocate call.
// One case which can arise is a phi node starting one of the successor blocks.
// We also need to be able to insert the gc.relocates only on the path which
// goes through the statepoint.  We might need to split an edge to make this
// possible.
static BasicBlock *
normalizeForInvokeSafepoint(BasicBlock *BB, BasicBlock *InvokeParent,
                            DominatorTree &DT) {
  BasicBlock *Ret = BB;
  if (!BB->getUniquePredecessor()) {
    Ret = SplitBlockPredecessors(BB, InvokeParent, "", &DT);
  }

  // Now that 'ret' has unique predecessor we can safely remove all phi nodes
  // from it
  FoldSingleEntryPHINodes(Ret);
  DCHECK(!isa<PHINode>(Ret->begin()));

  // At this point, we can safely insert a gc.relocate as the first instruction
  // in Ret if needed.
  return Ret;
}

static size_t find_index(ArrayRef<Value *> livevec, Value *val) {
  auto itr = std::find(livevec.begin(), livevec.end(), val);
  DCHECK(livevec.end() != itr);
  size_t index = std::distance(livevec.begin(), itr);
  DCHECK(index < livevec.size());
  return index;
}

// Create new attribute set containing only attributes which can be transferred
// from original call to the safepoint.
static AttributeSet legalizeCallAttributes(AttributeSet AS) {
  AttributeSet ret;

  for (unsigned Slot = 0; Slot < AS.getNumSlots(); Slot++) {
    unsigned index = AS.getSlotIndex(Slot);

    if (index == AttributeSet::ReturnIndex ||
        index == AttributeSet::FunctionIndex) {

      for (auto it = AS.begin(Slot), it_end = AS.end(Slot); it != it_end;
           ++it) {
        Attribute attr = *it;

        // Do not allow certain attributes - just skip them
        // Safepoint can not be read only or read none.
        if (attr.hasAttribute(Attribute::ReadNone) ||
            attr.hasAttribute(Attribute::ReadOnly))
          continue;

        ret = ret.addAttributes(
            AS.getContext(), index,
            AttributeSet::get(AS.getContext(), index, AttrBuilder(attr)));
      }
    }

    // Just skip parameter attributes for now
  }

  return ret;
}

/// Helper function to place all gc relocates necessary for the given
/// statepoint.
/// Inputs:
///   liveVariables - list of variables to be relocated.
///   liveStart - index of the first live variable.
///   basePtrs - base pointers.
///   statepointToken - statepoint instruction to which relocates should be
///   bound.
///   Builder - Llvm IR builder to be used to construct new calls.
static void CreateGCRelocates(ArrayRef<llvm::Value *> LiveVariables,
                              const int LiveStart,
                              Instruction *StatepointToken,
                              IRBuilder<> Builder) {
  if (LiveVariables.empty())
    return;

  Module *M = StatepointToken->getModule();
  auto AS = cast<PointerType>(LiveVariables[0]->getType())->getAddressSpace();
  // FIXME(llvm): write "Tagged"
  Type *Types[] = {Type::getInt8PtrTy(M->getContext(), AS)};
  Value *GCRelocateDecl =
    Intrinsic::getDeclaration(M, Intrinsic::experimental_gc_relocate, Types);

  for (unsigned i = 0; i < LiveVariables.size(); i++) {
    // Generate the gc.relocate call and save the result
    auto index = IntHelper::AsUInt32(
        LiveStart + find_index(LiveVariables, LiveVariables[i]));
    Value *LiveIdx = Builder.getInt32(index);
    Value *BaseIdx = LiveIdx;

    // only specify a debug name if we can give a useful one
    CallInst *Reloc = Builder.CreateCall(
        GCRelocateDecl, {StatepointToken, BaseIdx, LiveIdx},
        suffixed_name_or(LiveVariables[i], ".relocated", ""));
    // Trick CodeGen into thinking there are lots of free registers at this
    // fake call.
    Reloc->setCallingConv(CallingConv::Cold);
  }
}

static void
makeStatepointExplicitImpl(const CallSite &CS, /* to replace */
                           const SmallVectorImpl<llvm::Value *> &liveVariables,
                           Pass *P,
                           PartiallyConstructedSafepointRecord &result) {
  DCHECK(isStatepoint(CS) &&
         "This method expects to be rewriting a statepoint");

  BasicBlock *BB = CS.getInstruction()->getParent();
  DCHECK(BB);
  Function *F = BB->getParent();
  DCHECK(F && "must be set");
  Module *M = F->getParent();
  (void)M;
  DCHECK(M && "must be set");

  // We're not changing the function signature of the statepoint since the gc
  // arguments go into the var args section.
  Function *gc_statepoint_decl = CS.getCalledFunction();

  // Then go ahead and use the builder do actually do the inserts.  We insert
  // immediately before the previous instruction under the assumption that all
  // arguments will be available here.  We can't insert afterwards since we may
  // be replacing a terminator.
  Instruction *insertBefore = CS.getInstruction();
  IRBuilder<> Builder(insertBefore);
  // Copy all of the arguments from the original statepoint - this includes the
  // target, call args, and deopt args
  SmallVector<llvm::Value *, 64> args;
  args.insert(args.end(), CS.arg_begin(), CS.arg_end());
  // TODO: Clear the 'needs rewrite' flag

  // add all the pointers to be relocated (gc arguments)
  // Capture the start of the live variable list for use in the gc_relocates
  const int live_start = IntHelper::AsInt(args.size());
  args.insert(args.end(), liveVariables.begin(), liveVariables.end());

  // Create the statepoint given all the arguments
  Instruction *token = nullptr;
  AttributeSet return_attributes;
  if (CS.isCall()) {
    CallInst *toReplace = cast<CallInst>(CS.getInstruction());
    CallInst *call =
        Builder.CreateCall(gc_statepoint_decl, args, "safepoint_token");
    call->setTailCall(toReplace->isTailCall());
    call->setCallingConv(toReplace->getCallingConv());

    // Currently we will fail on parameter attributes and on certain
    // function attributes.
    AttributeSet new_attrs = legalizeCallAttributes(toReplace->getAttributes());
    // In case if we can handle this set of attributes - set up function attrs
    // directly on statepoint and return attrs later for gc_result intrinsic.
    call->setAttributes(new_attrs.getFnAttributes());
    return_attributes = new_attrs.getRetAttributes();

    token = call;

    // Put the following gc_result and gc_relocate calls immediately after the
    // the old call (which we're about to delete)
    BasicBlock::iterator next(toReplace);
    DCHECK(BB->end() != next && "not a terminator, must have next");
    next++;
    Instruction *IP = &*(next);
    Builder.SetInsertPoint(IP);
    Builder.SetCurrentDebugLocation(IP->getDebugLoc());

  } else {
    InvokeInst *toReplace = cast<InvokeInst>(CS.getInstruction());

    // Insert the new invoke into the old block.  We'll remove the old one in a
    // moment at which point this will become the new terminator for the
    // original block.
    InvokeInst *invoke = InvokeInst::Create(
        gc_statepoint_decl, toReplace->getNormalDest(),
        toReplace->getUnwindDest(), args, "statepoint_token", toReplace->getParent());
    invoke->setCallingConv(toReplace->getCallingConv());

    // Currently we will fail on parameter attributes and on certain
    // function attributes.
    AttributeSet new_attrs = legalizeCallAttributes(toReplace->getAttributes());
    // In case if we can handle this set of attributes - set up function attrs
    // directly on statepoint and return attrs later for gc_result intrinsic.
    invoke->setAttributes(new_attrs.getFnAttributes());
    return_attributes = new_attrs.getRetAttributes();

    token = invoke;

    // Generate gc relocates in exceptional path
    BasicBlock *unwindBlock = toReplace->getUnwindDest();
    DCHECK(!isa<PHINode>(unwindBlock->begin()) &&
           unwindBlock->getUniquePredecessor() &&
           "can't safely insert in this block!");

    Instruction *IP = &*(unwindBlock->getFirstInsertionPt());
    Builder.SetInsertPoint(IP);
    Builder.SetCurrentDebugLocation(toReplace->getDebugLoc());

    // Extract second element from landingpad return value. We will attach
    // exceptional gc relocates to it.
    const unsigned idx = 1;
    Instruction *exceptional_token =
        cast<Instruction>(Builder.CreateExtractValue(
            unwindBlock->getLandingPadInst(), idx, "relocate_token"));
    result.UnwindToken = exceptional_token;

    CreateGCRelocates(liveVariables, live_start, exceptional_token, Builder);

    // Generate gc relocates and returns for normal block
    BasicBlock *normalDest = toReplace->getNormalDest();
    DCHECK(!isa<PHINode>(normalDest->begin()) &&
           normalDest->getUniquePredecessor() &&
           "can't safely insert in this block!");

    IP = &*(normalDest->getFirstInsertionPt());
    Builder.SetInsertPoint(IP);

    // gc relocates will be generated later as if it were regular call
    // statepoint
  }
  DCHECK(token);

  // Take the name of the original value call if it had one.
  token->takeName(CS.getInstruction());

// The GCResult is already inserted, we just need to find it
#ifndef NDEBUG
  Instruction *toReplace = CS.getInstruction();
  USE(toReplace);
  DCHECK((toReplace->hasNUses(0) || toReplace->hasNUses(1)) &&
         "only valid use before rewrite is gc.result");
  DCHECK(!toReplace->hasOneUse() ||
         isGCResult(cast<Instruction>(*toReplace->user_begin())));
#endif

  // Update the gc.result of the original statepoint (if any) to use the newly
  // inserted statepoint.  This is safe to do here since the token can't be
  // considered a live reference.
  CS.getInstruction()->replaceAllUsesWith(token);

  result.StatepointToken = token;

  // Second, create a gc.relocate for every live variable
  CreateGCRelocates(liveVariables, live_start, token, Builder);
}

namespace {
struct name_ordering {
  Value *base;
  Value *derived;
  bool operator()(name_ordering const &a, name_ordering const &b) {
    return -1 == a.derived->getName().compare(b.derived->getName());
  }
};
}

// Replace an existing gc.statepoint with a new one and a set of gc.relocates
// which make the relocations happening at this safepoint explicit.
//
// WARNING: Does not do any fixup to adjust users of the original live
// values.  That's the callers responsibility.
static void
makeStatepointExplicit(DominatorTree &DT, const CallSite &CS, Pass *P,
                       PartiallyConstructedSafepointRecord &result) {
  auto liveset = result.liveset;

  // Convert to vector for efficient cross referencing.
  SmallVector<Value *, 64> livevec;
  livevec.reserve(liveset.size());
  for (Value *L : liveset) {
    livevec.push_back(L);
  }

  // Do the actual rewriting and delete the old statepoint
  makeStatepointExplicitImpl(CS, livevec, P, result);
  CS.getInstruction()->eraseFromParent();
}

// Helper function for the relocationViaAlloca.
// It receives iterator to the statepoint gc relocates and emits store to the
// assigned
// location (via allocaMap) for the each one of them.
// Add visited values into the visitedLiveValues set we will later use them
// for sanity check.
static void
insertRelocationStores(iterator_range<Value::user_iterator> GCRelocs,
                       DenseMap<Value *, Value *> &AllocaMap,
                       DenseSet<Value *> &VisitedLiveValues) {

  for (User *U : GCRelocs) {
    if (!isa<IntrinsicInst>(U))
      continue;

    IntrinsicInst *RelocatedValue = cast<IntrinsicInst>(U);

    // We only care about relocates
    if (RelocatedValue->getIntrinsicID() !=
        Intrinsic::experimental_gc_relocate) {
      continue;
    }

    GCRelocateOperands RelocateOperands(RelocatedValue);
    Value *OriginalValue =
        const_cast<Value *>(RelocateOperands.getDerivedPtr());
    DCHECK(AllocaMap.count(OriginalValue));
    Value *Alloca = AllocaMap[OriginalValue];

    // Emit store into the related alloca
    // All gc_relocate are i8 addrspace(1)* typed, and it must be bitcasted to
    // the correct type according to alloca.
    DCHECK(RelocatedValue->getNextNode() && "Should always have one since it's not a terminator");
    IRBuilder<> Builder(RelocatedValue->getNextNode());
    Value *CastedRelocatedValue =
      Builder.CreateBitCast(RelocatedValue,
                            cast<AllocaInst>(Alloca)->getAllocatedType(),
                            suffixed_name_or(RelocatedValue, ".casted", ""));

    StoreInst *Store = new StoreInst(CastedRelocatedValue, Alloca);
    Store->insertAfter(cast<Instruction>(CastedRelocatedValue));

#ifndef NDEBUG
    VisitedLiveValues.insert(OriginalValue);
#endif
  }
}

// Helper function for the "relocationViaAlloca". Similar to the
// "insertRelocationStores" but works for rematerialized values.
static void
insertRematerializationStores(
  RematerializedValueMapTy RematerializedValues,
  DenseMap<Value *, Value *> &AllocaMap,
  DenseSet<Value *> &VisitedLiveValues) {

  for (auto RematerializedValuePair: RematerializedValues) {
    Instruction *RematerializedValue = RematerializedValuePair.first;
    Value *OriginalValue = RematerializedValuePair.second;

    DCHECK(AllocaMap.count(OriginalValue) &&
           "Can not find alloca for rematerialized value");
    Value *Alloca = AllocaMap[OriginalValue];

    StoreInst *Store = new StoreInst(RematerializedValue, Alloca);
    Store->insertAfter(RematerializedValue);

#ifndef NDEBUG
    VisitedLiveValues.insert(OriginalValue);
#endif
  }
}

/// do all the relocation update via allocas and mem2reg
static void relocationViaAlloca(
    Function &F, DominatorTree &DT, ArrayRef<Value *> Live,
    ArrayRef<struct PartiallyConstructedSafepointRecord> Records) {
#ifndef NDEBUG
  // record initial number of (static) allocas; we'll check we have the same
  // number when we get done.
  int InitialAllocaNum = 0;
  for (auto I = F.getEntryBlock().begin(), E = F.getEntryBlock().end(); I != E;
       I++)
    if (isa<AllocaInst>(*I))
      InitialAllocaNum++;
#endif

  // TODO-PERF: change data structures, reserve
  DenseMap<Value *, Value *> AllocaMap;
  SmallVector<AllocaInst *, 200> PromotableAllocas;
  // Used later to chack that we have enough allocas to store all values
  std::size_t NumRematerializedValues = 0;
  PromotableAllocas.reserve(Live.size());

  // Emit alloca for "LiveValue" and record it in "allocaMap" and
  // "PromotableAllocas"
  auto emitAllocaFor = [&](Value *LiveValue) {
    AllocaInst *Alloca = new AllocaInst(LiveValue->getType(), "",
                                        F.getEntryBlock().getFirstNonPHI());
    AllocaMap[LiveValue] = Alloca;
    PromotableAllocas.push_back(Alloca);
  };

  // emit alloca for each live gc pointer
  for (unsigned i = 0; i < Live.size(); i++) {
    emitAllocaFor(Live[i]);
  }

  // emit allocas for rematerialized values
  for (size_t i = 0; i < Records.size(); i++) {
    const struct PartiallyConstructedSafepointRecord &Info = Records[i];

    for (auto RematerializedValuePair : Info.RematerializedValues) {
      Value *OriginalValue = RematerializedValuePair.second;
      if (AllocaMap.count(OriginalValue) != 0)
        continue;

      emitAllocaFor(OriginalValue);
      ++NumRematerializedValues;
    }
  }

  // The next two loops are part of the same conceptual operation.  We need to
  // insert a store to the alloca after the original def and at each
  // redefinition.  We need to insert a load before each use.  These are split
  // into distinct loops for performance reasons.

  // update gc pointer after each statepoint
  // either store a relocated value or null (if no relocated value found for
  // this gc pointer and it is not a gc_result)
  // this must happen before we update the statepoint with load of alloca
  // otherwise we lose the link between statepoint and old def
  for (size_t i = 0; i < Records.size(); i++) {
    const struct PartiallyConstructedSafepointRecord &Info = Records[i];
    Value *Statepoint = Info.StatepointToken;

    // This will be used for consistency check
    DenseSet<Value *> VisitedLiveValues;

    // Insert stores for normal statepoint gc relocates
    insertRelocationStores(Statepoint->users(), AllocaMap, VisitedLiveValues);

    // In case if it was invoke statepoint
    // we will insert stores for exceptional path gc relocates.
    if (isa<InvokeInst>(Statepoint)) {
      insertRelocationStores(Info.UnwindToken->users(), AllocaMap,
                             VisitedLiveValues);
    }

    // Do similar thing with rematerialized values
    insertRematerializationStores(Info.RematerializedValues, AllocaMap,
                                  VisitedLiveValues);

    if (ClobberNonLive) {
      // As a debugging aid, pretend that an unrelocated pointer becomes null at
      // the gc.statepoint.  This will turn some subtle GC problems into
      // slightly easier to debug SEGVs.  Note that on large IR files with
      // lots of gc.statepoints this is extremely costly both memory and time
      // wise.
      SmallVector<AllocaInst *, 64> ToClobber;
      for (auto Pair : AllocaMap) {
        Value *Def = Pair.first;
        AllocaInst *Alloca = cast<AllocaInst>(Pair.second);

        // This value was relocated
        if (VisitedLiveValues.count(Def)) {
          continue;
        }
        ToClobber.push_back(Alloca);
      }

      auto InsertClobbersAt = [&](Instruction *IP) {
        for (auto *AI : ToClobber) {
          auto AIType = cast<PointerType>(AI->getType());
          auto PT = cast<PointerType>(AIType->getElementType());
          Constant *CPN = ConstantPointerNull::get(PT);
          StoreInst *Store = new StoreInst(CPN, AI);
          Store->insertBefore(IP);
        }
      };

      // Insert the clobbering stores.  These may get intermixed with the
      // gc.results and gc.relocates, but that's fine.
      if (auto II = dyn_cast<InvokeInst>(Statepoint)) {
        InsertClobbersAt(II->getNormalDest()->getFirstInsertionPt());
        InsertClobbersAt(II->getUnwindDest()->getFirstInsertionPt());
      } else {
        BasicBlock::iterator Next(cast<CallInst>(Statepoint));
        Next++;
        InsertClobbersAt(Next);
      }
    }
  }
  // update use with load allocas and add store for gc_relocated
  for (auto Pair : AllocaMap) {
    Value *Def = Pair.first;
    Value *Alloca = Pair.second;

    // we pre-record the uses of allocas so that we dont have to worry about
    // later update
    // that change the user information.
    SmallVector<Instruction *, 20> Uses;
    // PERF: trade a linear scan for repeated reallocation
    Uses.reserve(std::distance(Def->user_begin(), Def->user_end()));
    for (User *U : Def->users()) {
      if (!isa<ConstantExpr>(U)) {
        // If the def has a ConstantExpr use, then the def is either a
        // ConstantExpr use itself or null.  In either case
        // (recursively in the first, directly in the second), the oop
        // it is ultimately dependent on is null and this particular
        // use does not need to be fixed up.
        Uses.push_back(cast<Instruction>(U));
      }
    }

    std::sort(Uses.begin(), Uses.end());
    auto Last = std::unique(Uses.begin(), Uses.end());
    Uses.erase(Last, Uses.end());

    for (Instruction *Use : Uses) {
      if (isa<PHINode>(Use)) {
        PHINode *Phi = cast<PHINode>(Use);
        for (unsigned i = 0; i < Phi->getNumIncomingValues(); i++) {
          if (Def == Phi->getIncomingValue(i)) {
            LoadInst *Load = new LoadInst(
                Alloca, "", Phi->getIncomingBlock(i)->getTerminator());
            Phi->setIncomingValue(i, Load);
          }
        }
      } else {
        LoadInst *Load = new LoadInst(Alloca, "", Use);
        Use->replaceUsesOfWith(Def, Load);
      }
    }

    // emit store for the initial gc value
    // store must be inserted after load, otherwise store will be in alloca's
    // use list and an extra load will be inserted before it
    StoreInst *Store = new StoreInst(Def, Alloca);
    if (Instruction *Inst = dyn_cast<Instruction>(Def)) {
      if (InvokeInst *Invoke = dyn_cast<InvokeInst>(Inst)) {
        // InvokeInst is a TerminatorInst so the store need to be inserted
        // into its normal destination block.
        BasicBlock *NormalDest = Invoke->getNormalDest();
        Store->insertBefore(NormalDest->getFirstNonPHI());
      } else {
        DCHECK(!Inst->isTerminator() &&
               "The only TerminatorInst that can produce a value is "
               "InvokeInst which is handled above.");
        Store->insertAfter(Inst);
      }
    } else {
      DCHECK(isa<Argument>(Def));
      Store->insertAfter(cast<Instruction>(Alloca));
    }
  }

  DCHECK(PromotableAllocas.size() == Live.size() + NumRematerializedValues &&
         "we must have the same allocas with lives");
  if (!PromotableAllocas.empty()) {
    // apply mem2reg to promote alloca to SSA
    PromoteMemToReg(PromotableAllocas, DT);
  }

#ifndef NDEBUG
  for (auto I = F.getEntryBlock().begin(), E = F.getEntryBlock().end(); I != E;
       I++)
    if (isa<AllocaInst>(*I))
      InitialAllocaNum--;
  DCHECK(InitialAllocaNum == 0 && "We must not introduce any extra allocas");
#endif
}

/// Implement a unique function which doesn't require we sort the input
/// vector.  Doing so has the effect of changing the output of a couple of
/// tests in ways which make them less useful in testing fused safepoints.
template <typename T> static void unique_unsorted(SmallVectorImpl<T> &Vec) {
  SmallSet<T, 8> Seen;
  Vec.erase(std::remove_if(Vec.begin(), Vec.end(), [&](const T &V) {
              return !Seen.insert(V).second;
            }), Vec.end());
}

/// Insert holders so that each Value is obviously live through the entire
/// lifetime of the call.
static void insertUseHolderAfter(CallSite &CS, const ArrayRef<Value *> Values,
                                 SmallVectorImpl<CallInst *> &Holders) {
  if (Values.empty())
    // No values to hold live, might as well not insert the empty holder
    return;

  Module *M = CS.getInstruction()->getParent()->getParent()->getParent();
  // Use a dummy vararg function to actually hold the values live
  Function *Func = cast<Function>(M->getOrInsertFunction(
      "__tmp_use", FunctionType::get(Type::getVoidTy(M->getContext()), true)));
  if (CS.isCall()) {
    // For call safepoints insert dummy calls right after safepoint
    BasicBlock::iterator Next(CS.getInstruction());
    Next++;
    Holders.push_back(CallInst::Create(Func, Values, "", Next));
    return;
  }
  // For invoke safepooints insert dummy calls both in normal and
  // exceptional destination blocks
  auto *II = cast<InvokeInst>(CS.getInstruction());
  Holders.push_back(CallInst::Create(
      Func, Values, "", II->getNormalDest()->getFirstInsertionPt()));
  Holders.push_back(CallInst::Create(
      Func, Values, "", II->getUnwindDest()->getFirstInsertionPt()));
}

static void findLiveReferences(
    Function &F, DominatorTree &DT, Pass *P, ArrayRef<CallSite> toUpdate,
    MutableArrayRef<struct PartiallyConstructedSafepointRecord> records,
    ValueSet& gc_collected_pointers) {
  GCPtrLivenessData OriginalLivenessData;
  computeLiveInValues(DT, F, OriginalLivenessData, gc_collected_pointers);
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    const CallSite &CS = toUpdate[i];
    analyzeParsePointLiveness(DT, OriginalLivenessData, CS, info,
                              gc_collected_pointers);
  }
}

/// Remove any vector of pointers from the liveset by scalarizing them over the
/// statepoint instruction.  Adds the scalarized pieces to the liveset.  It
/// would be preferable to include the vector in the statepoint itself, but
/// the lowering code currently does not handle that.  Extending it would be
/// slightly non-trivial since it requires a format change.  Given how rare
/// such cases are (for the moment?) scalarizing is an acceptable compromise.
static void splitVectorValues(Instruction *StatepointInst,
                              StatepointLiveSetTy &LiveSet,
                              DominatorTree &DT) {
  SmallVector<Value *, 16> ToSplit;
  for (Value *V : LiveSet)
    if (isa<VectorType>(V->getType()))
      ToSplit.push_back(V);

  if (ToSplit.empty())
    return;

  DenseMap<Value *, SmallVector<Value *, 16>> ElementMapping;

  Function &F = *(StatepointInst->getParent()->getParent());

  DenseMap<Value *, AllocaInst *> AllocaMap;
  // First is normal return, second is exceptional return (invoke only)
  DenseMap<Value *, std::pair<Value *, Value *>> Replacements;
  for (Value *V : ToSplit) {
    AllocaInst *Alloca =
        new AllocaInst(V->getType(), "", F.getEntryBlock().getFirstNonPHI());
    AllocaMap[V] = Alloca;

    VectorType *VT = cast<VectorType>(V->getType());
    IRBuilder<> Builder(StatepointInst);
    SmallVector<Value *, 16> Elements;
    for (unsigned i = 0; i < VT->getNumElements(); i++)
      Elements.push_back(Builder.CreateExtractElement(V, Builder.getInt32(i)));
    ElementMapping[V] = Elements;

    auto InsertVectorReform = [&](Instruction *IP) {
      Builder.SetInsertPoint(IP);
      Builder.SetCurrentDebugLocation(IP->getDebugLoc());
      Value *ResultVec = UndefValue::get(VT);
      for (unsigned i = 0; i < VT->getNumElements(); i++)
        ResultVec = Builder.CreateInsertElement(ResultVec, Elements[i],
                                                Builder.getInt32(i));
      return ResultVec;
    };

    if (isa<CallInst>(StatepointInst)) {
      BasicBlock::iterator Next(StatepointInst);
      Next++;
      Instruction *IP = &*(Next);
      Replacements[V].first = InsertVectorReform(IP);
      Replacements[V].second = nullptr;
    } else {
      InvokeInst *Invoke = cast<InvokeInst>(StatepointInst);
      // We've already normalized - check that we don't have shared destination
      // blocks
      BasicBlock *NormalDest = Invoke->getNormalDest();
      DCHECK(!isa<PHINode>(NormalDest->begin()));
      BasicBlock *UnwindDest = Invoke->getUnwindDest();
      DCHECK(!isa<PHINode>(UnwindDest->begin()));
      // Insert insert element sequences in both successors
      Instruction *IP = &*(NormalDest->getFirstInsertionPt());
      Replacements[V].first = InsertVectorReform(IP);
      IP = &*(UnwindDest->getFirstInsertionPt());
      Replacements[V].second = InsertVectorReform(IP);
    }
  }

  for (Value *V : ToSplit) {
    AllocaInst *Alloca = AllocaMap[V];

    // Capture all users before we start mutating use lists
    SmallVector<Instruction *, 16> Users;
    for (User *U : V->users())
      Users.push_back(cast<Instruction>(U));

    for (Instruction *I : Users) {
      if (auto Phi = dyn_cast<PHINode>(I)) {
        for (unsigned i = 0; i < Phi->getNumIncomingValues(); i++)
          if (V == Phi->getIncomingValue(i)) {
            LoadInst *Load = new LoadInst(
                Alloca, "", Phi->getIncomingBlock(i)->getTerminator());
            Phi->setIncomingValue(i, Load);
          }
      } else {
        LoadInst *Load = new LoadInst(Alloca, "", I);
        I->replaceUsesOfWith(V, Load);
      }
    }

    // Store the original value and the replacement value into the alloca
    StoreInst *Store = new StoreInst(V, Alloca);
    if (auto I = dyn_cast<Instruction>(V))
      Store->insertAfter(I);
    else
      Store->insertAfter(Alloca);

    // Normal return for invoke, or call return
    Instruction *Replacement = cast<Instruction>(Replacements[V].first);
    (new StoreInst(Replacement, Alloca))->insertAfter(Replacement);
    // Unwind return for invoke only
    Replacement = cast_or_null<Instruction>(Replacements[V].second);
    if (Replacement)
      (new StoreInst(Replacement, Alloca))->insertAfter(Replacement);
  }

  // apply mem2reg to promote alloca to SSA
  SmallVector<AllocaInst *, 16> Allocas;
  for (Value *V : ToSplit)
    Allocas.push_back(AllocaMap[V]);
  PromoteMemToReg(Allocas, DT);
}

static bool insertParsePoints(Function &F, DominatorTree &DT, Pass *P,
                              SmallVectorImpl<CallSite> &toUpdate,
                              ValueSet& gc_collected_pointers) {
#ifndef NDEBUG
  // sanity check the input
  std::set<CallSite> uniqued;
  uniqued.insert(toUpdate.begin(), toUpdate.end());
  DCHECK(uniqued.size() == toUpdate.size() && "no duplicates please!");

  for (size_t i = 0; i < toUpdate.size(); i++) {
    CallSite &CS = toUpdate[i];
    USE(CS);
    DCHECK(CS.getInstruction()->getParent()->getParent() == &F);
    DCHECK(isStatepoint(CS) && "expected to already be a deopt statepoint");
  }
#endif

  // When inserting gc.relocates for invokes, we need to be able to insert at
  // the top of the successor blocks.  See the comment on
  // normalForInvokeSafepoint on exactly what is needed.  Note that this step
  // may restructure the CFG.
  for (CallSite CS : toUpdate) {
    if (!CS.isInvoke())
      continue;
    InvokeInst *invoke = cast<InvokeInst>(CS.getInstruction());
    normalizeForInvokeSafepoint(invoke->getNormalDest(), invoke->getParent(),
                                DT);
    normalizeForInvokeSafepoint(invoke->getUnwindDest(), invoke->getParent(),
                                DT);
  }

  // A list of dummy calls added to the IR to keep various values obviously
  // live in the IR.  We'll remove all of these when done.
  SmallVector<CallInst *, 64> holders;

  // Insert a dummy call with all of the arguments to the vm_state we'll need
  // for the actual safepoint insertion.  This ensures reference arguments in
  // the deopt argument list are considered live through the safepoint (and
  // thus makes sure they get relocated.)
  for (size_t i = 0; i < toUpdate.size(); i++) {
    CallSite &CS = toUpdate[i];
    Statepoint StatepointCS(CS);

    SmallVector<Value *, 64> DeoptValues;
    for (Use &U : StatepointCS.vm_state_args()) {
      Value *Arg = cast<Value>(&U);
      if (isHandledGCPointerType(Arg->getType()))
        DeoptValues.push_back(Arg);
    }
    insertUseHolderAfter(CS, DeoptValues, holders);
  }

  SmallVector<struct PartiallyConstructedSafepointRecord, 64> records;
  records.reserve(toUpdate.size());
  for (size_t i = 0; i < toUpdate.size(); i++) {
    struct PartiallyConstructedSafepointRecord info;
    records.push_back(info);
  }
  DCHECK(records.size() == toUpdate.size());

  // A) Identify all gc pointers which are statically live at the given call
  // site.
  findLiveReferences(F, DT, P, toUpdate, records, gc_collected_pointers);

  // The base phi insertion logic (for any safepoint) may have inserted new
  // instructions which are now live at some safepoint.  The simplest such
  // example is:
  // loop:
  //   phi a  <-- will be a new base_phi here
  //   safepoint 1 <-- that needs to be live here
  //   gep a + 1
  //   safepoint 2
  //   br loop
  // We insert some dummy calls after each safepoint to definitely hold live
  // the base pointers which were identified for that safepoint.  We'll then
  // ask liveness for _every_ base inserted to see what is now live.  Then we
  // remove the dummy calls.
  holders.reserve(holders.size() + records.size());
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    CallSite &CS = toUpdate[i];

    SmallVector<Value *, 128> Bases;
    for (auto pointer: info.liveset) {
      Bases.push_back(pointer);
    }
    insertUseHolderAfter(CS, Bases, holders);
  }

  // By selecting base pointers, we've effectively inserted new uses. Thus, we
  // need to rerun liveness.  We may *also* have inserted new defs, but that's
  // not the key issue.
  recomputeLiveInValues(F, DT, P, toUpdate, records, gc_collected_pointers);

  for (size_t i = 0; i < holders.size(); i++) {
    holders[i]->eraseFromParent();
    holders[i] = nullptr;
  }
  holders.clear();

  // Do a limited scalarization of any live at safepoint vector values which
  // contain pointers.  This enables this pass to run after vectorization at
  // the cost of some possible performance loss.  TODO: it would be nice to
  // natively support vectors all the way through the backend so we don't need
  // to scalarize here.
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    Instruction *statepoint = toUpdate[i].getInstruction();
    splitVectorValues(cast<Instruction>(statepoint), info.liveset, DT);
  }

  // Now run through and replace the existing statepoints with new ones with
  // the live variables listed.  We do not yet update uses of the values being
  // relocated. We have references to live variables that need to
  // survive to the last iteration of this loop.  (By construction, the
  // previous statepoint can not be a live variable, thus we can and remove
  // the old statepoint calls as we go.)
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    CallSite &CS = toUpdate[i];
    makeStatepointExplicit(DT, CS, P, info);
  }
  toUpdate.clear(); // prevent accident use of invalid CallSites

  // Do all the fixups of the original live variables to their relocated selves
  SmallVector<Value *, 128> live;
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    // We can't simply save the live set from the original insertion.  One of
    // the live values might be the result of a call which needs a safepoint.
    // That Value* no longer exists and we need to use the new gc_result.
    // Thankfully, the liveset is embedded in the statepoint (and updated), so
    // we just grab that.
    Statepoint statepoint(info.StatepointToken);
    live.insert(live.end(), statepoint.gc_args_begin(),
                statepoint.gc_args_end());
#ifndef NDEBUG
    // Do some basic sanity checks on our liveness results before performing
    // relocation.  Relocation can and will turn mistakes in liveness results
    // into non-sensical code which is must harder to debug.
    // TODO: It would be nice to test consistency as well
    DCHECK(DT.isReachableFromEntry(info.StatepointToken->getParent()) &&
           "statepoint must be reachable or liveness is meaningless");
    for (Value *V : statepoint.gc_args()) {
      if (!isa<Instruction>(V))
        // Non-instruction values trivial dominate all possible uses
        continue;
      auto LiveInst = cast<Instruction>(V); 
      USE(LiveInst);
      DCHECK(DT.isReachableFromEntry(LiveInst->getParent()) &&
             "unreachable values should never be live");
      DCHECK(DT.dominates(LiveInst, info.StatepointToken) &&
             "basic SSA liveness expectation violated by liveness analysis");
    }
#endif
  }
  unique_unsorted(live);

  relocationViaAlloca(F, DT, live, records);
  return !records.empty();
}

// Handles both return values and arguments for Functions and CallSites.
template <typename AttrHolder>
static void RemoveDerefAttrAtIndex(LLVMContext &Ctx, AttrHolder &AH,
                                   unsigned Index) {
  AttrBuilder R;
  if (AH.getDereferenceableBytes(Index))
    R.addAttribute(Attribute::get(Ctx, Attribute::Dereferenceable,
                                  AH.getDereferenceableBytes(Index)));
  if (AH.getDereferenceableOrNullBytes(Index))
    R.addAttribute(Attribute::get(Ctx, Attribute::DereferenceableOrNull,
                                  AH.getDereferenceableOrNullBytes(Index)));

  if (!R.empty())
    AH.setAttributes(AH.getAttributes().removeAttributes(
        Ctx, Index, AttributeSet::get(Ctx, Index, R)));
}

void
RewriteStatepointsForGC::stripDereferenceabilityInfoFromPrototype(Function &F) {
  LLVMContext &Ctx = F.getContext();

  for (Argument &A : F.args())
    if (isa<PointerType>(A.getType()))
      RemoveDerefAttrAtIndex(Ctx, F, A.getArgNo() + 1);

  if (isa<PointerType>(F.getReturnType()))
    RemoveDerefAttrAtIndex(Ctx, F, AttributeSet::ReturnIndex);
}

void RewriteStatepointsForGC::stripDereferenceabilityInfoFromBody(Function &F) {
  if (F.empty())
    return;

  LLVMContext &Ctx = F.getContext();
  MDBuilder Builder(Ctx);

  for (Instruction &I : instructions(F)) {
    if (const MDNode *MD = I.getMetadata(LLVMContext::MD_tbaa)) {
      DCHECK(MD->getNumOperands() < 5 && "unrecognized metadata shape!");
      bool IsImmutableTBAA =
          MD->getNumOperands() == 4 &&
          mdconst::extract<ConstantInt>(MD->getOperand(3))->getValue() == 1;

      if (!IsImmutableTBAA)
        continue; // no work to do, MD_tbaa is already marked mutable

      MDNode *Base = cast<MDNode>(MD->getOperand(0));
      MDNode *Access = cast<MDNode>(MD->getOperand(1));
      uint64_t Offset =
          mdconst::extract<ConstantInt>(MD->getOperand(2))->getZExtValue();

      MDNode *MutableTBAA =
          Builder.createTBAAStructTagNode(Base, Access, Offset);
      I.setMetadata(LLVMContext::MD_tbaa, MutableTBAA);
    }

    if (CallSite CS = CallSite(&I)) {
      for (int i = 0, e = CS.arg_size(); i != e; i++)
        if (isa<PointerType>(CS.getArgument(i)->getType()))
          RemoveDerefAttrAtIndex(Ctx, CS, i + 1);
      if (isa<PointerType>(CS.getType()))
        RemoveDerefAttrAtIndex(Ctx, CS, AttributeSet::ReturnIndex);
    }
  }
}

/// Returns true if this function should be rewritten by this pass.  The main
/// point of this function is as an extension point for custom logic.
static bool shouldRewriteStatepointsIn(Function &F) {
  // TODO: This should check the GCStrategy
  if (F.hasGC()) {
    const char *FunctionGCName = F.getGC();
    const StringRef StatepointExampleName("statepoint-example");
    const StringRef CoreCLRName("coreclr");
    const StringRef V8GCName("v8-gc");
    return (StatepointExampleName == FunctionGCName) ||
           (CoreCLRName == FunctionGCName) ||
           (V8GCName == FunctionGCName);
  } else
    return false;
}

void RewriteStatepointsForGC::stripDereferenceabilityInfo(Module &M) {
#ifndef NDEBUG
  DCHECK(std::any_of(M.begin(), M.end(), shouldRewriteStatepointsIn) &&
         "precondition!");
#endif

  for (Function &F : M)
    stripDereferenceabilityInfoFromPrototype(F);

  for (Function &F : M)
    stripDereferenceabilityInfoFromBody(F);
}

bool RewriteStatepointsForGC::runOnFunction(Function &F) {
  // Nothing to do for declarations.
  if (F.isDeclaration() || F.empty())
    return false;

  // Policy choice says not to rewrite - the most common reason is that we're
  // compiling code without a GCStrategy.
  if (!shouldRewriteStatepointsIn(F))
    return false;

  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();

  // Gather all the statepoints which need rewritten.  Be careful to only
  // consider those in reachable code since we need to ask dominance queries
  // when rewriting.  We'll delete the unreachable ones in a moment.
  SmallVector<CallSite, 64> ParsePointNeeded;
  bool HasUnreachableStatepoint = false;
  for (Instruction &I : instructions(F)) {
    // TODO: only the ones with the flag set!
    if (isStatepoint(I)) {
      if (DT.isReachableFromEntry(I.getParent()))
        ParsePointNeeded.push_back(CallSite(&I));
      else
        HasUnreachableStatepoint = true;
    }
  }

  bool MadeChange = false;

  // Delete any unreachable statepoints so that we don't have unrewritten
  // statepoints surviving this pass.  This makes testing easier and the
  // resulting IR less confusing to human readers.  Rather than be fancy, we
  // just reuse a utility function which removes the unreachable blocks.
  if (HasUnreachableStatepoint)
    MadeChange |= removeUnreachableBlocks(F);

  // Return early if no work to do.
  if (ParsePointNeeded.empty())
    return MadeChange;

  // As a prepass, go ahead and aggressively destroy single entry phi nodes.
  // These are created by LCSSA.  They have the effect of increasing the size
  // of liveness sets for no good reason.  It may be harder to do this post
  // insertion since relocations and base phis can confuse things.
  for (BasicBlock &BB : F)
    if (BB.getUniquePredecessor()) {
      MadeChange = true;
      FoldSingleEntryPHINodes(&BB);
    }

  // Before we start introducing relocations, we want to tweak the IR a bit to
  // avoid unfortunate code generation effects.  The main example is that we
  // want to try to make sure the comparison feeding a branch is after any
  // safepoints.  Otherwise, we end up with a comparison of pre-relocation
  // values feeding a branch after relocation.  This is semantically correct,
  // but results in extra register pressure since both the pre-relocation and
  // post-relocation copies must be available in registers.  For code without
  // relocations this is handled elsewhere, but teaching the scheduler to
  // reverse the transform we're about to do would be slightly complex.
  // Note: This may extend the live range of the inputs to the icmp and thus
  // increase the liveset of any statepoint we move over.  This is profitable
  // as long as all statepoints are in rare blocks.  If we had in-register
  // lowering for live values this would be a much safer transform.
  auto getConditionInst = [](TerminatorInst *TI) -> Instruction* {
    if (auto *BI = dyn_cast<BranchInst>(TI))
      if (BI->isConditional())
        return dyn_cast<Instruction>(BI->getCondition());
    // TODO: Extend this to handle switches
    return nullptr;
  };
  for (BasicBlock &BB : F) {
    TerminatorInst *TI = BB.getTerminator();
    if (auto *Cond = getConditionInst(TI))
      // TODO: Handle more than just ICmps here.  We should be able to move
      // most instructions without side effects or memory access.
      if (isa<ICmpInst>(Cond) && Cond->hasOneUse()) {
        MadeChange = true;
        Cond->moveBefore(TI);
      }
  }

  MadeChange |= insertParsePoints(F, DT, this, ParsePointNeeded,
                                  gc_collected_pointers_);
  return MadeChange;
}

// liveness computation via standard dataflow
// -------------------------------------------------------------------

// TODO: Consider using bitvectors for liveness, the set of potentially
// interesting values should be small and easy to pre-compute.

/// Compute the live-in set for the location rbegin starting from
/// the live-out set of the basic block
static void computeLiveInValues(BasicBlock::reverse_iterator rbegin,
                                BasicBlock::reverse_iterator rend,
                                DenseSet<Value *> &LiveTmp,
                                ValueSet& gc_collected_pointers) {

  for (BasicBlock::reverse_iterator ritr = rbegin; ritr != rend; ritr++) {
    Instruction *I = &*ritr;

    // KILL/Def - Remove this definition from LiveIn
    LiveTmp.erase(I);

    // Don't consider *uses* in PHI nodes, we handle their contribution to
    // predecessor blocks when we seed the LiveOut sets
    if (isa<PHINode>(I))
      continue;

    // USE - Add to the LiveIn set for this instruction
    for (Value *V : I->operands()) {
      if (gc_collected_pointers.count(V)) {
        LiveTmp.insert(V);
      }
    }
  }
}

static void computeLiveOutSeed(BasicBlock *BB,
                               DenseSet<Value *> &LiveTmp,
                               ValueSet& gc_collected_pointers) {

  for (BasicBlock *Succ : successors(BB)) {
    const BasicBlock::iterator E(Succ->getFirstNonPHI());
    for (BasicBlock::iterator I = Succ->begin(); I != E; I++) {
      PHINode *Phi = cast<PHINode>(&*I);
      Value *V = Phi->getIncomingValueForBlock(BB);
      if (gc_collected_pointers.count(V))
        LiveTmp.insert(V);
    }
  }
}

static DenseSet<Value *> computeKillSet(BasicBlock *BB,
                                        ValueSet& gc_collected_pointers) {
  DenseSet<Value *> KillSet;
  for (Instruction &I : *BB)
    if (gc_collected_pointers.count(&I))
      KillSet.insert(&I);
  return KillSet;
}

#ifndef NDEBUG
/// Check that the items in 'Live' dominate 'TI'.  This is used as a basic
/// sanity check for the liveness computation.
static void checkBasicSSA(DominatorTree &DT, DenseSet<Value *> &Live,
                          TerminatorInst *TI, bool TermOkay = false) {
  for (Value *V : Live) {
    if (auto *I = dyn_cast<Instruction>(V)) {
      // The terminator can be a member of the LiveOut set.  LLVM's definition
      // of instruction dominance states that V does not dominate itself.  As
      // such, we need to special case this to allow it.
      if (TermOkay && TI == I)
        continue;
      DCHECK(DT.dominates(I, TI) &&
             "basic SSA liveness expectation violated by liveness analysis");
    }
  }
}

/// Check that all the liveness sets used during the computation of liveness
/// obey basic SSA properties.  This is useful for finding cases where we miss
/// a def.
static void checkBasicSSA(DominatorTree &DT, GCPtrLivenessData &Data,
                          BasicBlock &BB) {
  checkBasicSSA(DT, Data.LiveSet[&BB], BB.getTerminator());
  checkBasicSSA(DT, Data.LiveOut[&BB], BB.getTerminator(), true);
  checkBasicSSA(DT, Data.LiveIn[&BB], BB.getTerminator());
}
#endif

static void computeLiveInValues(DominatorTree &DT, Function &F,
                                GCPtrLivenessData &Data,
                                ValueSet& gc_collected_pointers) {

  SmallSetVector<BasicBlock *, 200> Worklist;
  auto AddPredsToWorklist = [&](BasicBlock *BB) {
    // We use a SetVector so that we don't have duplicates in the worklist.
    Worklist.insert(pred_begin(BB), pred_end(BB));
  };
  auto NextItem = [&]() {
    BasicBlock *BB = Worklist.back();
    Worklist.pop_back();
    return BB;
  };

  // Seed the liveness for each individual block
  for (BasicBlock &BB : F) {
    Data.KillSet[&BB] = computeKillSet(&BB, gc_collected_pointers);
    Data.LiveSet[&BB].clear();
    computeLiveInValues(BB.rbegin(), BB.rend(), Data.LiveSet[&BB],
                        gc_collected_pointers);

#ifndef NDEBUG
    for (Value *Kill : Data.KillSet[&BB]) {
      USE(Kill);
      DCHECK(!Data.LiveSet[&BB].count(Kill) && "live set contains kill");
    }
#endif

    Data.LiveOut[&BB] = DenseSet<Value *>();
    computeLiveOutSeed(&BB, Data.LiveOut[&BB], gc_collected_pointers);
    Data.LiveIn[&BB] = Data.LiveSet[&BB];
    set_union(Data.LiveIn[&BB], Data.LiveOut[&BB]);
    set_subtract(Data.LiveIn[&BB], Data.KillSet[&BB]);
    if (!Data.LiveIn[&BB].empty())
      AddPredsToWorklist(&BB);
  }

  // Propagate that liveness until stable
  while (!Worklist.empty()) {
    BasicBlock *BB = NextItem();

    // Compute our new liveout set, then exit early if it hasn't changed
    // despite the contribution of our successor.
    DenseSet<Value *> LiveOut = Data.LiveOut[BB];
    const auto OldLiveOutSize = LiveOut.size();
    for (BasicBlock *Succ : successors(BB)) {
      DCHECK(Data.LiveIn.count(Succ));
      set_union(LiveOut, Data.LiveIn[Succ]);
    }
    // DCHECK OutLiveOut is a subset of LiveOut
    if (OldLiveOutSize == LiveOut.size()) {
      // If the sets are the same size, then we didn't actually add anything
      // when unioning our successors LiveIn  Thus, the LiveIn of this block
      // hasn't changed.
      continue;
    }
    Data.LiveOut[BB] = LiveOut;

    // Apply the effects of this basic block
    DenseSet<Value *> LiveTmp = LiveOut;
    set_union(LiveTmp, Data.LiveSet[BB]);
    set_subtract(LiveTmp, Data.KillSet[BB]);

    DCHECK(Data.LiveIn.count(BB));
    const DenseSet<Value *> &OldLiveIn = Data.LiveIn[BB];
    // DCHECK: OldLiveIn is a subset of LiveTmp
    if (OldLiveIn.size() != LiveTmp.size()) {
      Data.LiveIn[BB] = LiveTmp;
      AddPredsToWorklist(BB);
    }
  } // while( !worklist.empty() )

#ifndef NDEBUG
  // Sanity check our output against SSA properties.  This helps catch any
  // missing kills during the above iteration.
  for (BasicBlock &BB : F) {
    checkBasicSSA(DT, Data, BB);
  }
#endif
}

static void findLiveSetAtInst(Instruction *Inst, GCPtrLivenessData &Data,
                              StatepointLiveSetTy &Out,
                              ValueSet& gc_collected_pointers) {

  BasicBlock *BB = Inst->getParent();

  // Note: The copy is intentional and required
  DCHECK(Data.LiveOut.count(BB));
  DenseSet<Value *> LiveOut = Data.LiveOut[BB];

  // We want to handle the statepoint itself oddly.  It's
  // call result is not live (normal), nor are it's arguments
  // (unless they're used again later).  This adjustment is
  // specifically what we need to relocate
  BasicBlock::reverse_iterator rend(Inst);
  computeLiveInValues(BB->rbegin(), rend, LiveOut, gc_collected_pointers);
  LiveOut.erase(Inst);
  Out.insert(LiveOut.begin(), LiveOut.end());
}

static void recomputeLiveInValues(GCPtrLivenessData &RevisedLivenessData,
                                  const CallSite &CS,
                                  PartiallyConstructedSafepointRecord &Info,
                                  ValueSet& gc_collected_pointers) {
  Instruction *Inst = CS.getInstruction();
  StatepointLiveSetTy Updated;
  findLiveSetAtInst(Inst, RevisedLivenessData, Updated,
                    gc_collected_pointers);
  Info.liveset = Updated;
}
