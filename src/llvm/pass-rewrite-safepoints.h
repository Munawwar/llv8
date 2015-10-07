// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_PASS_REWRITE_SAFEPOINTS_H_
#define V8_PASS_REWRITE_SAFEPOINTS_H_

#include "llvm-headers.h"


namespace v8 {
namespace internal {

using ValueSet = std::set<llvm::Value*>;

llvm::FunctionPass* createAppendLivePointersToSafepointsPass(ValueSet&);

// These structs are borrowed from
// llvm/lib/Transforms/Scalar/RewriteStatepointsForGC.cpp.

struct GCPtrLivenessData {
  /// Values defined in this block.
  llvm::DenseMap<llvm::BasicBlock*, llvm::DenseSet<llvm::Value*>> KillSet;
  /// Values used in this block (and thus live); does not included values
  /// killed within this block.
  llvm::DenseMap<llvm::BasicBlock*, llvm::DenseSet<llvm::Value*>> LiveSet;

  /// Values live into this basic block (i.e. used by any
  /// instruction in this basic block or ones reachable from here)
  llvm::DenseMap<llvm::BasicBlock*, llvm::DenseSet<llvm::Value*>> LiveIn;

  /// Values live out of this basic block (i.e. live into
  /// any successor block)
  llvm::DenseMap<llvm::BasicBlock*, llvm::DenseSet<llvm::Value*>> LiveOut;
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
typedef llvm::DenseMap<llvm::Value*, llvm::Value*> DefiningValueMapTy;
typedef llvm::DenseSet<llvm::Value*> StatepointLiveSetTy;
typedef llvm::DenseMap<llvm::Instruction*, llvm::Value*>
    RematerializedValueMapTy;

struct PartiallyConstructedSafepointRecord {
  /// The set of values known to be live across this safepoint
  StatepointLiveSetTy liveset;

  /// Mapping from live pointers to a base-defining-value
  llvm::DenseMap<llvm::Value*, llvm::Value*> PointerToBase;

  /// The *new* gc.statepoint instruction itself.  This produces the token
  /// that normal path gc.relocates and the gc.result are tied to.
  llvm::Instruction* StatepointToken;

  /// Instruction to which exceptional gc relocates are attached
  /// Makes it easier to iterate through them during relocationViaAlloca.
  llvm::Instruction* UnwindToken;

  /// Record live values we are rematerialized instead of relocating.
  /// They are not included into 'liveset' field.
  /// Maps rematerialized copy to it's original value.
  RematerializedValueMapTy RematerializedValues;
};

} }  // namespace v8::internal


#endif /* V8_PASS_REWRITE_SAFEPOINTS_H_ */
