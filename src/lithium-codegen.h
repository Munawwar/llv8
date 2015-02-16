// Copyright 2013 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_LITHIUM_CODEGEN_H_
#define V8_LITHIUM_CODEGEN_H_

#include "src/bailout-reason.h"
#include "src/compiler.h"
#include "src/low-chunk.h"
#include "src/deoptimizer.h"

namespace v8 {
namespace internal {

class LEnvironment;
class LInstruction;
class LPlatformChunk;

class LCodeGenBase : public LowCodeGenBase {
 public:
  LCodeGenBase(LChunk* chunk,
               MacroAssembler* assembler,
               CompilationInfo* info);
  ~LCodeGenBase() override {}

  // Simple accessors.
  MacroAssembler* masm() const { return masm_; }
  LPlatformChunk* chunk() const; // shadows base chunk()

  void FPRINTF_CHECKING Comment(const char* format, ...);
  void DeoptComment(const Deoptimizer::DeoptInfo& deopt_info);
  static Deoptimizer::DeoptInfo MakeDeoptInfo(
      LInstruction* instr, Deoptimizer::DeoptReason deopt_reason);

  bool GenerateBody();
  virtual void GenerateBodyInstructionPre(LInstruction* instr) {}
  virtual void GenerateBodyInstructionPost(LInstruction* instr) {}

  virtual void EnsureSpaceForLazyDeopt(int space_needed) = 0;
  virtual void RecordAndWritePosition(int position) = 0;

  int GetNextEmittedBlock() const;

  void RegisterWeakObjectsInOptimizedCode(Handle<Code> code);

  void WriteTranslationFrame(LEnvironment* environment,
                             Translation* translation);
  int DefineDeoptimizationLiteral(Handle<Object> literal);

  // Check that an environment assigned via AssignEnvironment is actually being
  // used. Redundant assignments keep things alive longer than necessary, and
  // consequently lead to worse code, so it's important to minimize this.
  void CheckEnvironmentUsage();

 protected:
  enum Status {
    UNUSED,
    GENERATING,
    DONE,
    ABORTED
  };

  MacroAssembler* const masm_;

  Status status_; // TODO(llvm) consider pulling up this field and the enum
  int current_block_;
  int current_instruction_;
  const ZoneList<LInstruction*>* instructions_;
  ZoneList<Handle<Object> > deoptimization_literals_;
  int last_lazy_deopt_pc_;

  bool is_unused() const { return status_ == UNUSED; }
  bool is_generating() const { return status_ == GENERATING; }
  bool is_done() const { return status_ == DONE; }
  bool is_aborted() const { return status_ == ABORTED; }

  void Abort(BailoutReason reason);
  void Retry(BailoutReason reason);

  // Methods for code dependencies.
  void AddDeprecationDependency(Handle<Map> map);
  void AddStabilityDependency(Handle<Map> map);
};


}  // namespace internal
}  // namespace v8

#endif  // V8_LITHIUM_CODEGEN_H_
