// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_LLVM_CHUNK_H_
#define V8_LLVM_CHUNK_H_

#include "../handles.h"
#include "../lithium.h"
#include "llvm-headers.h"

namespace v8 {
namespace internal {


class LLVMChunk FINAL : public LowChunk {
  public:
    virtual ~LLVMChunk() {}
    LLVMChunk(CompilationInfo* info, HGraph* graph)
      : LowChunk(info, graph) {}
    static LLVMChunk* NewChunk(HGraph *graph);
    Handle<Code> Codegen() override;
};


}  // namespace internal
}  // namespace v8
#endif  // V8_LLVM_CHUNK_H_
