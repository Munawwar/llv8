// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "low-chunk.h"

namespace v8 {
namespace internal {

LowChunk::~LowChunk() {}

LowChunkBuilderBase::LowChunkBuilderBase(CompilationInfo* info, HGraph* graph)
    : chunk_(nullptr),
      info_(info),
      graph_(graph),
      status_(UNUSED),
      argument_count_(0),
      current_block_(nullptr),
      next_block_(nullptr),
      zone_(graph->zone()) {}

Isolate* LowChunkBuilderBase::isolate() const {
  return graph_->isolate();
}

LowChunk::LowChunk(CompilationInfo* info, HGraph* graph)
    : stability_dependencies_(MapLess(), MapAllocator(info->zone())),
      info_(info),
      graph_(graph) {}

Isolate* LowChunk::isolate() const {
  return graph_->isolate();
}

void LowChunkBuilderBase::Abort(BailoutReason reason) {
  info()->AbortOptimization(reason);
  status_ = ABORTED;
}

void LowChunkBuilderBase::Retry(BailoutReason reason) {
  info()->RetryOptimization(reason);
  status_ = ABORTED;
}

} }  // namespace v8::internal
