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
      zone_(graph->zone()) {}

Isolate* LowChunkBuilderBase::isolate() const {
  return graph_->isolate();
}

LowChunk::LowChunk(CompilationInfo* info, HGraph* graph)
    : info_(info),
      graph_(graph) {}

Isolate* LowChunk::isolate() const {
  return graph_->isolate();
}

} }  // namespace v8::internal
