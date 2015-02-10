// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "low-chunk.h"

namespace v8 {
namespace internal {

LowChunk::LowChunk(CompilationInfo* info, HGraph* graph)
    : info_(info),
      graph_(graph) {}


} }  // namespace v8::internal
