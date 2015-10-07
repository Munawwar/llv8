// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_LLVM_PASSES_H_
#define V8_LLVM_PASSES_H_

#include "llvm-headers.h"


namespace v8 {
namespace internal {

llvm::FunctionPass* createNormalizePhisPass();

} }  // namespace v8::internal
#endif  // V8_LLVM_PASSES_H_
