// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_PASS_REWRITE_SAFEPOINTS_H_
#define V8_PASS_REWRITE_SAFEPOINTS_H_

#include "llvm-headers.h"


namespace v8 {
namespace internal {

using ValueSet = std::set<llvm::Value*>;

llvm::ModulePass* createRewriteStatepointsForGCPass(ValueSet&);


} }  // namespace v8::internal


#endif /* V8_PASS_REWRITE_SAFEPOINTS_H_ */
