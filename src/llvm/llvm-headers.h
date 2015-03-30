// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_LLVM_HEADERS_H_
#define V8_LLVM_HEADERS_H_

// FIXME(llvm): remove unneeded headers
// FIXME(llvm): sort headers (style)
#include <iostream>

#include <llvm/IR/IRBuilder.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/ExecutionEngine/MCJIT.h>
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"

namespace v8 {
namespace internal {

using llvm::LLVMContext;
//using llvm::Module;

}  // namespace internal
}  // namespace v8

#endif  // V8_LLVM_HEADERS_H_
