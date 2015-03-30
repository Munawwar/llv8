// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_MCJIT_MEMORY_MANAGER_H_
#define V8_MCJIT_MEMORY_MANAGER_H_

#include "llvm-headers.h"

namespace v8 {
namespace internal {

class MCJITMemoryManager : public llvm::RTDyldMemoryManager {
 public:
  MCJITMemoryManager();
  virtual ~MCJITMemoryManager();

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID,
                               llvm::StringRef SectionName) override;

  uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID, llvm::StringRef SectionName,
                               bool isReadOnly) override;

  bool finalizeMemory(std::string *ErrMsg) override;

 private:

};

} }  // namespace v8::internal
#endif  // V8_MCJIT_MEMORY_MANAGER_H_
