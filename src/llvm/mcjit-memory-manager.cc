// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "mcjit-memory-manager.h"
#include "src/allocation.h"
#include "src/base/logging.h"
#include "src/base/platform/platform.h"

#include <cstdbool>
#include <cstdint>
#include <string>

namespace v8 {
namespace internal {

std::unique_ptr<MCJITMemoryManager> MCJITMemoryManager::Create() {
  return llvm::make_unique<MCJITMemoryManager>();
}

MCJITMemoryManager::MCJITMemoryManager()
  : allocated_code_(1),
    allocated_data_(1),
    stackmaps_(1) {}

MCJITMemoryManager::~MCJITMemoryManager() {
  for (auto it = allocated_code_.begin(); it != allocated_code_.end(); ++it) {
    DeleteArray(it->buffer);
  }
  for (auto it = allocated_data_.begin(); it != allocated_data_.end(); ++it) {
    DeleteArray(*it);
  }
}

void MCJITMemoryManager::notifyObjectLoaded(llvm::ExecutionEngine* engine,
                                            const llvm::object::ObjectFile &) {
//  UNIMPLEMENTED();
}

byte* MCJITMemoryManager::allocateCodeSection(uintptr_t size,
                                              unsigned alignment,
                                              unsigned section_id,
                                              llvm::StringRef section_name) {
#ifdef DEBUG
  std::cerr << __FUNCTION__ << " section_name == "
      << section_name.str() << " section id == "
      << section_id << std::endl;
#endif
  CHECK(alignment <= base::OS::AllocateAlignment());
  int size_as_int = static_cast<int>(size);
  CHECK_EQ(size_as_int, size);
//  size_t actual_size;
//  uint8_t* buffer =
//      static_cast<uint8_t*>(base::OS::Allocate(size, &actual_size, true));
  byte* buffer = NewArray<byte>(RoundUp(size, alignment));
  CodeDesc desc;
  desc.buffer = buffer;
  desc.buffer_size = RoundUp(size_as_int, alignment);
  desc.instr_size = size_as_int;
  desc.reloc_size = 0;
  desc.origin = nullptr;
  allocated_code_.Add(desc);
  DCHECK_EQ(size_as_int, size);
  return buffer;
}

byte* MCJITMemoryManager::allocateDataSection(uintptr_t size,
                                              unsigned alignment,
                                              unsigned section_id,
                                              llvm::StringRef section_name,
                                              bool is_readonly) {
#ifdef DEBUG
  std::cerr << __FUNCTION__ << " section_name == "
      << section_name.str() << " section id == "
      << section_id << " size == "
      << size << std::endl;
#endif
  CHECK(alignment <= base::OS::AllocateAlignment());
  // TODO(llvm): handle is_readonly
  if (section_name.equals(".got")) {
      UNIMPLEMENTED(); // TODO(llvm): call allocateCodeSection
                       // as far as you understand what's happening
  }
  // FIXME(llvm): who frees that memory?
  // The destructor here does. Not sure if it is supposed to.
  byte* buffer = NewArray<byte>(RoundUp(size, alignment));
  allocated_data_.Add(buffer);
  if (section_name.equals(".llvm_stackmaps"))
    stackmaps_.Add(buffer);
  return buffer;
}

bool MCJITMemoryManager::finalizeMemory(std::string *ErrMsg) {
  return false;
}

} }  // namespace v8::internal
