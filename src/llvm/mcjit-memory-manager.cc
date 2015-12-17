// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "mcjit-memory-manager.h"
#include "src/allocation.h"
#include "src/base/logging.h"
#include "src/base/platform/platform.h"
// FIXME(llvm): we only need IntHelper from there. Move it to a separate file.
#include "llvm-chunk.h"

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
  // Note: we don't care for the executable attribute here.
  // Because before being executed the code gets copied to another place.
  byte* buffer = NewArray<byte>(size);
  CHECK(alignment == 0 ||
        (reinterpret_cast<uintptr_t>(buffer) &
        static_cast<uintptr_t>(alignment - 1)) == 0);
  CodeDesc desc;
  desc.buffer = buffer;
  desc.buffer_size = IntHelper::AsInt(size);
  desc.instr_size = IntHelper::AsInt(size);
  desc.reloc_size = 0;
  desc.origin = nullptr;
  allocated_code_.Add(desc);
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

  // FIXME(llvm): this is wrong understanding of the alignment parameter.
  // see allocateCodeSection.
  byte* buffer = NewArray<byte>(RoundUp(size, alignment));
  allocated_data_.Add(buffer);
  if (section_name.equals(".llvm_stackmaps"))
    stackmaps_.Add(buffer);
#ifdef DEBUG
  std::cerr << reinterpret_cast<void*>(buffer) << std::endl;
#endif

  return buffer;
}

bool MCJITMemoryManager::finalizeMemory(std::string *ErrMsg) {
  return false;
}

} }  // namespace v8::internal
