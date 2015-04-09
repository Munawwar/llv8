// Copyright 2015 ISP RAS. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_MCJIT_MEMORY_MANAGER_H_
#define V8_MCJIT_MEMORY_MANAGER_H_

#include "llvm-headers.h"

#include "src/globals.h"
#include "src/list-inl.h"

namespace v8 {
namespace internal {

class MCJITMemoryManager : public llvm::RTDyldMemoryManager {
 public:
  static std::unique_ptr<MCJITMemoryManager> Create();

  MCJITMemoryManager();
  virtual ~MCJITMemoryManager();

  // Allocate a memory block of (at least) the given size suitable for
  // executable code. The section_id is a unique identifier assigned by the
  // MCJIT engine, and optionally recorded by the memory manager to access a
  // loaded section.
  byte* allocateCodeSection(uintptr_t size, unsigned alignment,
                            unsigned section_id,
                            llvm::StringRef section_name) override;

  // Allocate a memory block of (at least) the given size suitable for data.
  // The SectionID is a unique identifier assigned by the JIT engine, and
  // optionally recorded by the memory manager to access a loaded section.
  byte* allocateDataSection(uintptr_t size, unsigned alignment,
                            unsigned section_id, llvm::StringRef section_name,
                            bool is_readonly) override;

  // This method is called after an object has been loaded into memory but
  // before relocations are applied to the loaded sections.  The object load
  // may have been initiated by MCJIT to resolve an external symbol for another
  // object that is being finalized.  In that case, the object about which
  // the memory manager is being notified will be finalized immediately after
  // the memory manager returns from this call.
  //
  // Memory managers which are preparing code for execution in an external
  // address space can use this call to remap the section addresses for the
  // newly loaded object.
  void notifyObjectLoaded(llvm::ExecutionEngine* engine,
                          const llvm::object::ObjectFile &) override;

  // This method is called when object loading is complete and section page
  // permissions can be applied.  It is up to the memory manager implementation
  // to decide whether or not to act on this method.  The memory manager will
  // typically allocate all sections as read-write and then apply specific
  // permissions when this method is called.  Code sections cannot be executed
  // until this function has been called.  In addition, any cache coherency
  // operations needed to reliably use the memory are also performed.
  //
  // Returns true if an error occurred, false otherwise.
  bool finalizeMemory(std::string *ErrMsg) override;

  CodeDesc LastAllocatedCode() { return allocated_code_.last(); }

 private:
  // TODO(llvm): is it OK to allocate those in the zone?
  List<CodeDesc> allocated_code_;
  List<byte*> allocated_data_;
//  Zone* zone_;
};

} }  // namespace v8::internal
#endif  // V8_MCJIT_MEMORY_MANAGER_H_
