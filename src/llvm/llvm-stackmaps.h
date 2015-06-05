// Copyright (C) 2013, 2014 Apple Inc. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY APPLE INC. ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL APPLE INC. OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// This code has been ported from JavaScriptCore (see FTLStackMaps.h).
// Copyright 2015 ISP RAS. All rights reserved.

#ifndef V8_LLVM_STACKMAPS_H_
#define V8_LLVM_STACKMAPS_H_

#include "llvm-headers.h"
//#include "src/list-inl.h"
#include "src/x64/assembler-x64-inl.h" // For now

#include <map>
#include <vector>

#define OVERLOAD_STREAM_INSERTION(type) \
  friend std::ostream& operator<<(std::ostream& os, type* rhs) { \
    rhs->dump(os); \
    return os; \
  } \
  friend std::ostream& operator<<(std::ostream& os, type& rhs) { \
    rhs.dump(os); \
    return os; \
  }


namespace v8 {
namespace internal {

class DataView {
 public:
  DataView(byte* array)
     : array_(array),
       offset_(0) {}

  template<typename T>
  T read(bool littleEndian) {
    // TODO(llvm): it's gonna be bad for big endian archs.
    USE(littleEndian);
    T result = *reinterpret_cast<T*>(array_ + offset_);
    offset_ += sizeof(T);
    return result;
  }

  int offset() { return offset_; }
 private:
  byte* array_;
  int offset_;
};

class DWARFRegister {
 public:
  DWARFRegister()
      : dwarf_reg_num_(-1) {}

  explicit DWARFRegister(int16_t dwarf_reg_num)
      : dwarf_reg_num_(dwarf_reg_num) {}

  int16_t dwarf_reg_num() const { return dwarf_reg_num_; }

  Register reg() const;

  // TODO(llvm): method names should start with a capital (style)
  void dump(std::ostream&) const;

  OVERLOAD_STREAM_INSERTION(DWARFRegister)

 private:
  int16_t dwarf_reg_num_;
};

struct StackMaps {
  struct ParseContext {
    unsigned version;
    DataView* view;
  };

  struct Constant {
    int64_t integer;

    void parse(ParseContext&);
    void dump(std::ostream&);

    OVERLOAD_STREAM_INSERTION(Constant)
  };

  struct StackSize {
    uint64_t functionOffset;
    uint64_t size;

    void parse(ParseContext&);
    void dump(std::ostream&);

    OVERLOAD_STREAM_INSERTION(StackSize)
  };

  struct Location {
    enum Kind : int8_t {
      kUnprocessed,
      kRegister = 0x1,
      kDirect,
      kIndirect,
      kConstant,
      kConstantIndex
    };

    DWARFRegister dwarf_reg;
    uint8_t size;
    Kind kind;
    int32_t offset;

    void parse(ParseContext&);
    void dump(std::ostream&);

    OVERLOAD_STREAM_INSERTION(Location)
//    GPRReg directGPR() const;
//    void restoreInto(MacroAssembler&, StackMaps&, char* savedRegisters, GPRReg result) const;
  };

  // TODO(llvm): https://bugs.webkit.org/show_bug.cgi?id=130802
  struct LiveOut {
    DWARFRegister dwarfReg;
    uint8_t size;

    void parse(ParseContext&);
    void dump(std::ostream&);

    OVERLOAD_STREAM_INSERTION(LiveOut)
  };

  struct Record {
    uint32_t patchpointID;
    uint32_t instructionOffset;
    uint16_t flags;

    std::vector<Location> locations;
    std::vector<LiveOut> live_outs;

    bool parse(ParseContext&);
    void dump(std::ostream&);

    OVERLOAD_STREAM_INSERTION(Record)
//
//    RegisterSet liveOutsSet() const;
//    RegisterSet locationSet() const;
//    RegisterSet usedRegisterSet() const;
  };

  unsigned version;
  std::vector<StackSize> stack_sizes;
  std::vector<Constant> constants;
  std::vector<Record> records;

  // Returns true on parse success, false on failure.
  // Failure means that LLVM is signaling compile failure to us.
  bool parse(DataView*);
  void dump(std::ostream&);
  void dumpMultiline(std::ostream&, const char* prefix);

  using RecordMap = std::map<uint32_t, Record>; // PatchPoint ID -> Record

  RecordMap computeRecordMap() const;

  unsigned stackSize() const;
};


} } // namespace v8::internal

#endif // V8_LLVM_STACKMAPS_H_
