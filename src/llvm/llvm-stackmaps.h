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

#include "src/list-inl.h"
#include "src/x64/assembler-x64.h" // For now

#include <map>

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

class DumpListVisitor {
 public:
  DumpListVisitor(std::ostream& os)
     : os_(os) {}

  template <typename T>
  void Apply(T* element) {
    os_ << *element << ", ";
  }
 private:
  std::ostream& os_;
};

class DWARFRegister {
 public:
  DWARFRegister()
      : dwarf_reg_num_(-1) {}

  explicit DWARFRegister(int16_t dwarf_reg_num)
      : dwarf_reg_num_(dwarf_reg_num) {}

  int16_t dwarf_reg_num() const { return dwarf_reg_num_; }

  Register reg() const;

  void dump(std::ostream&) const;

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
    void dump(std::ostream&) const;
  };

  struct StackSize {
    uint64_t functionOffset;
    uint64_t size;

    void parse(ParseContext&);
    void dump(std::ostream&) const;
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

    DWARFRegister dwarfReg;
    uint8_t size;
    Kind kind;
    int32_t offset;

    void parse(ParseContext&);
    void dump(std::ostream&) const;

//    GPRReg directGPR() const;
//    void restoreInto(MacroAssembler&, StackMaps&, char* savedRegisters, GPRReg result) const;
  };

  // TODO(llvm): https://bugs.webkit.org/show_bug.cgi?id=130802
  struct LiveOut {
    DWARFRegister dwarfReg;
    uint8_t size;

    void parse(ParseContext&);
    void dump(std::ostream&) const;
  };

  struct Record {
    uint32_t patchpointID;
    uint32_t instructionOffset;
    uint16_t flags;

    List<Location> locations;
    List<LiveOut> liveOuts;

    bool parse(ParseContext&);
    void dump(std::ostream&) const;
//
//    RegisterSet liveOutsSet() const;
//    RegisterSet locationSet() const;
//    RegisterSet usedRegisterSet() const;
  };

  unsigned version;
  List<StackSize> stackSizes;
  List<Constant> constants;
  List<Record> records;

  // Returns true on parse success, false on failure.
  // Failure means that LLVM is signaling compile failure to us.
  bool parse(DataView*);
  void dump(std::ostream&) const;
  void dumpMultiline(std::ostream&, const char* prefix) const;

//  typedef HashMap<uint32_t, Vector<Record>, WTF::IntHash<uint32_t>, WTF::UnsignedWithZeroKeyHashTraits<uint32_t>>
  using RecordMap = std::map<uint32_t, Record>; // PatchPoint ID -> Record

  RecordMap computeRecordMap() const;

  unsigned stackSize() const;
};


} } // namespace v8::internal

#endif V8_LLVM_STACKMAPS_H_
