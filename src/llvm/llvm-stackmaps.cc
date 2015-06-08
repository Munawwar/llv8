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

// This code has been ported from JavaScriptCore (see FTLStackMaps.cpp).
// Copyright 2015 ISP RAS. All rights reserved.

#include "llvm-stackmaps.h"

#include <algorithm>

namespace v8 {
namespace internal {

template<typename T>
T readObject(StackMaps::ParseContext& context) {
  T result;
  result.parse(context);
  return result;
}

Register DWARFRegister::reg() const {
  if (dwarf_reg_num_ < 0 ||
      dwarf_reg_num_ >= Register::kNumRegisters) {
    UNIMPLEMENTED();
  }
  int const map[] = { 0, 2, 1, 3, 6, 7, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15 };
  return Register::from_code(map[dwarf_reg_num_]);
}

void DWARFRegister::dump(std::ostream& os) const {
    Register reg = this->reg();
    os << Register::AllocationIndexToString(Register::ToAllocationIndex(reg));
}

void StackMaps::Constant::parse(StackMaps::ParseContext& context) {
  integer = context.view->read<int64_t>(true);
}

void StackMaps::Constant::dump(std::ostream& os) const {
  os << static_cast<unsigned long long>(integer);
}

void StackMaps::StackSize::parse(StackMaps::ParseContext& context) {
  switch (context.version) {
    case 0:
      functionOffset = context.view->read<uint32_t>(true);
      size = context.view->read<uint32_t>(true);
      break;

    default:
      functionOffset = context.view->read<uint64_t>(true);
      size = context.view->read<uint64_t>(true);
      break;
  }
}

void StackMaps::StackSize::dump(std::ostream& os) const {
  os << "(off:" << functionOffset << ", size:" << size << ")";
}

void StackMaps::Location::parse(StackMaps::ParseContext& context) {
  kind = static_cast<Kind>(context.view->read<uint8_t>(true));
  size = context.view->read<uint8_t>(true);
  dwarf_reg = DWARFRegister(context.view->read<uint16_t>(true));
  this->offset = context.view->read<int32_t>(true);
}

void StackMaps::Location::dump(std::ostream& os) const {
  os << "(" << kind << ", "
      << dwarf_reg << ", off:"
      << offset << ", size:"
      << size << ")";
}

//GPRReg StackMaps::Location::directGPR() const {
//  return FTL::Location::forStackmaps(nullptr, *this).directGPR();
//}
//
//void StackMaps::Location::restoreInto(
//    MacroAssembler& jit, StackMaps& stackmaps, char* savedRegisters, GPRReg result) const {
//  FTL::Location::forStackmaps(&stackmaps, *this).restoreInto(jit, savedRegisters, result);
//}

void StackMaps::LiveOut::parse(StackMaps::ParseContext& context) {
  dwarfReg = DWARFRegister(context.view->read<uint16_t>(true)); // regnum
  context.view->read<uint8_t>(true); // reserved
  size = context.view->read<uint8_t>(true); // size in bytes
}

void StackMaps::LiveOut::dump(std::ostream& os) const {
  os << "(" << dwarfReg << ", " << size << ")";
}

bool StackMaps::Record::parse(StackMaps::ParseContext& context) {
  int64_t id = context.view->read<int64_t>(true);
  DCHECK(static_cast<int32_t>(id) == id);
  patchpointID = static_cast<uint32_t>(id);
  if (static_cast<int32_t>(patchpointID) < 0)
    return false;

  instructionOffset = context.view->read<uint32_t>(true);
  flags = context.view->read<uint16_t>(true);

  unsigned length = context.view->read<uint16_t>(true);
  while (length--)
    locations.push_back(std::move(readObject<Location>(context)));

  if (context.version >= 1)
    context.view->read<uint16_t>(true); // padding

  unsigned numLiveOuts = context.view->read<uint16_t>(true);
  while (numLiveOuts--)
    live_outs.push_back(readObject<LiveOut>(context));

  if (context.version >= 1) {
    if (context.view->offset() & 7) {
      DCHECK(!(context.view->offset() & 3));
      context.view->read<uint32_t>(true); // padding
    }
  }

  return true;
}

void StackMaps::Record::dump(std::ostream& os) const {
  os << "(#" << patchpointID << ", offset = "
      << instructionOffset << ", flags = "
      << flags << ", locations = "
      << "[" ;
  std::for_each(locations.begin(), locations.end(),
                [&os](const Location &n){ os << n << ", "; });
  os << "] live_outs = [";
  std::for_each(live_outs.begin(), live_outs.end(),
                [&os](const LiveOut &n){ os << n << ", "; });
  os << "])";
}
//
//RegisterSet StackMaps::Record::locationSet() const {
//  RegisterSet result;
//  for (unsigned i = locations.size(); i--;) {
//    Register reg = locations[i].dwarfReg.reg();
//    if (!reg) continue; // FIXME(llvm): what does it mean now?
//    result.set(reg);
//  }
//  return result;
//}
//
//RegisterSet StackMaps::Record::liveOutsSet() const {
//  RegisterSet result;
//  for (unsigned i = live_outs.size(); i--;) {
//    LiveOut liveOut = live_outs[i];
//    Reg reg = liveOut.dwarfReg.reg();
//    // FIXME: Either assert that size is not greater than sizeof(pointer), or actually
//    // save the high bits of registers.
//    // https://bugs.webkit.org/show_bug.cgi?id=130885
//    if (!reg) {
//      UNREACHABLE();
//    }
//    result.set(reg);
//  }
//  return result;
//}
//
//RegisterSet StackMaps::Record::usedRegisterSet() const {
//  RegisterSet result;
//  result.merge(locationSet());
//  result.merge(liveOutsSet());
//  return result;
//}

bool StackMaps::parse(DataView* view) {
  ParseContext context;
  context.view = view;

  version = context.version = context.view->read<uint8_t>(true);

  context.view->read<uint8_t>(true); // Reserved
  context.view->read<uint8_t>(true); // Reserved
  context.view->read<uint8_t>(true); // Reserved

  uint32_t numFunctions;
  uint32_t numConstants;
  uint32_t numRecords;

  numFunctions = context.view->read<uint32_t>(true);
  if (context.version >= 1) {
    numConstants = context.view->read<uint32_t>(true);
    numRecords = context.view->read<uint32_t>(true);
  }
  while (numFunctions--)
    stack_sizes.push_back(readObject<StackSize>(context));

  if (!context.version)
    numConstants = context.view->read<uint32_t>(true);
  while (numConstants--)
    constants.push_back(readObject<Constant>(context));

  if (!context.version)
    numRecords = context.view->read<uint32_t>(true);
  while (numRecords--) {
    Record record;
    if (!record.parse(context))
      return false;
    records.push_back(record);
  }

  return true;
}

void StackMaps::dump(std::ostream& os) const {
  os << "Version:" << version << ", StackSizes[";
  std::for_each(stack_sizes.begin(), stack_sizes.end(),
                [&os](const StackSize &n){ os << n << ", "; });
  os << "], Constants:[";
  std::for_each(constants.begin(), constants.end(),
                [&os](const Constant &n){ os << n << ", "; });
  os << "], Records:[";
  std::for_each(records.begin(), records.end(),
                [&os](const Record &n){ os << n << ", "; });
  os << "]";
}

void StackMaps::dumpMultiline(std::ostream& os, const char* prefix) const {
  os << prefix << "Version: " << version << "\n";
  os << prefix << "StackSizes:\n";
  for (unsigned i = 0; i < stack_sizes.size(); ++i)
    os << prefix << "    " << stack_sizes[i] << "\n";
  os << prefix << "Constants:\n";
  for (unsigned i = 0; i < constants.size(); ++i)
    os << prefix << "    " << constants[i] << "\n";
  os << prefix << "Records:\n";
  for (unsigned i = 0; i < records.size(); ++i)
    os << prefix << "    " << records[i] << "\n";
}

StackMaps::RecordMap StackMaps::computeRecordMap() const {
  // FIXME(llvm): seems that it's best to cache results (but see usage).
  RecordMap result;
  for (unsigned i = records.size(); i--;)
    result[records[i].patchpointID] = records[i]; // Careful (life-time)!
  return result;
}

unsigned StackMaps::stackSize() const {
  CHECK(stack_sizes.size() == 1);

  return stack_sizes[0].size;
}

} } // namespace v8::internal

//namespace WTF {
//
//using namespace JSC::FTL;
//
//void printInternal(PrintStream& out, StackMaps::Location::Kind kind) {
// Style!
//    switch (kind) {
//    case StackMaps::Location::Unprocessed:
//        out.print("Unprocessed");
//        return;
//    case StackMaps::Location::Register:
//        out.print("Register");
//        return;
//    case StackMaps::Location::Direct:
//        out.print("Direct");
//        return;
//    case StackMaps::Location::Indirect:
//        out.print("Indirect");
//        return;
//    case StackMaps::Location::Constant:
//        out.print("Constant");
//        return;
//    case StackMaps::Location::ConstantIndex:
//        out.print("ConstantIndex");
//        return;
//    }
//    dataLog("Unrecognized kind: ", static_cast<int>(kind), "\n");
//    RELEASE_ASSERT_NOT_REACHED();
//}
//
//} // namespace WTF
//
