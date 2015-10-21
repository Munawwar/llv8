#include "src/x64/assembler-x64-inl.h"

namespace v8 {
namespace internal {

// FIXME(llvm): our rebase to a more recent trunk has rendered
// pretty much everything in here useless. So get rid of it!
class StackMapReg {
 public:
   StackMapReg()
     :index_ (-1) {}

  static StackMapReg FromIndex(unsigned index) {
     StackMapReg result;
     result.index_ = index;
     return result;
  }
  bool IsIntReg() {
     return index_ < Register::kNumRegisters;
  }

  bool IsDoubleReg() {
     return index_ >= kFirstXMMRegNumber
         && index_ - kFirstXMMRegNumber < XMMRegister::kMaxNumRegisters;
  }

  Register IntReg() {
     DCHECK(IsIntReg());
     int const map[] = { 0, 2, 1, 3, 6, 7, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15 };
     return Register::from_code(map[index_]);
  }

  XMMRegister XMMReg() {
     DCHECK(IsDoubleReg());
     return XMMRegister::from_code(index_ - kFirstXMMRegNumber);
  }

  const char* ToString() {
    if (IsIntReg())
      return IntReg().ToString();
    else if (IsDoubleReg())
      return XMMReg().ToString();
    else
      UNREACHABLE();
    return "unknown";
  }

 private:
     int index_;
     static const int kFirstXMMRegNumber = 17;
};

} } // v8::internal
