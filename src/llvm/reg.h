#include "src/x64/assembler-x64-inl.h"
namespace v8 {
namespace internal { 
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
   return index_ - Register::kNumRegisters < XMMRegister::kMaxNumRegisters;
}

Register IntReg() {
   DCHECK(IsIntReg());
   int const map[] = { 0, 2, 1, 3, 6, 7, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15 };
   return Register::from_code(map[index_]);
}

XMMRegister XMMReg() {
   DCHECK(IsDoubleReg());
   XMMRegister reg { index_ - Register::kNumRegisters };  
   return reg;
}
const char* ToString() {
  return "unknown"; //FIXME: fix
}
private:
   int index_;
};
}}
