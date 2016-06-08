LLV8
=============
LLV8 is an experimental top-tier compiler for V8. It leverages the power of LLVM MCJIT to produce highly optimized code. It is supposed to be used as a third tier for cases where it makes sense to spend more time compiling to achieve higher throughput.

LLV8 (backend) is implemented as a patch to V8 and it cannot function without the virtual machine. Although LLV8 is only a fraction of the entire patched VM, we also refer to the whole thing (our fork of V8) as LLV8.

LLV8 codebase effectively consists of two repositories, both of which are hosted at github:
  - [LLVM fork ](https://github.com/ispras/llvm-for-v8)
  - [V8 fork (LLV8)](https://github.com/ispras/llv8)

Building LLV8
=============
Building instructions can be found in the project's [wiki](https://github.com/ispras/llv8/wiki/Building%20with%20Gyp). They are duplicated in this readme for convenience. 

We are going to check out sources of LLV8 and the modified LLVM. V8 (and thus LLV8) comes with binaries of a clang compiler, so we are going to use it to build both LLVM and LLV8 to avoid linking problems.

### Checking out LLV8

The easiest way is to follow the process of building regular V8, adding the LLV8 branch as a remote.

```
cd $LLV8_ROOT # Choose any directory you want. 
# E.g. cd /some/dir && export LLV8_ROOT=`pwd`
```

Install [depot_tools](http://www.chromium.org/developers/how-tos/install-depot-tools):
```
git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
export PATH=`pwd`/depot_tools:"$PATH"
```
Fetch all the code:
```
fetch v8
cd v8
git remote add llv8 https://github.com/ispras/llv8.git
git checkout -b llv8 llv8/llv8
gclient sync
```
Note that we don't run make yet, since first we need to build LLVM libraries to link against.
We had to check out V8 first to obtain the clang compiler though.

### LLVM

Check out and build our version of LLVM:
```
cd $LLV8_ROOT
git clone https://github.com/ispras/llvm-for-v8.git
mkdir build-llvm
cd build-llvm
export CC=$LLV8_ROOT/v8/third_party/llvm-build/Release+Asserts/bin/clang
export CXX=$LLV8_ROOT/v8/third_party/llvm-build/Release+Asserts/bin/clang++
../llvm-for-v8/configure --enable-assertions --enable-optimized --disable-zlib
make -j9
sudo make install # Note: this installs the built version of llvm system-wide.
```
You can in theory pass `--prefix` to configure or not install llvm at all and use it from the build directory, because all you need to build llv8 is the `llvm-configure` of this freshly built llvm in your `$PATH`.
But this makes the subsequent compilation of llv8 a bit more involved (the C++ compiler spews warnings-errors as it compiles LLVM headers).

### Building LLV8

Finally, run make (substitute "release" for "debug" if you'd like to test performance):
```
cd $LLV8_ROOT/v8
export LINK=$CXX
make x64.debug -j9 i18nsupport=off gdbjit=off
```

Project documentation
=============

Design documentation, building and runnig insructions can be found on
[LLV8 wiki](https://github.com/ispras/llv8/wiki).

Usage example
=============
Let's compile a simple piece of JavaScript code with LLV8.

```
cat > a-plus-b.js
```
```
var N = 10000; // Should be big enough to pass optimization threshold.

function foo(a, b) {
    return a + b;
}

var k = 1;
for (var i = 0; i < N; i++) {
    k += foo(i, i % k);
}

print(k);
```
Now run it through LLV8:
```
$LLV8_ROOT/v8/out/x64.debug/d8 a-plus-b.js --llvm-filter=*
```
It should spew a lot of debug information to stderr, mostly LLVM IR before and after various passes and disassembly. Here is an abridged stderr output:
```
...
====================vvv Module  AFTER optimization vvv====================
; ModuleID = '0'
target triple = "x86_64-unknown-linux-gnu"

define x86_64_v8cc i8* @"0"(i8* %pointer_0, i8* nocapture readnone, i8* nocapture readnone, i8* %pointer_1, i8* %pointer_2, i8* %pointer_3) #0 gc "v8-gc" {
BlockEntry0:
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 0, i32 0)
  %2 = ptrtoint i8* %pointer_2 to i64
  %3 = and i64 %2, 1
  %4 = icmp eq i64 %3, 0
  br i1 %4, label %BlockCont, label %DeoptBlock

BlockCont:                                        ; preds = %BlockEntry0
  %5 = ptrtoint i8* %pointer_1 to i64
  %6 = and i64 %5, 1
  %7 = icmp eq i64 %6, 0
  br i1 %7, label %BlockCont1, label %DeoptBlock2

DeoptBlock:                                       ; preds = %BlockEntry0
  %8 = tail call i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 1, i32 5, i8* null, i32 0, i8* %pointer_3, i8* %pointer_2, i8* %pointer_1, i8* %pointer_0)
  unreachable

BlockCont1:                                       ; preds = %BlockCont
  %9 = tail call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %2, i64 %5)
  %10 = extractvalue { i64, i1 } %9, 1
  br i1 %10, label %DeoptBlock4, label %BlockCont3

DeoptBlock2:                                      ; preds = %BlockCont
  %11 = tail call i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 2, i32 5, i8* null, i32 0, i8* %pointer_3, i8* %pointer_2, i8* %pointer_1, i8* %pointer_0)
  unreachable

BlockCont3:                                       ; preds = %BlockCont1
  %12 = extractvalue { i64, i1 } %9, 0
  %13 = inttoptr i64 %12 to i8*
  ret i8* %13

DeoptBlock4:                                      ; preds = %BlockCont1
  %14 = tail call i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 3, i32 5, i8* null, i32 0, i8* %pointer_3, i8* %pointer_2, i8* %pointer_1, i8* %pointer_0)
  unreachable
}

declare void @llvm.experimental.stackmap(i64, i32, ...)

declare i64 @llvm.experimental.patchpoint.i64(i64, i32, i8*, i32, ...)

; Function Attrs: nounwind readnone
declare { i64, i1 } @llvm.sadd.with.overflow.i64(i64, i64) #1

attributes #0 = { "no-frame-pointer-elim"="true" "put-constantpool-in-fn-section"="true" "put-jumptable-in-fn-section"="true" }
attributes #1 = { nounwind readnone }
====================^^^ Module  AFTER optimization ^^^====================
...
  Version: 1
  StackSizes:
      (off:48232192, size:24)
  Constants:
  Records:
      (#0, offset = 18, flags = 0, locations = [] live_outs = [])
      (#1, offset = 42, flags = 0, locations = [(Register, rcx, off:0, size:8), (Register, rdx, off:0, size:8), (Register, rbx, off:0, size:8), (Register, rsi, off:0, size:8), ] live_outs = [])
      (#2, offset = 47, flags = 0, locations = [(Register, rcx, off:0, size:8), (Register, rdx, off:0, size:8), (Register, rbx, off:0, size:8), (Register, rsi, off:0, size:8), ] live_outs = [])
      (#3, offset = 52, flags = 0, locations = [(Register, rcx, off:0, size:8), (Register, rdx, off:0, size:8), (Register, rbx, off:0, size:8), (Register, rsi, off:0, size:8), ] live_outs = [])
Instruction start: 0xa31d33a040
0xa31d33a040		push	rbp
0xa31d33a041		mov	rbp, rsp
0xa31d33a044		push	rsi
0xa31d33a045		push	rdi
0xa31d33a046		mov	rcx, qword ptr [rbp + 0x20]
0xa31d33a04a		mov	rdx, qword ptr [rbp + 0x18]
0xa31d33a04e		mov	rbx, qword ptr [rbp + 0x10]
0xa31d33a052		test	dl, 0x1
0xa31d33a055		jne	0x13
0xa31d33a057		test	bl, 0x1
0xa31d33a05a		jne	0x13
0xa31d33a05c		mov	rax, rdx
0xa31d33a05f		add	rax, rbx
0xa31d33a062		jo	0x10
0xa31d33a064		pop	rdi
0xa31d33a065		pop	rsi
0xa31d33a066		pop	rbp
0xa31d33a067		ret	0x18
0xa31d33a06a		call	-0x33406f
0xa31d33a06f		call	-0x33406a
0xa31d33a074		call	-0x334065
...
RelocInfo (size = 6)
0xa31d33a06b  runtime entry  (deoptimization bailout 0)
0xa31d33a070  runtime entry  (deoptimization bailout 1)
0xa31d33a075  runtime entry  (deoptimization bailout 2)

Safepoints (size = 8)
```
And of course the answer is printed to stdout.
```
99989998
```
