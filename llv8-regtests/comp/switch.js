// Fails when llvm Optimize is on
// LLVM creates switch op during opt
// Test fails becouse of worng jmp dest

function foo(a, b, c) {
    switch (a) {
       case 0:
           a++;
           break;
       case 1:
           a += a;
           break;
       case 2:
           a *= a;
           break;
       case 3:
           a += b;
           break;
       default:
           return c
    }
    return b;
}

for (var i =0; i < 50000; i++)
  foo(3, i+1, i+2);
