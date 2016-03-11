// Fails when llvm Optimize is on
// LLVM creates switch op during opt
// Test fails becouse of worng jmp dest

function foo(a, b, c) {
    if (a == 0) {
         a++;
    } else if (a == 1) {
        a +=a;
    } else if (a == 2) {
        a*=a;
    } else if (a == 3) { 
        a+=b;
    } else {
        return c;
    }
    return b;
}

for (var i =0; i < 50000; i++)
  foo(3, i+1, i+2);
