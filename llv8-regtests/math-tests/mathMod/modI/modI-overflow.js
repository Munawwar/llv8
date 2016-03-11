// Flags: --allow-natives-syntax --llvm-filter=* 

function foo(a, b) {
    return (a|0) % (b|0);
}


var total = 0
for (var a = 0; a < 100; a++)
    total += foo(a, a + 1);

%OptimizeFunctionOnNextCall(foo);

for (var a = 3; a < 100; a++)
    total += foo(a, a + 1);

var kMinInt = -0x7FFFFFFF - 1
total += foo(kMinInt, -1) // the essential part
print (total)
