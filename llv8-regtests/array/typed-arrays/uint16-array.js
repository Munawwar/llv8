// FIXME(llvm): this test fails with --llvm-filter=*
function foo() {
    var arr = new Uint16Array(100);
    arr[0] = 3
    for (var i = 1; i < arr.length; i++) {
        arr[i] = arr[i - 1] * arr[i - 1]
    }
    return arr[arr.length / 4]
}

var a = 0
var ITER = 1000
for (var i = 1; i < ITER; i++)
    a += foo()
print(a)
