function foo(a, b, c) {
    p(a)
    return a 
}

function p(x) {
    var a = Math.random() // we don't support gc yet, so no doubles for now. // run with --trace-gc
    var b = Math.sin(a) * Math.sin(a)
    var c = Math.cos(a) * Math.cos(a)
    if (b + c > 1.1) print ("dfdfs")

    if (x) return 1
    return 0
}

//%NeverOptimizeFunction(p);
//%NeverOptimizeFunction(TimeFunc);
cnt = 0
function TimeFunc() {
    for (var i = 0; i < 1; i++) {
        var sum = 0;
        for(var x = 0; x < 50; x++)
            for(var y = 0; y < 256; y++)
                //sum += foo(64206, y + x) 
            sum += foo(64206, y, 48879); 
    }
    return sum;
}

sum = TimeFunc();

print (sum)
