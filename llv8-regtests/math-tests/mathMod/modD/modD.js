//var x = 1.2
function foo(y) {
//  y+=x;
  return y%1.3;
}

var r;
for (var i = 0; i< 100000; i++) {
  r = foo(i)
}
print(r);
