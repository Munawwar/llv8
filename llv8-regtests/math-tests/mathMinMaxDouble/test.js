function foo (arg1, arg2) {
  arg1 += 0.2;
  arg2 += 0.2;
  return Math.max(arg1, arg2);
}

var x;
for (var i = 0; i<1000; i++) {
 x = foo(-4, -6); 
}

print(x);
