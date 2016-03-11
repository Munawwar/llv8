function foo (arg) {
  arg -= 0.5;
  return Math.abs(arg);
 }


var x= 0;
for (var i =0; i<1000; i++) {
  x = foo(-i);
}

print (x);
