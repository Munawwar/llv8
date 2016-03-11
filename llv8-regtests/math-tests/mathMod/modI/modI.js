var out;
function foo (x,y) {
  x+=1;
  return x%y;
}

for(var i =0; i<10000; i++) {
  out = foo(i, 3)
  print(out);
}
print(out);
