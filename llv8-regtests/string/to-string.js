function foo(x){
  x+=1;
  print(i.toString());
}

for (var i = 0; i<1000000; i++)
  foo(i);
