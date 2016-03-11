function foo(a) {
//  if (a > 1) 
//      return new Construct(a)
//  else 
      return new Construct(10)
}

function Construct(x) {
  this.x = x;
}

var p
for (var i = 0; i < 10000; i++)
    p = foo(i)
