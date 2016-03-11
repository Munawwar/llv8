var str;

function foo(i) {
  str=i+"asd";
}

for (var i =0; i<10000;i++ )
   foo(i)

print(str);
