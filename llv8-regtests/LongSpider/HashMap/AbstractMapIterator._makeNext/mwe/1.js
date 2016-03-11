function ArrObj() {
    this.arr = new Array(2)
}

function MyObj() {
    this.field = new ArrObj()
    this.field.arr[0] = 100500
}

MyObj.prototype.foo = function(x) {
   var ret = this.field.arr[0] 
   this.field2 = ret 
   return ret
}


a = new MyObj()

var t = 0
for (var i = 0; i < 10000000; i++)
    t += a.foo()

print (t)
