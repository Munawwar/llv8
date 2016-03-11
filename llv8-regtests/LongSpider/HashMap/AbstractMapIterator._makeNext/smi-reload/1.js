function MyObj() {
    this.field = 11 
}

MyObj.prototype.foo = function() {
    var smi = this.field + 1
    this.field = smi
}


a = new MyObj()

var t = 0
for (var i = 0; i < 10000000; i++)
    a.foo()

print (a.field) 
