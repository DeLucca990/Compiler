
var A int = 2
func myprint(text string) {
  Println(text)
}

func fac(x int) int {
  if x == 1 {
    return 1
  }
  return x * fac(x-1)
}
func sum(x int, y int) int {
  return x + y
}

func tautology() bool {
  return true
}

func main() {
  var x_1 int
  x_1 = Scan()
  var x_2 int = fac(4)
  Println(A) 
  Println(x_2)
  {
    var x_2 int = 7
    x_1 = 9
    A = 8
    Println(x_2)
  }
  Println(A)
  Println(x_1)
  
  if (x_1 > 1 && !!!(x_1 < 1)) || x_1 == 9 {
    x_1 = 2
  }
  
  var x int = 3+6/3   *  2 -+-  +  2*4/2 + 0/1 -((6+ ((4)))/(2)) // Teste // Teste 2
  var y_1 int = 3
  y_1 = sum(y_1, x_1)
  var z__ int
  z__ = x + y_1
  
  if x_1 == 2 {
    x_1 = 2
  }
  
  if x_1 == 3 {
    x_1 = 2
  } else {
    x_1 = 3
  }
  
  x_1 = 0
  for x_1 < 1 || x == 2 {
    Println(x_1)
    x_1 = x_1 + 1
  }
  
  
  
  // Saida final
  Println(x_1)
  Println(x)
  Println(z__+1)
  
  // All int operations
  var y int = 2
  var z int
  z = (y - 1)
  Println(y+z)
  Println(y-z)
  Println(y*z)
  Println(y/z)
  Println(y == z)
  Println(y < z)
  Println(y > z)
  
  // All str operations 
  var a string
  var b string
  
  x_1 = 1
  y = 1 
  z = 2
  a = "abc"
  b = "def"
  myprint(a+b)
  myprint(a)
  Println(a+x_1)
  Println(x_1+a)
  Println(a+(x_1==1))
  Println(a == a)
  Println(a < b)
  Println(a > b)
}