
def func1(a, b):
  return a + b


if __name__=="__main__":
  c = func1(3,4)
  print("c = ", c)

  func2 = lambda x: func1(x,1)

  d = func2(100)  # 101
  print("d = ", d)