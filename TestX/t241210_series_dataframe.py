
import pandas as pd

s = pd.Series([1.1, 2.2, 3.3, 4.4])

#print(f"s = {s}, \n{type(s)}")

def largeThanThree(x):
  return x>3.0

# Series 类，具有 apply 函数，可以对内部每个元素应用一个“输出的函数”，比如 “largeThanThree”
t = s.apply(largeThanThree)
#print(f"{t},\n{type(t)}")

s1 = pd.Series([15, 2, 3, 100])

# two Series build a DataFrame
d = pd.DataFrame({"f":s, "d":s1})
#print(d)

d1 = d.loc[t] # 选择 3、4两行
#print(d1)

d2 = d.loc[~t]  # 选择除 3、4 意外的其它行
#print(d2)

z = d.reset_index() # 增加一属性列 “index”
print(z)