
import pandas as pd

# 从一个列表/一维数组，构建 dataframe
a = [1.1, 4.4, 3.2, 5.8]
#print(a)

df_a = pd.DataFrame(a, columns=["A"], index=['a','b','c','d'])
#print(df_a)

df_a["B"] = df_a["A"] # 类似 Excel
#print(df_a)

df_a["C"] = df_a["B"] + 3.5
#print(df_a)

#print(df_a["C"])
#print(df_a.loc["C"])
#print(df_a.iloc[0])
#print(df_a.loc['a'])

#####################################################################
# 从二维列表/数组，构建 DataFrame
df_b = pd.DataFrame([[1, 2], [4, 5], [7, 8]], \
                    index=['cobra', 'viper', 'sidewinder'], \
                    columns=['max_speed', 'shield'])

print(df_b)


