
import pandas as pd

data = [[1,2,3,4,5],
        [6,7,8,9,10],
        [11,12,13,14,15],
        [16,17,18,19,20],
        [21,22,23,24,25]] # Example data
df = pd.DataFrame(data, columns=['A',"B","C","D","E"])

print(df[["A","C"]].values)  # This will print the values of the DataFrame as a NumPy array
print(df)

import matplotlib.pyplot as plt
#df.plot(kind="scatter", grid=True,x="A",y="C")  # This will plot the values of the DataFrame as a scatter plot
df.plot(kind="line", grid=True,x="A",y="C")  # This will plot the values of the DataFrame as a scatter plot
plt.show()  # This will plot the values of the DataFrame as a line plot