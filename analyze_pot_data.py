import pandas as pd

pot_data = pd.read_csv('output.txt', delimiter=',')

values = pot_data.iloc[:, 1]

print(f"Max: {values.max()}\n")
print(f"Min: {values.min()}\n")
print(f"Mean: {values.mean()}\n")
