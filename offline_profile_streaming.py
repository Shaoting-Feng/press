import pandas as pd

df = pd.read_csv('qmsum/results_rate_06_processed.csv')
average_rougel = df['ROUGEL'].mean()
print(f"ROUGEL 平均值：{average_rougel}")
