import pandas as pd

df = pd.read_csv('qmsum/results_rate_06_processed.csv')

# 过滤掉 ROUGEL == -1 的行
df_valid = df[df['ROUGEL'] != -1]

# 计算平均值
average_rougel = df_valid['ROUGEL'].mean()
print(f"ROUGEL 平均值（排除 -1）：{average_rougel}")

