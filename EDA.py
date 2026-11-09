import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('data/clean/orders.csv')

sns.histplot(df['order_amount'], kde=True, color='purple', line_kws={'color': 'yellow', 'linewidth': 2})
plt.axvline(df['order_amount'].mean(), color='red', linestyle='--', label='Mean')
plt.axvline(df['order_amount'].median(), color='green', linestyle='-', label='Median')
plt.legend()
plt.show()

print("Skewness:", df['order_amount'].skew())

sns.boxplot(x=df['order_amount'])
plt.title('Order Amount: Quartiles, 1.5xIQR Fences, Outliers')
plt.show()

q1 = df['order_amount'].quantile(0.25)
q3 = df['order_amount'].quantile(0.75)
iqr = q3 - q1

lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

below = (df['order_amount'] < lower_fence).sum()
above = (df['order_amount'] > upper_fence).sum()

print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}")
print(f"Lower fence: {lower_fence.round(2)}, Upper fence: {upper_fence.round(2)}")
print(f"Outliers below lower fence: {below}")
print(f"Outliers above upper fence: {above}")


completed = df[df['status'] == 'completed']
cols = ['order_amount', 'discount_pct', 'quantity', 'delivery_minutes']

corr = completed[cols].corr()

sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix: Completed Orders')
plt.show()

# Find the pair with the largest absolute correlation (ignoring the diagonal)
abs_corr = corr.abs().where(~np.eye(len(cols), dtype=bool))
pair = abs_corr.stack().idxmax()
print(f"Largest |correlation|: {pair} = {corr.loc[pair]:.4f}")

#Check README.md file for analysis of data from the obtained EDA
