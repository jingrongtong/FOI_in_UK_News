import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df=pd.read_csv('file.csv',encoding='Latin-1')
df.head

heatmap_data = df.pivot_table(index='Ngram', columns='Newspaper', values='Frequency', aggfunc='sum').fillna(0)

top_ngrams_heatmap = heatmap_data.loc[heatmap_data.sum(axis=1).nlargest(20).index]

plt.figure(figsize=(14, 8))
sns.heatmap(top_ngrams_heatmap, annot=True, cmap='coolwarm', fmt='.0f')
plt.title('Heatmap')
plt.xlabel('Newspaper')
plt.ylabel('Frequency')

plt.show()
