import matplotlib
matplotlib.use('qt5agg')  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim import corpora, models
import pandas as pd
import seaborn as sns


df=pd.read_csv(r'file.csv',encoding='Latin-1')


topic_columns = [col for col in df.columns if col.startswith('Topic')]

grouped = df.groupby('Year').agg({col: ['sum', 'count'] for col in topic_columns})

for col in topic_columns:
    grouped[(col, 'average')] = grouped[(col, 'sum')].to_numpy() / grouped[(col, 'count')].to_numpy()

grouped_averages = grouped.xs('average', level=1, axis=1)

grouped_averages.columns = [col.split(':')[1] if ':' in col else col for col in grouped_averages.columns]


plt.figure(figsize=(12, 8))
sns.heatmap(grouped_averages.T, cmap='YlGnBu', annot=True, fmt='.2f', cbar_kws={'label': 'Average Value'})

plt.title('Heatmap of Prevalence of Topics by Year')
plt.xlabel('Year')
#plt.ylabel('Topics')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Display the plot
plt.tight_layout()
plt.show()
