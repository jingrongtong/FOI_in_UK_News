import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('qt5agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import textwrap


df=pd.read_csv(r"file.csv", encoding='Latin-1')


topic_columns = [col for col in df.columns if col.startswith('Topic')]
df[topic_columns] = df[topic_columns].apply(pd.to_numeric, errors='coerce')

df = df.dropna(subset=topic_columns)

correlation_matrix = df[topic_columns].corr()

threshold = 0.09

significant_correlations = correlation_matrix[abs(correlation_matrix) > threshold].stack().reset_index()

significant_correlations.columns = ['Topic_1', 'Topic_2', 'Correlation']

significant_correlations = significant_correlations[significant_correlations['Topic_1'] < significant_correlations['Topic_2']]


G = nx.Graph()


for _, row in significant_correlations.iterrows():
    G.add_edge(row['Topic_1'], row['Topic_2'], weight=row['Correlation'])

def preprocess_labels(label):
    if ':' in label:
        return label.split(':', 1)[-1].strip()  
    return label


node_labels = {node: preprocess_labels(node) for node in G.nodes()}


def wrap_label(label, width=20):
    return '\n'.join(textwrap.wrap(label, width=width))


wrapped_labels = {node: wrap_label(label) for node, label in node_labels.items()}


plt.figure(figsize=(16, 14))  


pos = nx.circular_layout(G)

edge_weights = nx.get_edge_attributes(G, 'weight')


max_weight = max(edge_weights.values(), default=1)  

widths = [edge_weights[edge] / max_weight * 4 for edge in G.edges()]  


nx.draw(
    G,
    pos,
    node_color='lightblue',
    node_size=2000,  # Decrease node size to reduce overlap
    edge_color='gray',
    linewidths=0.5,
    width=widths  # Apply scaled edge thickness
)


nx.draw_networkx_labels(G, pos, labels=wrapped_labels, font_size=10, verticalalignment='center', horizontalalignment='center')

plt.title('Correlation Network of Topic Columns')

plt.show()
