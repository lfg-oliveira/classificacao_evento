import pandas as pd
import numpy as np


df = pd.read_csv('Classificacao.tsv', sep='\t', header=0)
print(df['saida'].value_counts())
print(np.average(df['descricao'].apply(lambda x: len(x)*8)))
