# ML
#Conjunto de dados com mais de 80.000 registros
#Analise exploratória de variáveis categórias e numéricas
#Analise e tratamento de valores missing (nulos)
#Analise estatística de variáveis
#Tratamento de Dados 
#Engenharia de Atributos    
#Gráficos
#Outliers
#Normalização e Padronização de Dados
#Balanceamento da variável ALVO (TARGET)    
#OneHotEncoding
#Criação, treino e teste dos modelos preditivos com 3 algoritmos diferentes (Random Forest, Suport Vector Machine e KNN
#GridSearch para ajustes de hiperparametros automáticos e treino de mais de 1.000 modelos    
#Analise dos pesos das melhores variáveis    

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore") 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.2f}'.format

#Importação do arquivo de dados

#df_original=pd.read_csv("dados_coletados_20k.csv")
#df_original=pd.read_csv("dados_coletados_80k.csv")

df_original = pd.read_csv("dados_coletados10k.csv")

#Tamanho do conjunto de dados. xx.xxx linhas e xx variáveis
df_original.shape

#Visão geral do conjunto de dados
df_original.head()

#Avaliar o período dos dados coletados
inicio = pd.to_datetime(df_original['Data_Contratacao']).dt.date.min()
fim = pd.to_datetime(df_original['Data_Contratacao']).dt.date.max()
print('Período dos dados - De:', inicio, 'Até:',fim)

# Verificando se há valores nulos (dados missing)
df_original.isnull().sum()

#Informações básicas sobre tipos de variáveis
df_original.info(verbose=True)

# Total de valores únicos de cada variável
# A variável CONTRATO é um valor único para cada registro, pois refere-se ao Contrato do Cliente
valores_unicos = []
for i in df_original.columns[0:24].tolist():
    print(i, ':', len(df_original[i].astype(str).value_counts()))
    valores_unicos.append(len(df_original[i].astype(str).value_counts()))

# Visualizando algumas medidas estatisticas.
df_original.describe()

# Avaliando o maior e menor valor da variavel Valor_Renda
print('Maior Renda:', df_original['Valor_Renda'].max())
print('Menor Renda:', df_original['Valor_Renda'].min())

# Avaliando o maior e menor valor da variavel QT_Dias_Atraso
print('Maior quantidade de dias atraso: ', df_original['QT_Dias_Atraso'].max())
print('Menor quantidade de dias atraso: ', df_original['QT_Dias_Atraso'].min())

# Avaliando o maior e menor valor da variavel Prazo_Restante
print('Maior quantidade de dias restante: ', df_original['Prazo_Restante'].max())
print('Menor quantidade de dias restante: ', df_original['Prazo_Restante'].min())

# Quantidade de dias em atraso
df_original.groupby(['QT_Dias_Atraso']).size()

# Prazo emprestimo
df_original.groupby(['Prazo_Emprestimo']).size()

# Prazo Restante
df_original.groupby(['Prazo_Restante']).size()

# Sexo
df_original.groupby(['Sexo']).size()

# UF dos Clientes
df_original.groupby(['UF_Cliente']).size()

# Idade dos clientes
df_original.groupby(['Idade']).size()

# Estado civil dos clientes
df_original.groupby(['Estado_Civil']).size()

# Escolaridade dos clientes
df_original.groupby(['Escolaridade']).size()

# Patrimonio dos clientes
df_original.groupby(['Possui_Patrimonio']).size()

# Valor do patrimonio dos clientes
df_original.groupby(['VL_Patrimonio']).size()

# Variavel TARGET - ALVO
df_original.groupby(['Possivel_Fraude']).size()

### Tratando os dados que identificamos que precisam ser ajustados em nossa analise acima
# Ajustando ESTADO_CIVIL
df_original['Estado_Civil'] = df_original['Estado_Civil'].replace(['NENHUM'], 'OUTRO')
df_original['Estado_Civil'] = df_original['Estado_Civil'].replace(['UNIÃO ESTAVEL'], 'CASADO (A)')

df_original.groupby(['Estado_Civil']).size()

# Criando faixa etaria para utilizarmos no modelo preditivo
bins = [0, 21, 30, 40, 50, 60, 100]
labels = ['Até 21 Anos', 'De 22 até 30 Anos', 'De 31 até 40 Anos', 'De 41 até 50 Anos', 'De 51 até 60', 'Acima de 60 Anos']
df_original['Faixa_Etaria'] = pd.cut(df_original['Idade'], bins=bins, labels=labels)
df_original.groupby(['Faixa_Etaria']).size()

df_original.head()

# Criando faixa salarial para utilizarmos no modelo preditivo
bins = [-100, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 9000000000]
labels = ['Até 1k', 'De 1k até 2k', 'De 2k até 3k', 'De 3k até 5k', 'De 5k até 10k', 'De 10k até 20k',
          'De 20k até 30k', 'Acima de 50k']
df_original['Faixa_Salarial'] = pd.cut(df_original['Valor_Renda'], bins=bins, labels=labels)
df_original.groupby(['Faixa_Salarial']).size()

# Precisamos tratar os valores nulos dessa variavel antes de fazermos nossa engenharia de atributos
# Vamos preencher os valores nulos usando a mediana dos dados
df_original['QT_Dias_Atraso'].median()

# Preenchendo os valores nulo com a mediana
df_original['QT_Dias_Atraso'] = df_original['QT_Dias_Atraso'].fillna((df_original['QT_Dias_Atraso'].median()))

# Criando faixa de dias em atraso da cota para utilizarmos no modelo preditivo
bins = [-100, 30, 60, 90, 180, 240, 360, 500]
labels = ['Até 30 dias', 'De 31 até 60', 'De 61 até 90', 'De 91 até 180', 'De 181 até 240','De 241 até 360', 'Acima de 360']
df_original['Faixa_Dias_Atraso'] = pd.cut(df_original['QT_Dias_Atraso'], bins=bins, labels=labels)
df_original.groupby(['Faixa_Dias_Atraso']).size()

# Criando faixa de prazo de emprestimo para utilizarmos no modelo preditivo
bins = [0, 60, 120, 200, 720]
labels = ['Até 60 Meses', 'De 61 até 120 Meses', 'De 121 até 200 Meses', 'Acima de 200 Meses']
df_original['Faixa_Prazo_Emprestimo'] = pd.cut(df_original['Prazo_Emprestimo'], bins=bins, labels=labels)
pd.value_counts(df_original.Faixa_Prazo_Emprestimo)

# Criando faixa de prazo restante do emprestimo para utilizarmos no modelo preditivo
bins = [-1, 60, 120, 200, 500]
labels = ['Até 60 Meses', 'De 61 até 120 Meses', 'De 121 até 200 Meses', 'Acima de 200 Meses']
df_original['Faixa_Prazo_Restante'] = pd.cut(df_original['Prazo_Restante'], bins=bins, labels=labels)
pd.value_counts(df_original.Faixa_Prazo_Restante)

### Agora após os ajustes vamos visualizar de forma gráfica para avaliarmos melhor

df_original.Sexo.value_counts().plot(kind='bar', title='Sexo',color = ['#1F77B4', '#FF7F0E']);

df_original.UF_Cliente.value_counts().plot(kind='bar', title='UF Cliente',color = ['#1F77B4', '#FF7F0E']);

df_original.Faixa_Prazo_Emprestimo.value_counts().plot(kind='bar', title='Prazo Emprestimo',color = ['#1F77B4', '#FF7F0E']);

df_original.Faixa_Prazo_Restante.value_counts().plot(kind='bar', title='Prazo Restante',color = ['#1F77B4', '#FF7F0E']);

df_original.Estado_Civil.value_counts().plot(kind='bar', title='Estado Civil',color = ['#1F77B4', '#FF7F0E']);

df_original.Escolaridade.value_counts().plot(kind='bar', title='Escolaridade',color = ['#1F77B4', '#FF7F0E']);

df_original.Faixa_Dias_Atraso.value_counts().plot(kind='bar', title='Faixa Dias Atraso',color = ['#1F77B4', '#FF7F0E']);

df_original.Faixa_Salarial.value_counts().plot(kind='bar', title='Faixa Salarial',color = ['#1F77B4', '#FF7F0E']);

df_original.Possui_Patrimonio.value_counts().plot(kind='bar', title='Possui Patrimonio',color = ['#1F77B4', '#FF7F0E']);

df_original.Faixa_Etaria.value_counts().plot(kind='bar', title='Faixa Etaria',color = ['#1F77B4', '#FF7F0E']);

#Analisando como a variavel alvo está distribuida.
#Aqui podemos observar que há muito mais cotas como NÃO POSSÍVEL FRAUDE
#dessa forma, precisaremos balancear o dataset mais adiante.
df_original.Possivel_Fraude.value_counts().plot(kind='bar', title='Possíveis Fraudes',color = ['#1F77B4', '#FF7F0E']);

# Vamos visualizar novamente como está nosso DataFrame original após a engenharia de atributos
df_original.info(verbose=True)

# Vamos selecionar as colunas que iremos utilizar e algumas iremos descartar
df_original.columns

# APÓS ANALISE INICIAL QUE REALIZAMOS ACIMA, ENTENDEMOSO QUE ALGUMAS VARIÁVEIS NÃO POSSUEM RELEVANCIA.

#  Contrato --> Essa variável é a identificação de cada cliente
#  Data_Contratacao, VL_Patrimonio, Possui_Patrimonio, Escolaridade, Idade --> Essas não irão ter relevancia no modelo
#  Valor_Renda, Prazo_Emprestimo, QT_Dias_Atraso, Prazo_Restante --> Essas variáveis já transformamos em categoria


# Chamaremos nosso novo conjunto de dados de df_dados

columns = ['Sexo', 'UF_Cliente', 'Perc_Juros', 
       'VL_Emprestimo', 'VL_Emprestimo_ComJuros', 'QT_Total_Parcelas_Pagas',
       'QT_Total_Parcelas_Pagas_EmDia', 'QT_Total_Parcelas_Pagas_EmAtraso',
       'Qt_Renegociacao', 'Estado_Civil', 'QT_Parcelas_Atraso', 'Saldo_Devedor', 
       'Total_Pago', 'Faixa_Prazo_Restante', 'Faixa_Salarial', 'Faixa_Prazo_Emprestimo', 'Faixa_Etaria', 
       'Faixa_Dias_Atraso', 'Possivel_Fraude']

df_dados = pd.DataFrame(df_original, columns=columns)

df_dados.shape

df_dados.info(verbose = True)

# Vamos constatar que realmente não há valores nulos
df_dados.isnull().sum()
