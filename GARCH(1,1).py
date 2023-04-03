
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

#Dados do Ibov
ibov = pd.read_csv('ibov.csv', index_col='Data')

# Preparar os dados: limpe e organize os dados, converta-os em um formato
# que possa ser utilizado pelo modelo de previsão.
ibov_ret = np.log(ibov['Último']/ibov['Último'].shift(1)).dropna()*100

# Divisão dos dados. Aqui você poderá optar por escolher um período para análise!
train = ibov_ret[:'2021-01-01']
test = ibov_ret['2021-01-01':]

# Escolher o modelo: escolha o modelo de previsão que você deseja utilizar.
# Aqui, usaremos o modelo GARCH(1,1)

model = arch_model(train, p=1, q=1)

# Treinar o modelo: ajuste o modelo ao conjunto de treinamento
res = model.fit()

# Testar o modelo: avalie o desempenho do modelo utilizando o conjunto de teste.
volatility = res.forecast(horizon=len(test), reindex=False)

# Fazer a previsão: uma vez que o modelo tenha sido treinado e testado, você pode
# fazer previsões da volatilidade do IBOV usando dados recentes ou futuros.

plt.plot(test)
plt.plot(volatility.variance.dropna())
plt.title('Previsão de volatilidade do IBOV')
plt.legend(['Retornos', 'Previsão de volatilidade'])
plt.show()

