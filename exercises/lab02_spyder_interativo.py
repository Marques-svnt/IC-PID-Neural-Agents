# %% [1] Importação das bibliotecas necessárias
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ativar visualização de todas as colunas no console
pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 1000)

print("Passo 1 concluído: Bibliotecas importadas.")

# %% [2] Definição clara de todas as constantes e parâmetros do processo
TSP = 80.0          # Setpoint de temperatura (°C)
Kp = 2.0            # Ganho proporcional (%/°C)
u0 = 33.3           # Abertura nominal de bias da válvula (%)
Kv = 0.01           # Constante da válvula (kg/(min·%))
lambda_v = 2.0e6    # Calor latente de vaporização do vapor (J/kg)
cp = 4000.0         # Calor específico do fluido de processo (J/(kg·°C))
Tin = 30.0          # Temperatura de entrada da corrente fria (°C)
Tamb = 25.0         # Temperatura ambiente (°C)
UL = 20.0           # Coeficiente global de perdas para o ambiente (W/°C)
C = 1.0e5           # Capacidade térmica do sistema (J/°C)

print("Passo 2 concluído: Constantes do processo definidas.")

# %% [3] Importação do arquivo CSV e inicialização do DataFrame
caminho_csv = Path("dados_temperatura_trocador.csv")
df = pd.read_csv(caminho_csv)

# Visualizar as colunas iniciais (tempo e Tpv)
print("\n--- Visualização Inicial do DataFrame ---")
print(df[["tempo", "Tpv"]].head())

# %% [4] Perturbação de carga térmica: criação da coluna m_in
# m_in = 3.0 kg/min para t < 10 min | m_in = 4.0 kg/min para t >= 10 min
df["m_in"] = np.where(df["tempo"] < 10.0, 3.0, 4.0)

print("\n--- Coluna m_in adicionada ---")
print(df[["tempo", "Tpv", "m_in"]].iloc[[0, 9, 10, 11]])

# %% [5] Cálculo do Erro de Controle (e_k)
# e_k = TSP - TPV,k
df["e"] = TSP - df["Tpv"]

print("\n--- Coluna 'e' (Erro de controle) adicionada ---")
print(df[["tempo", "Tpv", "e"]].head())

# %% [6] Cálculo da Ação Proporcional do Controlador (u_k)
# u_k = Kp * e_k
df["u"] = Kp * df["e"]

print("\n--- Coluna 'u' (Ação do controlador em %) adicionada ---")
print(df[["tempo", "e", "u"]].head())

# %% [7] Cálculo da Abertura Total da Válvula de Vapor com Saturação (uv,k)
# uv,k = u0 + u_k com saturação física entre [0, 100]%
df["uv"] = np.clip(u0 + df["u"], 0.0, 100.0)

print("\n--- Coluna 'uv' (Abertura da válvula em %) adicionada ---")
print(df[["tempo", "u", "uv"]].head())

# %% [8] Cálculo da Vazão Mássica de Vapor (m_v,k em kg/min)
# m_v,k = Kv * uv,k
df["m_v"] = Kv * df["uv"]

print("\n--- Coluna 'm_v' (Vazão de vapor em kg/min) adicionada ---")
print(df[["tempo", "uv", "m_v"]].head())

# %% [9] Cálculo da Potência Térmica Fornecida pelo Vapor (Q_v em W)
# Q_v,k = (m_v,k / 60) * lambda_v
df["Q_v"] = (df["m_v"] / 60.0) * lambda_v

print("\n--- Coluna 'Q_v' (Potência do vapor em W) adicionada ---")
print(df[["tempo", "m_v", "Q_v"]].head())

# %% [10] Cálculo da Potência Térmica Transferida para a Corrente de Processo (Q_processo em W)
# Q_processo,k = (m_in,k / 60) * cp * (TPV,k - Tin)
df["Q_processo"] = (df["m_in"] / 60.0) * cp * (df["Tpv"] - Tin)

print("\n--- Coluna 'Q_processo' (Potência transferida em W) adicionada ---")
print(df[["tempo", "m_in", "Q_processo"]].head())

# %% [11] Cálculo das Perdas Térmicas para o Ambiente (Q_perdas em W)
# Q_perdas,k = UL * (TPV,k - Tamb)
df["Q_perdas"] = UL * (df["Tpv"] - Tamb)

print("\n--- Coluna 'Q_perdas' (Perdas para o ambiente em W) adicionada ---")
print(df[["tempo", "Tpv", "Q_perdas"]].head())

# %% [12] Cálculo da Potência Térmica Líquida (Q_liq em W)
# Q_liq,k = Q_v,k - Q_processo,k - Q_perdas,k
df["Q_liq"] = df["Q_v"] - df["Q_processo"] - df["Q_perdas"]

print("\n--- Coluna 'Q_liq' (Potência líquida em W) adicionada ---")
print(df[["tempo", "Q_v", "Q_processo", "Q_perdas", "Q_liq"]].head())

# %% [13] Cálculo da Taxa de Variação de Temperatura (dT/dt em °C/s e °C/min)
# (dT/dt)_s = Q_liq / C  |  (dT/dt)_min = 60 * (dT/dt)_s
df["dT_dt_s"] = df["Q_liq"] / C
df["dT_dt_min"] = 60.0 * df["dT_dt_s"]

print("\n--- Colunas de Taxa de Variação da Temperatura adicionadas ---")
print(df[["tempo", "Q_liq", "dT_dt_s", "dT_dt_min"]].head())

# %% [14] Visualização Completa da Tabela Final no Console
print("\n" + "="*80)
print("TABELA COMPLETA DE VARIÁVEIS COMPUTADAS (LABORATÓRIO 02)")
print("="*80)
print(df.to_string(index=False))

# Salvar arquivo com os resultados
df.to_csv("resultados_trocador_calculados.csv", index=False)
print("\nTabela salva com sucesso em 'resultados_trocador_calculados.csv'!")

# %% [15] Gráficos de Análise e Interpretação

# Gráfico 1: Tpv x t com linha de Setpoint
plt.figure(figsize=(7, 4), dpi=150)
plt.plot(df["tempo"], df["Tpv"], "bo-", label=r"$T_{PV}$ Medida")
plt.axhline(TSP, color="red", linestyle="--", label=f"Setpoint $T_{{SP}} = {TSP}^\\circ$C")
plt.axvline(10, color="gray", linestyle=":", label="Perturbação ($t = 10$ min)")
plt.title(r"Resposta Temporal da Temperatura $T_{PV} \times t$")
plt.xlabel("Tempo (min)")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Gráfico 2: dT/dt x t
plt.figure(figsize=(7, 4), dpi=150)
plt.plot(df["tempo"], df["dT_dt_min"], "go-", label=r"$\frac{dT}{dt}$ (°C/min)")
plt.axhline(0, color="black", linestyle="-", alpha=0.7)
plt.axvline(10, color="gray", linestyle=":", label="Perturbação ($t = 10$ min)")
plt.title(r"Taxa de Variação da Temperatura $\frac{dT}{dt} \times t$")
plt.xlabel("Tempo (min)")
plt.ylabel(r"$\frac{dT}{dt}$ (°C/min)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Gráfico 3: Balanço de Potências Térmicas (Q_v, Q_processo, Q_perdas)
plt.figure(figsize=(7, 4), dpi=150)
plt.plot(df["tempo"], df["Q_v"], "r-o", label=r"$\dot{Q}_v$ (Vapor)")
plt.plot(df["tempo"], df["Q_processo"], "b-^", label=r"$\dot{Q}_{processo}$ (Processo)")
plt.plot(df["tempo"], df["Q_perdas"], "y-v", label=r"$\dot{Q}_{perdas}$ (Perdas)")
plt.axvline(10, color="gray", linestyle=":", label="Perturbação ($t = 10$ min)")
plt.title("Balanço de Potências Térmicas no Trocador de Calor")
plt.xlabel("Tempo (min)")
plt.ylabel("Potência Térmica (W)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
