# Laboratório de Exercícios — Controle de Processos

Ambiente estruturado de exercícios práticos, simulações computacionais e relatórios técnicos da disciplina de Controle de Processos / Iniciação Científica (UESC).

---

## 📁 Estrutura de Pastas

```
exercises/
├── data/                                 # Dados experimentais e de simulação
│   ├── raw/                              # Dados brutos de entrada
│   │   └── dados_temperatura_trocador.csv
│   └── processed/                        # Dados calculados e resultados
│       └── resultados_trocador_calculados.csv
│
├── figuras/                              # Imagens, esquemas do processo e enunciado
│   ├── trocador_calor_controle_temperatura.png
│   ├── trocador_calor_controle_temperatura_dados_1.png
│   └── trocador_calor_controle_temperatura_dados_2.png
│
├── plots/                                # Gráficos e figuras geradas pelas simulações
│   ├── grafico_1_temperatura_vs_tempo.png
│   ├── grafico_2_taxa_variacao_temperatura.png
│   ├── grafico_3_balanco_termico.png
│   └── painel_completo_controle_trocador.png
│
├── reports/                              # Relatórios técnicos em Markdown/LaTeX
│   └── relatorio_laboratorio_02.md
│
├── src/                                  # Código-fonte Python
│   ├── __init__.py
│   ├── heat_exchanger.py                 # Modelo modular (HeatExchangerAnalyzer)
│   └── lab02_spyder_interativo.py        # Script em células para o Spyder (F5/F9)
│
├── tests/                                # Testes unitários automatizados (pytest)
│   ├── __init__.py
│   ├── conftest.py                       # Configuração de sys.path para testes
│   └── test_heat_exchanger.py            # Testes do modelo de trocador
│
├── requirements.txt                      # Dependências Python
└── README.md                             # Este guia de organização
```

---

## 🚀 Como Executar

### 1. Executar a Simulação Completa (Pipeline Modular)
Gera automaticamente os dados em `data/processed/` e os gráficos em `plots/`:
```bash
python src/heat_exchanger.py
```

### 2. Executar no Spyder (Modo Interativo com Células `#%%`)
Abra o arquivo [`src/lab02_spyder_interativo.py`](file:///src/lab02_spyder_interativo.py) no Spyder e execute célula a célula (`Shift+Enter` ou `Ctrl+Enter`) para inspecionar DataFrames e variáveis no console.

### 3. Executar os Testes Unitários (pytest)
Todos os testes físicos (conservação de energia, saturação de válvula, derivadas) rodam com:
```bash
pytest tests/ -v
```

---

## 🔬 Descrição do Estudo (Laboratório 02)
* **Sistema:** Trocador de calor com injeção de vapor saturado e perturbação de vazão de carga em $t = 10\text{ min}$.
* **Controlador:** Proporcional puro ($P\text{-only}$) com ganho $K_p = 2.0\,\%/\,^\circ\text{C}$ e bias $u_0 = 33.3\%$.
* **Balanço Térmico Dinâmico:** $C \frac{dT}{dt} = \dot{Q}_v - \dot{Q}_{processo} - \dot{Q}_{perdas}$.
