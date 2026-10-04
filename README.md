# PID Neural Control — PROIC/UESC

> **Iniciação Científica | UESC · CNPq/FAPESB/PROIC · Edital 34/2026**
> Controle Computacional Avançado em Projetos de Engenharia Química

## Sobre o Projeto

Este projeto compara o desempenho de **controladores PID clássicos** (Ziegler-Nichols, Cohen-Coon, IMC) com **controladores baseados em Redes Neurais** (NN-PID, LSTM Controller) aplicados a múltiplos sistemas de aquecimento típicos de plantas químicas.

Um **sistema multi-agentes (LangGraph)** orquestra automaticamente o pipeline de design, sintonia, simulação e análise comparativa.

## Estrutura do Projeto

```
pid-neural-control-proic/
├── src/
│   ├── systems/          # Modelos matemáticos dos sistemas térmicos (EDOs)
│   ├── controllers/
│   │   ├── classical/    # PID digital com sintonia Z-N, Cohen-Coon, IMC
│   │   └── neural/       # NN-PID, LSTM Controller, (opcional: RL)
│   ├── agents/           # Agentes LangGraph (Modeling, Tuning, Neural, Evaluation, Report)
│   ├── evaluation/       # Métricas: ISE, IAE, ITAE, sobressinal, tr, ts
│   ├── reporting/        # Geração de gráficos e relatórios
│   └── utils/            # Logging, config, seeds
├── notebooks/            # Exploração interativa
├── data/                 # Dados de simulação
├── results/              # Figuras e tabelas finais
├── tests/                # Testes unitários (pytest)
└── docs/                 # Documentação e relatórios PROIC
```

## Setup

```bash
# 1. Clonar o repositório
git clone <repo-url>
cd pid-neural-control-proic

# 2. Criar ambiente virtual e instalar dependências
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"

# 3. Configurar variáveis de ambiente
cp .env.example .env
# Editar .env com suas API keys

# 4. Rodar testes
pytest
```

## Cronograma de Execução — Projeto 2 (10/2026 – 05/2027)

> **Linha de Base:** O **Projeto 1 / Artigo 1** (*Controle PID Clássico de CSTRs Exotérmicos, Anti-Windup, Limites de Van Heerden e Fronteira de Pareto*) está **concluído** e documentado em `docs/reports/latex-abnt/`.

O ciclo atual (Projeto 2) é executado em **16 sprints quinzenais**:

| Fase | Período | Sprints | Descrição / Metas | Status |
|:---:|:---:|:---:|---|:---:|
| **0** | Anterior | — | **Projeto 1:** CSTR, PID clássico, anti-windup, fouling e relatório ABNT | ✅ Concluído |
| **1** | 10/2026 | S01 – S02 | Controle Neural: Geração de dados de dinâmica e arquitetura NN-PID | 🔵 Em andamento |
| **2** | 11/26 – 12/26 | S03 – S06 | NN-PID em malha fechada, controlador LSTM e salvaguardas de estabilidade | ⬜ Pendente |
| **3** | 01/27 – 02/27 | S07 – S10 | Orquestração Multi-Agente com LangGraph e ensaios autônomos massivos | ⬜ Pendente |
| **4** | 03/2027 | S11 – S12 | Benchmark Comparativo: PID Clássico vs. Neurais e nova Fronteira de Pareto | ⬜ Pendente |
| **5** | 04/2027 | S13 – S14 | **Dedicação integral à escrita:** redação do Artigo 2 / Relatório Final ABNT | ⬜ Pendente |
| **6** | 05/2027 | S15 – S16 | **Dedicação integral à banca:** elaboração dos slides, ensaios e defesa oficial | ⬜ Pendente |

> O plano executivo detalhado de cada sprint encontra-se registrado no cronograma do projeto.

## Stack Tecnológica

- **Controle Clássico:** `scipy`, `control`, `numpy`
- **Redes Neurais:** `torch`, `scikit-learn`
- **Agentes IA:** `langgraph`, `langchain`, `langchain-google-genai`
- **Visualização:** `matplotlib`, `seaborn`
- **Qualidade:** `pytest`, `black`, `ruff`, `isort`

## Autor

**[Gabriel Marques de Andrade]** — Discente IC PROIC/UESC
Orientador: [Elilton Rodrigues Edwards]
