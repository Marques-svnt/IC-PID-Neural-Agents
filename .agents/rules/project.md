# Contexto do Projeto — IC PROIC/UESC

## Descrição Geral
Projeto de Iniciação Científica (PROIC/UESC, Edital 34/2026) sobre controle de processos.
Título: "Controle PID em Sistema de Aquecimento em Planta Química" (expandido).

## Expansão do Escopo
1. **Controle clássico:** P, PI, PID digital com sintonia Z-N, Cohen-Coon e IMC
2. **Controle moderno:** Redes Neurais (NN-PID, LSTM Controller, opcionalmente RL)
3. **Automação via MAS:** Sistema multi-agentes com LangGraph orquestrando o pipeline
4. **Múltiplos sistemas:** Diferentes configurações de sistemas térmicos (a definir com orientador)

## Fase Atual
> Atualizar conforme avança: Fase 1 — Fundamentação (Revisão Bibliográfica + Setup)

## Domínio Técnico
- **Modelagem:** EDOs, balanço de energia, identificação FOPDT/SOPDT
- **Controle clássico:** discretização Tustin/Euler, anti-windup, estruturas paralela e ISA
- **Controle neural:** PyTorch para NN-PID e LSTM; stable-baselines3 para RL (opcional)
- **Multi-Agentes:** LangGraph com tools Python bem definidas, LangSmith para observabilidade
- **Métricas de desempenho:** ISE, IAE, ITAE, sobressinal (%OS), tempo de subida (tr), tempo de acomodação (ts)

## Estrutura do Repositório
```
pid-neural-control-proic/
├── src/
│   ├── systems/          # Modelos matemáticos dos sistemas térmicos
│   ├── controllers/
│   │   ├── classical/    # PID, PI, P com métodos de sintonia
│   │   └── neural/       # NN-PID, LSTM Controller, RL
│   ├── agents/           # LangGraph: Modeling, Tuning, Neural, Evaluation, Report agents
│   ├── evaluation/       # Métricas ISE, IAE, ITAE, gráficos comparativos
│   ├── reporting/        # Geração de relatórios e figuras
│   └── utils/            # Helpers gerais (logging, config, etc.)
├── notebooks/            # Exploração e visualização interativa
├── data/                 # Dados de simulação e resultados
├── results/              # Figuras e tabelas finais para o relatório
├── tests/                # Testes unitários pytest
└── docs/                 # Documentação, bibliografia e relatórios
```

## Regras Específicas para Este Projeto

### Modelagem
- Sempre definir unidades explicitamente nas variáveis e docstrings (K, kg, J, W, m³...)
- Modelos de sistemas ficam em `src/systems/` — cada sistema é uma classe independente
- Todo modelo deve ter método `step_response()` para validação visual imediata

### Controle
- Controladores clássicos: implementar em tempo discreto (não contínuo) por padrão
- Sempre implementar anti-windup em controladores PID
- Parâmetros de sintonia devem ser dataclasses tipadas, nunca dicts soltos

### Redes Neurais
- Usar PyTorch; separar arquitetura, treinamento e inferência em módulos distintos
- Salvar checkpoints de modelos em `data/processed/checkpoints/`
- Sempre registrar métricas de treinamento com logging estruturado

### LangGraph / Agentes
- Cada agente tem responsabilidade única e bem definida (SRP)
- Ferramentas (tools) devem ter type hints e docstrings completas
- Estado do grafo deve ser um TypedDict documentado
- Incluir checkpoints de estado do grafo para retomada em caso de falha

### Relatórios e Gráficos
- Figuras para relatório: DPI mínimo 300, fonte mínima 12pt, labels em português
- Paleta de cores acessível (usar matplotlib style científico ou seaborn-paper)
- Tabelas comparativas sempre incluem unidades nas colunas

### Qualidade
- Toda simulação deve ser reproduzível: fixar seeds (`np.random.seed`, `torch.manual_seed`)
- Usar `pyproject.toml` para gerenciamento de dependências (uv ou pip)
- Testes unitários obrigatórios para: modelos de sistema, controladores e métricas de avaliação
