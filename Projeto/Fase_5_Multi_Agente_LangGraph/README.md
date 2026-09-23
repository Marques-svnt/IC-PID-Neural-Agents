# Fase 5 — Orquestração Multi-Agente com LangGraph

**Status:** `[ ] Não iniciada`  
**Período PROIC:** Meses 7–8

---

## Objetivo

Orquestrar os componentes do projeto (sensor, modelo, sintonia, segurança, métricas) como **agentes autônomos** usando LangGraph, criando um pipeline de controle inteligente end-to-end.

## Arquitetura do Grafo de Agentes

```
┌─────────────────────────────────────────────────────┐
│                   LangGraph State                    │
│  {T, CA, e, Kp, Ki, Kd, metrics, safety_flag}      │
└───────────────────────────┬─────────────────────────┘
                            │
          ┌─────────────────▼─────────────────┐
          │        Agente Telemetria           │
          │  - Lê T(t) da simulação            │
          │  - Adiciona ruído gaussiano         │
          └─────────────────┬─────────────────┘
                            │
          ┌─────────────────▼─────────────────┐
          │      Agente Modelo (Surrogate)     │
          │  - Prediz T(t+k) com MLP/PINN      │
          │  - Estima sensibilidade do processo │
          └─────────────────┬─────────────────┘
                            │
          ┌─────────────────▼─────────────────┐
          │       Agente Neuro-PID             │
          │  - Propõe [ΔKp, ΔKi, ΔKd]         │
          │  - Usa modelo da Fase 4             │
          └─────────────────┬─────────────────┘
                            │
          ┌─────────────────▼─────────────────┐
          │     Agente Guardião (Safety)       │
          │  - Valida limites dos ganhos       │
          │  - Veto se T > Tcritico             │
          │  - Fallback para PID nominal        │
          └─────────────────┬─────────────────┘
                            │
          ┌─────────────────▼─────────────────┐
          │      Agente de Métricas            │
          │  - Calcula IAE, ITAE, TV           │
          │  - Registra histórico               │
          └─────────────────────────────────────┘
```

## Entregáveis

- [ ] Módulo `agents/` com um arquivo por agente:
  - `agent_telemetry.py`
  - `agent_surrogate.py`
  - `agent_neuro_pid.py`
  - `agent_safety.py`
  - `agent_metrics.py`
- [ ] `graph.py` — definição do LangGraph (nós, arestas, estado compartilhado)
- [ ] `run_simulation.py` — loop de simulação completo com o grafo
- [ ] Visualização do grafo de agentes (Mermaid ou Matplotlib)
- [ ] Seção "Metodologia – Sistema Multi-Agente" do artigo

## Dependências Python

```
langgraph>=0.2
langchain-core>=0.3
torch>=2.0
scipy>=1.11
numpy>=1.24
```

## Notas
