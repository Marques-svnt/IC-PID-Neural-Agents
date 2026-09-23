# Fase 6 — Comparação de Resultados e Preparação do Artigo

**Status:** `[ ] Não iniciada`  
**Período PROIC:** Meses 9–12

---

## Objetivo

Consolidar todos os resultados das fases anteriores, gerar as visualizações definitivas e redigir o artigo científico completo para submissão.

## Entregáveis

### Código e Análise
- [ ] Script `benchmark.py` — comparação automatizada dos 3 métodos sob os mesmos cenários
- [ ] Cenários de teste padronizados:
  1. Mudança de setpoint (step +10°C)
  2. Perturbação de carga (ΔCAf = +20%)
  3. Variação de parâmetros do modelo (±15% em k0)
- [ ] Tabela comparativa final com todas as métricas
- [ ] Gráficos de publicação (matplotlib com estilo IEEE/ABNT)

### Artigo (Overleaf)
- [ ] Abstract
- [ ] 1. Introduction
- [ ] 2. Mathematical Modeling (CSTR)
- [ ] 3. Classical PID Tuning Methods
- [ ] 4. Neural Network-Based Adaptive PID
- [ ] 5. Multi-Agent Orchestration with LangGraph
- [ ] 6. Results and Discussion
- [ ] 7. Conclusion
- [ ] References (BibTeX)

## Estratégia de Comparação

| Estratégia | Kp | Ki | Kd | Adaptativo |
|---|---|---|---|---|
| PID Ziegler-Nichols | Fixo | Fixo | Fixo | ✗ |
| PID IMC (λ-tuning) | Fixo | Fixo | Fixo | ✗ |
| Neuro-PID (MLP) | Dinâmico | Dinâmico | Dinâmico | ✓ |
| Multi-Agente LangGraph | Dinâmico | Dinâmico | Dinâmico | ✓✓ |

## Eventos/Periódicos Alvo

- **COBEQ 2027** — prazo a confirmar
- **SBAI 2027** — prazo a confirmar
- **Brazilian Journal of Chemical Engineering** — submissão contínua
- **ISA Transactions** — JCR Q1, IF ~6

## Notas
