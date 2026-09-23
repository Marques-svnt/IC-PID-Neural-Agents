# Fase 3 — Implementação e Sintonia dos Controladores PID Clássicos

**Status:** `[ ] Não iniciada`  
**Período PROIC:** Meses 5–6

---

## Objetivo

Implementar o controlador PID digital em Python com features realistas e comparar pelo menos 3 métodos de sintonia. Esta fase gera o **baseline** para comparação com Neuro-PID e Multi-Agente.

## Entregáveis

- [ ] Classe `PIDController` com:
  - Anti-windup (clamping ou back-calculation)
  - Filtro derivativo (N-filter)
  - Bumpless transfer (inicialização suave)
- [ ] Módulo `tuning_methods.py`:
  - `ZieglerNichols` (resposta ao degrau / oscilação sustentada)
  - `CohenCoon`
  - `IMC` (Internal Model Control — sintonia por λ-tuning)
- [ ] Script de simulação fechada completa
- [ ] Gráficos: resposta ao degrau, rastreamento de setpoint, rejeição de perturbação
- [ ] Tabela comparativa das métricas de desempenho (IAE, ITAE, TV, Mp%, ts)
- [ ] Seção "Metodologia – PID Clássico" do artigo

## Métricas de Avaliação

| Métrica | Fórmula | Interpreta |
|---|---|---|
| IAE | ∫|e(t)|dt | Erro acumulado total |
| ITAE | ∫t·|e(t)|dt | Penaliza oscilações tardias |
| TV (Valve) | Σ|MV(k+1) - MV(k)| | Desgaste do atuador |
| Mp% | (ymax - yss)/yss × 100 | Sobressinal |
| ts | tempo p/ |e| < 2% | Tempo de acomodação |

## Notas
