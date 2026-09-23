# Fase 4 — Desenvolvimento do Neuro-PID (PyTorch)

**Status:** `[ ] Não iniciada`  
**Período PROIC:** Meses 7–8 (paralelo ao início da Fase 5)

---

## Objetivo

Desenvolver um controlador adaptativo baseado em Rede Neural que ajusta dinamicamente os ganhos do PID (Kp, Ki, Kd) a partir do estado atual do processo. Comparar seu desempenho com o PID clássico da Fase 3.

## Arquitetura Proposta

```
Entradas:  [e(t), Δe(t), T(t), t]  →  Rede Neural (MLP/LSTM)  →  [ΔKp, ΔKi, ΔKd]
                                                                          ↓
                                                               PID com ganhos adaptados
                                                                          ↓
                                                                    Planta (CSTR)
```

## Entregáveis

- [ ] Módulo `neural_pid.py` — rede MLP em PyTorch para predição de ganhos
- [ ] Pipeline de treinamento:
  - Geração de dados de trajetória via simulação do CSTR
  - Loss function: IAE ou ITAE como reward proxy
- [ ] Validação em cenários: mudança de setpoint, perturbação de carga, variação de parâmetros
- [ ] Comparação direta Neuro-PID vs PID clássico (mesmas condições de simulação)
- [ ] Seção "Neuro-PID" do artigo

## Alternativas de Arquitetura Neural

| Arquitetura | Vantagem | Complexidade |
|---|---|---|
| MLP (Multilayer Perceptron) | Simples, interpretável | Baixa |
| LSTM / GRU | Captura dinâmica temporal | Média |
| PINN | Incorpora leis físicas na loss | Alta |
| Reinforcement Learning (SAC/PPO) | Otimização direta de política | Alta |

**Recomendação para IC:** Iniciar com MLP. Estender para LSTM se o tempo permitir.

## Notas
