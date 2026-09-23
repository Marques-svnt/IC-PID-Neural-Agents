# Fase 2 — Modelagem Matemática e Discretização

**Status:** `[ ] Não iniciada`  
**Período PROIC:** Meses 3–4

---

## Objetivo

Implementar em Python o modelo matemático do sistema térmico escolhido e validar seu comportamento dinâmico antes de aplicar qualquer controlador.

## Entregáveis

- [ ] Script `plant_model.py` com o modelo em EDO (balanço de massa e energia)
- [ ] Integração numérica via `scipy.integrate.solve_ivp` (ou RK4 próprio)
- [ ] Análise de pontos de operação (steady-state) e linearização em torno do ponto nominal
- [ ] Gráficos de resposta ao degrau em malha aberta
- [ ] Modelo discretizado (para uso no PID digital)
- [ ] Seção "Modelagem" do artigo em rascunho

## Modelo CSTR Não-Isotérmico (referência)

**EDO — Balanço de Concentração:**
```
dCA/dt = (q/V)(CAf - CA) - k0·exp(-E/RT)·CA
```

**EDO — Balanço de Energia:**
```
dT/dt = (q/V)(Tf - T) + (-ΔH)/(ρCp) · k0·exp(-E/RT)·CA + UA/(VρCp)·(Tc - T)
```

**Variável controlada (PV):** Temperatura do reator T(t)  
**Variável manipulada (MV):** Temperatura da camisa Tc(t)  
**Perturbações (DV):** CAf, Tf

## Parâmetros de Referência (Seborg et al.)

| Parâmetro | Valor | Unidade |
|---|---|---|
| q | 100 | L/min |
| V | 100 | L |
| k0 | 7.2×10¹⁰ | min⁻¹ |
| E/R | 8750 | K |
| ΔH | -5×10⁴ | cal/mol |
| ρCp | 500 | cal/(L·K) |
| UA | 5×10⁴ | cal/(min·K) |

## Notas
