# Projeto IC — Controle PID com Redes Neurais e Agentes Autônomos

**Instituição:** Universidade Estadual de Santa Cruz (UESC)  
**Programa:** PROIC — Edital 34/2026 | Apoio: CNPq / FAPESB  
**Plano de Trabalho ID:** 45-2270-1917-2025  

---

## Escopo Expandido

| Título Original | Título Expandido |
|---|---|
| Controle PID em Sistema de Aquecimento em Planta Química | Controle Adaptativo com Redes Neurais e Agentes Autônomos para PID em Processos Químicos |

### Contribuições Planejadas
1. Modelagem dinâmica de sistema térmico (CSTR / Trocador de Calor)
2. Implementação de controladores PID clássicos com diferentes métodos de sintonia
3. Desenvolvimento de controlador Neuro-PID via PyTorch (comparação com clássico)
4. Orquestração multi-agente com LangGraph (pipeline autônomo de sintonia)
5. Análise comparativa quantitativa (IAE, ITAE, TV, Mp%, ts)
6. Publicação de artigo científico

---

## Fases do Projeto

| Fase | Descrição | Status |
|---|---|---|
| [Fase 1](Fase_1_Revisao_Bibliografica/README.md) | Revisão Bibliográfica e Modelagem do Sistema Térmico | `[ ] Não iniciada` |
| [Fase 2](Fase_2_Modelagem_Matematica/README.md) | Modelagem Matemática e Discretização | `[ ] Não iniciada` |
| [Fase 3](Fase_3_PID_Classico/README.md) | Implementação e Sintonia dos Controladores PID Clássicos | `[ ] Não iniciada` |
| [Fase 4](Fase_4_Redes_Neurais/README.md) | Desenvolvimento do Neuro-PID (PyTorch) | `[ ] Não iniciada` |
| [Fase 5](Fase_5_Multi_Agente_LangGraph/README.md) | Orquestração Multi-Agente com LangGraph | `[ ] Não iniciada` |
| [Fase 6](Fase_6_Comparacao_Resultados/README.md) | Comparação de Resultados e Preparação de Artigo | `[ ] Não iniciada` |

---

## Cronograma PROIC (12 Meses)

| Atividade | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 | M10 | M11 | M12 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Revisão bibliográfica + modelagem | ✓ | ✓ | | | | | | | | | | |
| Implementação do modelo em Python | | | ✓ | ✓ | | | | | | | | |
| PID clássico + Neuro-PID | | | | | ✓ | ✓ | | | | | | |
| Multi-Agente LangGraph | | | | | | | ✓ | ✓ | | | | |
| Consolidação e gráficos | | | | | | | | | ✓ | ✓ | | |
| Artigo e relatório final | | | | | | | | | | | ✓ | ✓ |

---

## Stack Técnica

- **Linguagem:** Python 3.10+
- **Simulação:** `scipy.integrate.solve_ivp`, `numpy`
- **PID Clássico:** Implementação própria com anti-windup
- **Redes Neurais:** PyTorch
- **Agentes:** LangGraph
- **Métricas:** IAE, ITAE, TV (Variation Total), Mp%, ts
- **Ambiente:** VS Code + `.venv`
