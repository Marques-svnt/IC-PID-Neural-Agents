# Projeto IC — Controle PID com Redes Neurais e Agentes Autônomos

**Instituição:** Universidade Estadual de Santa Cruz (UESC) — DCET — Colegiado de Engenharia Química  
**Programa:** PROIC — Edital 34/2026 | Apoio: CNPq / FAPESB  
**Plano de Trabalho ID:** 45-2270-1917-2025  
**Pesquisador Bolsista / Autor:** Gabriel Marques de Andrade  
**Orientador:** Prof. Dr. Elilton Rodrigues Edwards  
**Tese / Relatório Técnico:** [`../Artigos_Rascunhos/Tese.pdf`](../Artigos_Rascunhos/Tese.pdf)  

---

## Escopo Expandido

| Título Original | Título Expandido |
|---|---|
| Controle PID em Sistema de Aquecimento em Planta Química | Controle Adaptativo com Redes Neurais e Agentes Autônomos para PID em Processos Químicos |

### Contribuições Planejadas
1. Modelagem dinâmica de sistema térmico (CSTR / Trocador de Calor) — **Concluído**
2. Implementação de controladores PID clássicos com diferentes métodos de sintonia — **Concluído**
3. Desenvolvimento de controlador Neuro-PID via PyTorch (comparação com clássico) — *Em andamento*
4. Orquestração multi-agente com LangGraph (pipeline autônomo de sintonia) — *Em andamento*
5. Análise comparativa quantitativa (IAE, ITAE, TV, Mp%, ts) e Fronteira de Pareto — **Concluído**
6. Publicação de artigos científicos (Artigo 1 pronto para submissão) — **Concluído**

---

## Fases do Projeto

| Fase | Descrição | Status | Documentação de Referência |
|---|---|---|---|
| [Fase 1](Fase_1_Revisao_Bibliografica/README.md) | Revisão Bibliográfica e Modelagem do Sistema Térmico | `[x] Concluída` | Cap. 1, 2 e 3 da [Tese](../Artigos_Rascunhos/Tese.pdf) |
| [Fase 2](Fase_2_Modelagem_Matematica/README.md) | Modelagem Matemática, Dinâmica de Camisa e Discretização | `[x] Concluída` | Cap. 2 e 4 da [Tese](../Artigos_Rascunhos/Tese.pdf) |
| [Fase 3](Fase_3_PID_Classico/README.md) | Implementação, Anti-Windup Clamping e Sintonia PID Clássica | `[x] Concluída` | Cap. 4, 5, 6 e 7 da [Tese](../Artigos_Rascunhos/Tese.pdf) / [Artigo 1](../Artigos_Rascunhos/Artigo_01_PID_Neural_Comparativo/) |
| [Fase 4](Fase_4_Redes_Neurais/README.md) | Desenvolvimento do Neuro-PID (PyTorch) | `[ ] Em planejamento` | Cap. 8 da Tese (Trabalhos Futuros) |
| [Fase 5](Fase_5_Multi_Agente_LangGraph/README.md) | Orquestração Multi-Agente com LangGraph | `[ ] Em planejamento` | Cap. 8 da Tese (Trabalhos Futuros) |
| [Fase 6](Fase_6_Comparacao_Resultados/README.md) | Comparação de Resultados e Preparação do Artigo 1 | `[x] Concluída` | [Artigo 1 (Camera-Ready)](../Artigos_Rascunhos/Artigo_01_PID_Neural_Comparativo/) |

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
