# Manuscritos Científicos e Relatórios Técnicos (PROIC/UESC)

Esta pasta abriga a produção científica formal do projeto de pesquisa, incluindo a **Tese / Relatório Técnico-Científico Final** de Iniciação Científica e os manuscritos para publicação internacional em formato LaTeX.

---

## 📄 Documento Principal: Tese / Relatório Técnico-Científico Final

| Metadado | Detalhe |
|---|---|
| **Arquivo** | [`Tese.pdf`](Tese.pdf) (468 KB, 39 páginas) |
| **Título** | **Contornabilidade, Saturação de Atuador e Trade-offs de Robustez no Controle PID Clássico de CSTRs Exotérmicos Altamente Não Lineares** |
| **Autor** | Gabriel Marques de Andrade |
| **Orientador** | Prof. Dr. Elilton Rodrigues Edwards |
| **Instituição** | Universidade Estadual de Santa Cruz (UESC) — Departamento de Ciências Exatas e Tecnológicas (DCET) |
| **Colegiado** | Colegiado de Engenharia Química |
| **Programa** | Iniciação Científica (PROIC/FAPESB/CNPq, Edital 34/2026) |
| **Local e Ano** | Ilhéus – Bahia, 2026 |

### Resumo da Tese
O trabalho investiga de forma exaustiva e rigorosa as limitações de contornabilidade, a saturação de atuadores e os compromissos de robustez no controle PID clássico de um Reator Contínuo de Tanque Agitado (CSTR) encamisado e altamente não linear, operando em regime de alta conversão ($X_A \approx 95\%$). Esse ponto de equilíbrio é caracterizado por instabilidade em malha aberta ($\lambda_1 = +0{,}284\text{ min}^{-1}$) e multiplicidade estacionária explicada pelas curvas de Van Heerden. Avaliam-se as metodologias de sintonia de Ziegler-Nichols (ZN), Cohen-Coon (CC), SIMC de Skogestad e otimização por critério ITAE restrito à sensibilidade máxima ($M_s \le 1{,}6$). Demonstra-se que métodos heurísticos clássicos falham catastroficamente ($M_p > 800\%$), enquanto SIMC e ITAE asseguram regulação estável. Adicionalmente, analisa-se a degradação da transferência térmica por incrustação (*fouling*), identificando o limite termodinâmico intransponível em $\alpha = 0{,}60$, além de se construir a fronteira contínua de Pareto relacionando desempenho de rastreamento (IAE) e desgaste mecânico do atuador (TV), delimitando a região de compromisso ótimo no intervalo $\tau_c \in [0{,}25; 0{,}40]\text{ min}$.

**Palavras-chave:** Reator CSTR. Controle PID Digital. Anti-windup. Limites de Van Heerden. Fronteira de Pareto.

---

### Mapeamento dos Capítulos da Tese para a Base de Código

| Capítulo da Tese | Tema Central | Módulos e Scripts Correspondentes |
|---|---|---|
| **Cap. 1** | Introdução, Motivações e Trade-offs Fundamentais | [`README.md`](../README.md), [`Projeto/README.md`](../Projeto/README.md) |
| **Cap. 2** | Modelagem Fenomenológica do CSTR e Camisa | [`src/sim_core/cstr_plant.py`](../src/sim_core/cstr_plant.py), [`src/sim_core/integrator.py`](../src/sim_core/integrator.py) |
| **Cap. 3** | Contornabilidade e Limites de Van Heerden | [`src/sim_core/actuators.py`](../src/sim_core/actuators.py), [`src/sim_core/cstr_plant.py`](../src/sim_core/cstr_plant.py) |
| **Cap. 4** | Arquitetura PID Digital com Anti-Windup Clamping | [`src/classical_control/pid_controller.py`](../src/classical_control/pid_controller.py), [`src/sim_core/actuators.py`](../src/sim_core/actuators.py) |
| **Cap. 5** | Identificação FOPTD e Sintonias (ZN, CC, SIMC, ITAE Ótimo) | [`src/classical_control/foptd.py`](../src/classical_control/foptd.py), [`src/classical_control/tuning_analytical.py`](../src/classical_control/tuning_analytical.py), [`src/classical_control/tuning_optimization.py`](../src/classical_control/tuning_optimization.py) |
| **Cap. 6** | Benchmark Nominal e Rejeição de Distúrbios | [`src/classical_control/closed_loop.py`](../src/classical_control/closed_loop.py), [`experiments/exp_01_classical_benchmark.py`](../experiments/exp_01_classical_benchmark.py) |
| **Cap. 7** | Degradação por Fouling e Fronteira de Pareto (IAE vs TV) | [`experiments/exp_02_robustness_campaign.py`](../experiments/exp_02_robustness_campaign.py), [`experiments/exp_03_pareto_tuning_sweep.py`](../experiments/exp_03_pareto_tuning_sweep.py), [`src/evaluation/pareto.py`](../src/evaluation/pareto.py) |
| **Cap. 8** | Conclusões, Contribuições Originais e Trabalhos Futuros | [`Artigo_01_PID_Neural_Comparativo/sections/07_conclusion.tex`](Artigo_01_PID_Neural_Comparativo/sections/07_conclusion.tex) |

---

## 📝 Artigos Derivados

### Artigo 1: Benchmark Abrangente em Periódico Internacional
- **Diretório:** [`Artigo_01_PID_Neural_Comparativo/`](Artigo_01_PID_Neural_Comparativo/)
- **Título:** *Controllability Boundaries, Actuator Saturation, and Robustness Trade-offs in Classical PID Control of Highly Non-linear Exothermic CSTRs*
- **Autores:** Gabriel Marques de Andrade, Elilton Rodrigues Edwards
- **Template:** IEEE Transactions Journal (8 páginas, duas colunas, estrito)
- **Status:** ✅ Finalizado / Camera-Ready
- **Pacote Overleaf:** [`Artigo_01_Overleaf.zip`](Artigo_01_PID_Neural_Comparativo/Artigo_01_Overleaf.zip) (contém os 27 arquivos necessários: tex, bib, tabelas e figuras em PDF/PNG)

---

## 🎯 Periódicos e Eventos Alvo

- **Periódicos:**
  - *Journal of Process Control* (Elsevier)
  - *Computers & Chemical Engineering* (Elsevier)
  - *ISA Transactions* (Elsevier / ISA)
  - *Brazilian Journal of Chemical Engineering* (Springer / ABEQ)
- **Congressos:**
  - **COBEQ** — Congresso Brasileiro de Engenharia Química
  - **SBAI** — Simpósio Brasileiro de Automação Inteligente
  - **CBA** — Congresso Brasileiro de Automática
