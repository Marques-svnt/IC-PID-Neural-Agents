---
name: skill-orchestrator
description: >-
  Autonomous skill orchestration runbook for PID Neural Control PROIC/UESC.
  Analyzes incoming user engineering tasks, determines the exact subset of required domain skills
  (process-control-systems, python-pro, ml-best-practices, academic-paper-latex, llm-application-dev-langchain-agent),
  and applies progressive disclosure to avoid context pollution.
---

# Autonomous Skill Orchestrator Runbook

Runbook de orquestração autônoma de skills para o projeto **PID Neural Control (Iniciação Científica — PROIC/UESC)**.

## Quando Usar

- Sempre que o assistente receber qualquer comando, pergunta ou plano de implementação no repositório.
- Para direcionar a consulta dinâmica de runbooks especializados antes de gerar código, modelos matemáticos ou relatórios.
- Para assegurar que as restrições da Regra de Ouro (Golden Rule) e padrões de engenharia sejam respeitados.

## Procedimento de Execução Autônoma

1. **Análise de Intenção**:
   - Classifique os domínios requeridos pela tarefa do usuário:
     - `Controle/Dinâmica de Planta`: Requer `process-control-systems`.
     - `Código Python/Testes/Refatoração`: Requer `python-pro`.
     - `Modelos Neurais/Séries Temporais`: Requer `ml-best-practices`.
     - `Relatórios/LaTex/Publicações`: Requer `academic-paper-latex`.
     - `Grafo de Agentes/LangGraph/Telemetria`: Requer `llm-application-dev-langchain-agent`.
     - `Revisão Bibliográfica/Artigos`: Requer `literature-search-arxiv` ou `literature-search-openalex`.

2. **Carga Otimizada (Progressive Disclosure)**:
   - Invoque `view_file` **apenas** no arquivo `SKILL.md` correspondente.
   - Não carregue referências extensas a menos que haja ambiguidade matemática ou algorítmica.

3. **Verificação de Restrições**:
   - Confirme que simulações numéricas e sintonias são executadas via ferramentas Python determinísticas (servidor MCP de controle).
   - Valide que todo código gerado possui type hints, docstrings Google-style e cobertura de testes.
