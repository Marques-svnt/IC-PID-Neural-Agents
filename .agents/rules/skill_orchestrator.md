# Governança de Orquestração Autônoma de Skills

Este documento estabelece o protocolo mandatório para o assistente Antigravity selecionar, carregar e executar skills de forma autônoma e otimizada no contexto do projeto **PID Neural Control (PROIC/UESC)**.

---

## 1. Princípio do Progressive Disclosure

Para evitar poluição da janela de contexto e perda de foco do modelo:
1. **Nunca carregar o conteúdo de múltiplas skills de uma vez**.
2. O assistente deve analisar a solicitação do usuário, identificar o domínio de engenharia envolvido e consultar via `view_file` **apenas o `SKILL.md` estritamente necessário** antes de planejar ou executar a tarefa.
3. Se a tarefa envolver apenas código Python padrão, ativar `python-pro`. Se envolver dinâmica de processo ou sintonia, ativar `process-control-systems`.

---

## 2. Matriz de Ativação Autônoma por Domínio

| Domínio da Tarefa | Indicadores / Palavras-chave | Skill Obrigatória a Carregar | Caminho do Runbook |
|---|---|---|---|
| **Controle de Processos & EDOs** | CSTR, temperatura, balanço de energia, FOPTD, sintonia, Z-N, Cohen-Coon, IMC, Skogestad, anti-windup, IAE, ISE, ITAE, TV, $M_s$, estabilidade | `process-control-systems` | `C:\Users\biely\.gemini\config\plugins\engineering-suite\skills\process-control-systems\SKILL.md` |
| **Engenharia de Software Python** | Nova função, refatoração, pytest, ruff, black, logging, dataclass, tipagem, async, modularização | `python-pro` | `C:\Users\biely\.gemini\config\plugins\engineering-suite\skills\python-pro\SKILL.md` |
| **Machine Learning & Redes Neurais** | NN-PID, LSTM, PINN, PyTorch, treinamento, split cronológico, data leakage, validação cruzada, métricas preditivas | `ml-best-practices` | `C:\Users\biely\.gemini\config\plugins\data-agent-kit-plugin\skills\ml_best_practices\SKILL.md` |
| **Redação Científica & LaTeX** | Relatório ABNT, artigo IEEE, Elsevier, Journal of Process Control, booktabs, siunitx, tabelas, equações, BibTeX | `academic-paper-latex` | `C:\Users\biely\.gemini\config\plugins\engineering-suite\skills\academic-paper-latex\SKILL.md` |
| **Orquestração Multi-Agente & LLM** | LangGraph, StateGraph, nós, arestas condicionais, checkpoint, Langfuse, telemetria, subagentes | `llm-application-dev-langchain-agent` & `google-antigravity-sdk` | `C:\Users\biely\.gemini\config\plugins\engineering-suite\skills\llm-application-dev-langchain-agent\SKILL.md` |
| **Pesquisa Bibliográfica** | Estado da arte, papers, literatura, referências, arXiv, OpenAlex, busca acadêmica | `literature-search-arxiv` / `literature-search-openalex` | `C:\Users\biely\.gemini\config\plugins\science\skills\literature_search_arxiv\SKILL.md` |
| **Notebooks & Análise Exploratória** | Jupyter notebook, `.ipynb`, gráficos exploratórios de dados | `notebook-guidance` | `C:\Users\biely\.gemini\config\plugins\data-agent-kit-plugin\skills\notebook_guidance\SKILL.md` |

---

## 3. Diretriz de Exclusão de Skills Irrelevantes

> [!IMPORTANT]
> O assistente **NUNCA DEVE** carregar ou referenciar skills pertencentes aos seguintes domínios no escopo deste repositório:
> - **Bioinformática, Genômica e Proteômica**: AlphaFold, BLAST, Ensembl, ChEMBL, UniProt, ClinVar, dbSNP, etc.
> - **Google Cloud Corporativo / Enterprise Data**: BigQuery ML, Bigtable, Cloud Composer, Dataform, Dataproc, GCS Bucket Architect.
> - **Desenvolvimento Mobile & Web Geral**: Firebase Auth/Hosting/Firestore, Xcode.

Toda tentativa de roteamento para essas ferramentas deve ser sumariamente descartada em prol da precisão matemática e científica da planta de controle térmico.
