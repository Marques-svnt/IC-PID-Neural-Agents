"""Autonomous Skill Orchestration module for PID Neural Control PROIC/UESC.

This module analyzes incoming user and agent engineering tasks, classifies
their technical domains, resolves the optimal minimal subset of Antigravity skills,
and provides progressive-disclosure runbook content while blocking irrelevant
domain skills (e.g. bioinformatics, cloud warehousing, mobile development).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SkillTier(str, Enum):
    """Categorization tier indicating skill relevance and loading priority."""

    CORE_TIER_1 = "core_tier_1"
    SUPPORT_TIER_2 = "support_tier_2"
    BLOCKED_TIER_3 = "blocked_tier_3"


class SkillDomain(str, Enum):
    """Technical domain taxonomy relevant to the thermal control project."""

    PROCESS_CONTROL = "process_control"
    SOFTWARE_ENGINEERING = "software_engineering"
    MACHINE_LEARNING = "machine_learning"
    ACADEMIC_REPORTING = "academic_reporting"
    AGENT_ORCHESTRATION = "agent_orchestration"
    LITERATURE_SEARCH = "literature_search"
    OPTIMIZATION = "optimization"
    NOTEBOOKS = "notebooks"
    UI_VISUALIZATION = "ui_visualization"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class SkillDescriptor:
    """Metadata specification for an Antigravity skill.

    Attributes:
        name: Unique lowercase identifier of the skill.
        tier: Relevance priority tier.
        domain: Functional engineering domain.
        description: Summary of skill scope and intent.
        path: Expected filesystem path to SKILL.md.
        keywords: Set of identifying terminology and regex tokens.
        dependencies: Sibling skills often co-activated.
    """

    name: str
    tier: SkillTier
    domain: SkillDomain
    description: str
    path: str
    keywords: Set[str] = field(default_factory=set)
    dependencies: List[str] = field(default_factory=list)


# Registry of known Antigravity skills mapped to project tiers
PROJECT_SKILLS: Dict[str, SkillDescriptor] = {
    # --- Tier 1: Core Project Skills ---
    "process-control-systems": SkillDescriptor(
        name="process-control-systems",
        tier=SkillTier.CORE_TIER_1,
        domain=SkillDomain.PROCESS_CONTROL,
        description=(
            "Non-linear ODE plant modeling (CSTR/CSTH), Arrhenius kinetics, "
            "FOPTD identification, classical PID tuning (SIMC, Z-N, Cohen-Coon), "
            "anti-windup, and multi-objective Pareto metrics (IAE, ISE, ITAE, TV, Ms)."
        ),
        path=r"C:\Users\biely\.gemini\config\plugins\engineering-suite\skills\process-control-systems\SKILL.md",
        keywords={
            "cstr",
            "reactor",
            "ode",
            "foptd",
            "soptd",
            "pid",
            "tuning",
            "sintonia",
            "arrhenius",
            "thermal",
            "temperatura",
            "simc",
            "skogestad",
            "cohen-coon",
            "ziegler-nichols",
            "anti-windup",
            "windup",
            "iae",
            "itae",
            "ise",
            "tv",
            "total variation",
            "overshoot",
            "sobressinal",
            "ms",
            "sensitivity",
        },
        dependencies=["python-pro"],
    ),
    "python-pro": SkillDescriptor(
        name="python-pro",
        tier=SkillTier.CORE_TIER_1,
        domain=SkillDomain.SOFTWARE_ENGINEERING,
        description=(
            "Idiomatic Python 3.12+ engineering, strict type hints, Google-style docstrings, "
            "Ruff linting, pytest testing, modularization, and memory profiling."
        ),
        path=r"C:\Users\biely\.gemini\config\plugins\engineering-suite\skills\python-pro\SKILL.md",
        keywords={
            "python",
            "refactor",
            "pytest",
            "testes",
            "unit test",
            "typing",
            "type hints",
            "docstrings",
            "ruff",
            "black",
            "logging",
            "pep8",
            "async",
        },
        dependencies=[],
    ),
    "ml-best-practices": SkillDescriptor(
        name="ml-best-practices",
        tier=SkillTier.CORE_TIER_1,
        domain=SkillDomain.MACHINE_LEARNING,
        description=(
            "Machine learning workflows, neural network policies (NN-PID, LSTM, PINN), "
            "chronological splits, avoiding data leakage, cross-validation, and model comparison."
        ),
        path=r"C:\Users\biely\.gemini\config\plugins\data-agent-kit-plugin\skills\ml_best_practices\SKILL.md",
        keywords={
            "ml",
            "machine learning",
            "neural",
            "neural network",
            "rede neural",
            "nn-pid",
            "lstm",
            "pinn",
            "pytorch",
            "epoch",
            "loss",
            "training",
            "treinamento",
            "time series",
            "series temporais",
            "data leakage",
        },
        dependencies=["python-pro"],
    ),
    "academic-paper-latex": SkillDescriptor(
        name="academic-paper-latex",
        tier=SkillTier.CORE_TIER_1,
        domain=SkillDomain.ACADEMIC_REPORTING,
        description=(
            "Scientific LaTeX typesetting for IEEE Transactions and Elsevier Journal of "
            "Process Control, publication-grade tables (booktabs, siunitx), vector figures."
        ),
        path=r"C:\Users\biely\.gemini\config\plugins\engineering-suite\skills\academic-paper-latex\SKILL.md",
        keywords={
            "latex",
            "abnt",
            "ieee",
            "elsevier",
            "journal",
            "relatorio",
            "report",
            "artigo",
            "paper",
            "booktabs",
            "siunitx",
            "bibtex",
            "tabela",
            "figura",
        },
        dependencies=[],
    ),
    "llm-application-dev-langchain-agent": SkillDescriptor(
        name="llm-application-dev-langchain-agent",
        tier=SkillTier.CORE_TIER_1,
        domain=SkillDomain.AGENT_ORCHESTRATION,
        description=(
            "Production LangGraph StateGraph architectures, multi-agent workflows, "
            "conditional branching, persistent checkpoints, and Langfuse observability."
        ),
        path=r"C:\Users\biely\.gemini\config\plugins\engineering-suite\skills\llm-application-dev-langchain-agent\SKILL.md",
        keywords={
            "langgraph",
            "langchain",
            "stategraph",
            "multi-agent",
            "mas",
            "agente",
            "checkpoint",
            "langfuse",
            "telemetria",
            "node",
            "edges",
        },
        dependencies=["python-pro"],
    ),
    "literature-search-arxiv": SkillDescriptor(
        name="literature-search-arxiv",
        tier=SkillTier.CORE_TIER_1,
        domain=SkillDomain.LITERATURE_SEARCH,
        description="Literature search across arXiv for control theory, neural PID, and thermal benchmarks.",
        path=r"C:\Users\biely\.gemini\config\plugins\science\skills\literature_search_arxiv\SKILL.md",
        keywords={
            "arxiv",
            "paper search",
            "pesquisa bibliografica",
            "literature",
            "artigos",
        },
        dependencies=[],
    ),
    "literature-search-openalex": SkillDescriptor(
        name="literature-search-openalex",
        tier=SkillTier.CORE_TIER_1,
        domain=SkillDomain.LITERATURE_SEARCH,
        description="Literature search across OpenAlex database for citation and academic journal queries.",
        path=r"C:\Users\biely\.gemini\config\plugins\science\skills\literature_search_openalex\SKILL.md",
        keywords={"openalex", "doi", "citations", "autores", "bibliometria"},
        dependencies=[],
    ),
    "google-antigravity-sdk": SkillDescriptor(
        name="google-antigravity-sdk",
        tier=SkillTier.CORE_TIER_1,
        domain=SkillDomain.AGENT_ORCHESTRATION,
        description="Google Antigravity SDK agent configuration, subagents, tools, and hooks.",
        path=r"C:\Users\biely\.gemini\config\plugins\google-antigravity-sdk\skills\google-antigravity-sdk\SKILL.md",
        keywords={
            "antigravity sdk",
            "subagent",
            "subagente",
            "localagentconfig",
            "agentbehavior",
        },
        dependencies=["python-pro"],
    ),
    # --- Tier 2: Support Skills ---
    "notebook-guidance": SkillDescriptor(
        name="notebook-guidance",
        tier=SkillTier.SUPPORT_TIER_2,
        domain=SkillDomain.NOTEBOOKS,
        description="Guidelines for structuring exploratory Jupyter Notebooks (.ipynb) cleanly.",
        path=r"C:\Users\biely\.gemini\config\plugins\data-agent-kit-plugin\skills\notebook_guidance\SKILL.md",
        keywords={"notebook", "jupyter", "ipynb"},
        dependencies=[],
    ),
    "operations-research-supply-chain": SkillDescriptor(
        name="operations-research-supply-chain",
        tier=SkillTier.SUPPORT_TIER_2,
        domain=SkillDomain.OPTIMIZATION,
        description="Mathematical optimization (MILP, LP, Pyomo, PuLP, OR-Tools).",
        path=r"C:\Users\biely\.gemini\config\plugins\engineering-suite\skills\operations-research-supply-chain\SKILL.md",
        keywords={"milp", "lp", "pyomo", "pulp", "or-tools", "otimizacao matematica"},
        dependencies=["python-pro"],
    ),
    "generative_ui": SkillDescriptor(
        name="generative_ui",
        tier=SkillTier.SUPPORT_TIER_2,
        domain=SkillDomain.UI_VISUALIZATION,
        description="Interactive inline HTML widgets, diagrams, and dashboards.",
        path=r"C:\Users\biely\.gemini\antigravity\builtin\skills\generative_ui\SKILL.md",
        keywords={"widget", "interativo", "dashboard", "generative ui"},
        dependencies=[],
    ),
}

# Prefixes and markers of strictly blocked skills (Tier 3)
BLOCKED_SKILL_PATTERNS = [
    r"^alpha.*",
    r".*genome.*",
    r".*database.*",
    r"^chembl.*",
    r"^clin.*",
    r"^dbsnp.*",
    r"^embl.*",
    r"^encode.*",
    r"^ensembl.*",
    r"^firebase.*",
    r"^firestore.*",
    r"^gcp.*",
    r"^bigquery.*",
    r"^bigtable.*",
    r"^dataform.*",
    r"^dbt.*",
    r"^xcode.*",
    r"^protein.*",
    r"^uniprot.*",
    r"^pymol.*",
    r"^foldseek.*",
]


class AutonomousSkillOrchestrator:
    """Intelligent Skill Orchestrator for the Antigravity agent system.

    Provides intent classification, progressive disclosure skill routing,
    safety-boundary verification, and plan synthesis tailored to the
    PID Neural Control project.
    """

    def __init__(self, catalog: Optional[Dict[str, SkillDescriptor]] = None) -> None:
        """Initialize orchestrator with skill catalog.

        Args:
            catalog: Optional custom dictionary of SkillDescriptor entries.
                     Defaults to PROJECT_SKILLS.
        """
        self._catalog = catalog if catalog is not None else dict(PROJECT_SKILLS)
        logger.info(
            "AutonomousSkillOrchestrator initialized with %d recognized skills.",
            len(self._catalog),
        )

    def is_skill_allowed(self, skill_name: str) -> bool:
        """Check whether a skill is allowed in the project's technical scope.

        Blocks external domains (bioinformatics, corporate GCP, mobile)
        to prevent hallucination and token waste.

        Args:
            skill_name: Name of the skill to inspect.

        Returns:
            True if allowed (Tier 1 or Tier 2), False if blocked (Tier 3).
        """
        normalized_name = skill_name.strip().lower()

        # Check explicit catalog entry
        if normalized_name in self._catalog:
            tier = self._catalog[normalized_name].tier
            return tier in (SkillTier.CORE_TIER_1, SkillTier.SUPPORT_TIER_2)

        # Check blocked patterns
        for pattern in BLOCKED_SKILL_PATTERNS:
            if re.match(pattern, normalized_name):
                logger.warning(
                    "Skill '%s' matches blocked pattern '%s'. Denying access.",
                    normalized_name,
                    pattern,
                )
                return False

        # If not explicitly mapped or blocked, reject by default (safe-by-default)
        return False

    def classify_intent(self, task_description: str) -> List[SkillDomain]:
        """Classify task description into one or more project skill domains.

        Args:
            task_description: Natural language prompt or task statement.

        Returns:
            List of detected SkillDomain enums ordered by relevance score.
        """
        if not task_description or not task_description.strip():
            logger.debug("Empty task description received; returning UNSUPPORTED.")
            return [SkillDomain.UNSUPPORTED]

        normalized_text = task_description.lower()
        domain_scores: Dict[SkillDomain, int] = {domain: 0 for domain in SkillDomain}

        # Score matching keywords from registered skills
        for descriptor in self._catalog.values():
            if descriptor.tier == SkillTier.BLOCKED_TIER_3:
                continue
            for kw in descriptor.keywords:
                # Word boundary search
                if re.search(rf"\b{re.escape(kw)}\b", normalized_text):
                    domain_scores[descriptor.domain] += 2
                elif kw in normalized_text:
                    domain_scores[descriptor.domain] += 1

        # Heuristic boosts for common project scenarios
        if any(
            term in normalized_text
            for term in ["cstr", "foptd", "simc", "sintonia", "malha"]
        ):
            domain_scores[SkillDomain.PROCESS_CONTROL] += 5
        if any(
            term in normalized_text
            for term in ["pytest", "refatorar", "type hint", "ruff", "black"]
        ):
            domain_scores[SkillDomain.SOFTWARE_ENGINEERING] += 5
        if any(
            term in normalized_text
            for term in ["nn-pid", "lstm", "pinn", "treinar", "pytorch"]
        ):
            domain_scores[SkillDomain.MACHINE_LEARNING] += 5
        if any(
            term in normalized_text
            for term in ["relatorio", "abnt", "ieee", "latex", "tabela"]
        ):
            domain_scores[SkillDomain.ACADEMIC_REPORTING] += 5
        if any(
            term in normalized_text
            for term in ["langgraph", "orquestrador", "stategraph"]
        ):
            domain_scores[SkillDomain.AGENT_ORCHESTRATION] += 5
        if any(
            term in normalized_text
            for term in ["paper", "artigo", "arxiv", "busca biblio"]
        ):
            domain_scores[SkillDomain.LITERATURE_SEARCH] += 5

        # Filter domains with positive scores and sort descending
        matched = [
            domain
            for domain, score in sorted(
                domain_scores.items(), key=lambda item: item[1], reverse=True
            )
            if score > 0 and domain != SkillDomain.UNSUPPORTED
        ]

        if not matched:
            logger.info(
                "No matching domains identified for prompt: '%s'", task_description[:60]
            )
            return [SkillDomain.SOFTWARE_ENGINEERING]  # Default to software quality

        logger.debug("Classified task into domains: %s", matched)
        return matched

    def resolve_skills(
        self,
        task_description: str,
        max_skills: int = 3,
    ) -> List[SkillDescriptor]:
        """Resolve the optimal minimal set of skills for a task (Progressive Disclosure).

        Args:
            task_description: Prompt or task statement to evaluate.
            max_skills: Upper ceiling of simultaneous skills to avoid context bloat.

        Returns:
            List of SkillDescriptor instances sorted by priority.
        """
        domains = self.classify_intent(task_description)
        selected_skills: List[SkillDescriptor] = []
        selected_names: Set[str] = set()

        for domain in domains:
            domain_skills = [
                s
                for s in self._catalog.values()
                if s.domain == domain and s.tier == SkillTier.CORE_TIER_1
            ]
            for skill in domain_skills:
                if (
                    skill.name not in selected_names
                    and len(selected_skills) < max_skills
                ):
                    selected_skills.append(skill)
                    selected_names.add(skill.name)

        # If capacity remains, check Tier 2 support skills
        if len(selected_skills) < max_skills:
            for domain in domains:
                support_skills = [
                    s
                    for s in self._catalog.values()
                    if s.domain == domain and s.tier == SkillTier.SUPPORT_TIER_2
                ]
                for skill in support_skills:
                    if (
                        skill.name not in selected_names
                        and len(selected_skills) < max_skills
                    ):
                        selected_skills.append(skill)
                        selected_names.add(skill.name)

        logger.info(
            "Resolved %d skills for task: %s",
            len(selected_skills),
            [s.name for s in selected_skills],
        )
        return selected_skills

    def load_skill_instructions(self, skill_name: str) -> Optional[str]:
        """Safely read instructions from SKILL.md for a given skill.

        Args:
            skill_name: Identifier of the requested skill.

        Returns:
            Content string of SKILL.md if found, or None if file does not exist
            or skill is blocked.
        """
        if not self.is_skill_allowed(skill_name):
            logger.warning(
                "Attempted to load blocked or unregistered skill: '%s'", skill_name
            )
            return None

        descriptor = self._catalog.get(skill_name)
        if not descriptor:
            return None

        skill_file = Path(descriptor.path)
        if not skill_file.is_file():
            logger.warning("SKILL.md not found at expected path: %s", skill_file)
            return None

        try:
            content = skill_file.read_text(encoding="utf-8")
            logger.info(
                "Loaded SKILL.md for skill '%s' (%d bytes).", skill_name, len(content)
            )
            return content
        except OSError as exc:
            logger.error("Failed to read SKILL.md for '%s': %s", skill_name, exc)
            return None

    def plan_execution(self, task_description: str) -> Dict[str, Any]:
        """Synthesize an autonomous execution plan with resolved skills and tools.

        Args:
            task_description: Technical task description.

        Returns:
            Dictionary containing classified domains, loaded skills, recommended
            deterministic MCP tools, and engineering verification checkpoints.
        """
        domains = self.classify_intent(task_description)
        skills = self.resolve_skills(task_description)

        # Identify recommended deterministic MCP tools based on domain
        mcp_tools: List[str] = []
        if SkillDomain.PROCESS_CONTROL in domains:
            mcp_tools.extend(
                [
                    "simulate_thermal_plant",
                    "compute_transient_metrics",
                    "tune_classical_pid",
                ]
            )
        if SkillDomain.MACHINE_LEARNING in domains:
            mcp_tools.append("simulate_thermal_plant")

        plan: Dict[str, Any] = {
            "task": task_description,
            "classified_domains": [d.value for d in domains],
            "selected_skills": [
                {
                    "name": s.name,
                    "tier": s.tier.value,
                    "domain": s.domain.value,
                    "path": s.path,
                }
                for s in skills
            ],
            "recommended_mcp_tools": mcp_tools,
            "golden_rule_enforced": True,
            "quality_checks": [
                "Strict typing (PEP 484) and Google-style docstrings",
                "Deterministic ODE integration via scipy (no LLM approximation)",
                "Full pytest unit test coverage",
                "Ruff code quality compliance",
            ],
        }

        logger.info(
            "Plan generated successfully for task with %d domains.", len(domains)
        )
        return plan
