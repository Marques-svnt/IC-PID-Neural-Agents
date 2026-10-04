"""Unit tests for the AutonomousSkillOrchestrator module."""

from __future__ import annotations

import pytest

from src.agents.skill_orchestrator import (
    AutonomousSkillOrchestrator,
    SkillDescriptor,
    SkillDomain,
    SkillTier,
)


class TestSkillOrchestratorFiltering:
    """Test skill validation and safety boundary filtering."""

    @pytest.fixture
    def orchestrator(self) -> AutonomousSkillOrchestrator:
        return AutonomousSkillOrchestrator()

    def test_core_skills_are_allowed(
        self, orchestrator: AutonomousSkillOrchestrator
    ) -> None:
        assert orchestrator.is_skill_allowed("process-control-systems") is True
        assert orchestrator.is_skill_allowed("python-pro") is True
        assert orchestrator.is_skill_allowed("ml-best-practices") is True
        assert orchestrator.is_skill_allowed("academic-paper-latex") is True
        assert (
            orchestrator.is_skill_allowed("llm-application-dev-langchain-agent") is True
        )

    def test_support_skills_are_allowed(
        self, orchestrator: AutonomousSkillOrchestrator
    ) -> None:
        assert orchestrator.is_skill_allowed("notebook-guidance") is True
        assert orchestrator.is_skill_allowed("operations-research-supply-chain") is True
        assert orchestrator.is_skill_allowed("generative_ui") is True

    def test_blocked_bioinformatics_skills_are_rejected(
        self, orchestrator: AutonomousSkillOrchestrator
    ) -> None:
        assert (
            orchestrator.is_skill_allowed("alphafold-database-fetch-and-analyze")
            is False
        )
        assert (
            orchestrator.is_skill_allowed("alphagenome-single-variant-analysis")
            is False
        )
        assert orchestrator.is_skill_allowed("chembl-database") is False
        assert orchestrator.is_skill_allowed("clinvar-database") is False
        assert orchestrator.is_skill_allowed("pymol") is False

    def test_blocked_cloud_and_mobile_skills_are_rejected(
        self, orchestrator: AutonomousSkillOrchestrator
    ) -> None:
        assert orchestrator.is_skill_allowed("bigquery-sql") is False
        assert orchestrator.is_skill_allowed("bigtable-basics") is False
        assert orchestrator.is_skill_allowed("gcp-dataflow") is False
        assert orchestrator.is_skill_allowed("firebase-auth-basics") is False
        assert orchestrator.is_skill_allowed("xcode-project-setup") is False

    def test_unknown_skills_are_rejected_by_default(
        self, orchestrator: AutonomousSkillOrchestrator
    ) -> None:
        assert orchestrator.is_skill_allowed("arbitrary-unknown-skill") is False


class TestSkillOrchestratorIntentClassification:
    """Test domain intent classification for diverse engineering tasks."""

    @pytest.fixture
    def orchestrator(self) -> AutonomousSkillOrchestrator:
        return AutonomousSkillOrchestrator()

    def test_classify_empty_prompt_returns_unsupported(
        self, orchestrator: AutonomousSkillOrchestrator
    ) -> None:
        assert orchestrator.classify_intent("") == [SkillDomain.UNSUPPORTED]
        assert orchestrator.classify_intent("   ") == [SkillDomain.UNSUPPORTED]

    def test_classify_process_control_task(
        self, orchestrator: AutonomousSkillOrchestrator
    ) -> None:
        prompt = "Sintonizar os ganhos do PID para o CSTR usando o metodo SIMC Skogestad e checar IAE"
        domains = orchestrator.classify_intent(prompt)
        assert domains[0] == SkillDomain.PROCESS_CONTROL

    def test_classify_software_engineering_task(
        self, orchestrator: AutonomousSkillOrchestrator
    ) -> None:
        prompt = "Refatorar as funcoes adicionando type hints rigorosos e testes unitarios com pytest"
        domains = orchestrator.classify_intent(prompt)
        assert domains[0] == SkillDomain.SOFTWARE_ENGINEERING

    def test_classify_machine_learning_task(
        self, orchestrator: AutonomousSkillOrchestrator
    ) -> None:
        prompt = "Treinar modelo NN-PID com PyTorch usando divisao cronologica para evitar data leakage"
        domains = orchestrator.classify_intent(prompt)
        assert domains[0] == SkillDomain.MACHINE_LEARNING

    def test_classify_academic_reporting_task(
        self, orchestrator: AutonomousSkillOrchestrator
    ) -> None:
        prompt = "Gerar tabela em LaTeX com booktabs e siunitx no padrao ABNT para o relatorio de IC"
        domains = orchestrator.classify_intent(prompt)
        assert domains[0] == SkillDomain.ACADEMIC_REPORTING

    def test_classify_literature_search_task(
        self, orchestrator: AutonomousSkillOrchestrator
    ) -> None:
        prompt = "Buscar papers no arXiv sobre controle neural adaptativo em reatores quimicos"
        domains = orchestrator.classify_intent(prompt)
        assert domains[0] == SkillDomain.LITERATURE_SEARCH

    def test_classify_fallback_for_generic_prompt(
        self, orchestrator: AutonomousSkillOrchestrator
    ) -> None:
        prompt = "Verifique o status do sistema"
        domains = orchestrator.classify_intent(prompt)
        assert SkillDomain.SOFTWARE_ENGINEERING in domains


class TestSkillResolutionAndLoading:
    """Test minimal skill resolution and instructions loading."""

    @pytest.fixture
    def orchestrator(self) -> AutonomousSkillOrchestrator:
        return AutonomousSkillOrchestrator()

    def test_resolve_skills_respects_max_limit(
        self, orchestrator: AutonomousSkillOrchestrator
    ) -> None:
        prompt = "Simular CSTR, refatorar codigo Python, treinar NN-PID e gerar relatorio LaTeX ABNT"
        skills = orchestrator.resolve_skills(prompt, max_skills=2)
        assert len(skills) == 2
        skill_names = [s.name for s in skills]
        # Should include core control and another high-priority skill
        assert "process-control-systems" in skill_names

    def test_resolve_skills_deduplicates_skills(
        self, orchestrator: AutonomousSkillOrchestrator
    ) -> None:
        prompt = "CSTR EDO arrhenius foptd sintonia simc iae itae"
        skills = orchestrator.resolve_skills(prompt, max_skills=3)
        names = [s.name for s in skills]
        assert len(names) == len(set(names))

    def test_load_skill_instructions_blocked_skill_returns_none(
        self, orchestrator: AutonomousSkillOrchestrator
    ) -> None:
        assert orchestrator.load_skill_instructions("alphafold-database") is None

    def test_load_skill_instructions_existing_skill_returns_content(
        self, orchestrator: AutonomousSkillOrchestrator
    ) -> None:
        content = orchestrator.load_skill_instructions("process-control-systems")
        assert content is not None
        assert "process-control-systems" in content.lower()
        assert "cstr" in content.lower()


class TestPlanExecution:
    """Test plan execution synthesis."""

    @pytest.fixture
    def orchestrator(self) -> AutonomousSkillOrchestrator:
        return AutonomousSkillOrchestrator()

    def test_plan_execution_for_control_task(
        self, orchestrator: AutonomousSkillOrchestrator
    ) -> None:
        task = "Avaliar sintonia IMC para o CSTR e comparar IAE versus Total Variation"
        plan = orchestrator.plan_execution(task)

        assert plan["task"] == task
        assert SkillDomain.PROCESS_CONTROL.value in plan["classified_domains"]
        assert any(
            s["name"] == "process-control-systems" for s in plan["selected_skills"]
        )
        assert "simulate_thermal_plant" in plan["recommended_mcp_tools"]
        assert "compute_transient_metrics" in plan["recommended_mcp_tools"]
        assert "tune_classical_pid" in plan["recommended_mcp_tools"]
        assert plan["golden_rule_enforced"] is True
        assert len(plan["quality_checks"]) > 0

    def test_custom_catalog_injection(self) -> None:
        custom_catalog = {
            "custom-control": SkillDescriptor(
                name="custom-control",
                tier=SkillTier.CORE_TIER_1,
                domain=SkillDomain.PROCESS_CONTROL,
                description="Custom control skill",
                path=r"C:\fake\path\SKILL.md",
                keywords={"custom", "control"},
            )
        }
        custom_orchestrator = AutonomousSkillOrchestrator(catalog=custom_catalog)
        assert custom_orchestrator.is_skill_allowed("custom-control") is True
        assert custom_orchestrator.is_skill_allowed("process-control-systems") is False
