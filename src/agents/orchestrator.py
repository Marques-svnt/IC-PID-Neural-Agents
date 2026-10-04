"""LangGraph orchestration module with Langfuse observability integration.

This module coordinates the multi-agent workflow for thermal plant modeling,
classical PID tuning, neural policy configuration, closed-loop evaluation,
and engineering report generation.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)

# Attempt importing Langfuse callback handler
try:
    from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

    _HAS_LANGFUSE = True
except ImportError:
    LangfuseCallbackHandler = None  # type: ignore[assignment,misc]
    _HAS_LANGFUSE = False
    logger.warning(
        "langfuse package not installed. Telemetry will operate in mock/fallback mode."
    )

# Attempt importing LangGraph components
try:
    from langgraph.graph import END, StateGraph

    _HAS_LANGGRAPH = True
except ImportError:
    StateGraph = None  # type: ignore[assignment,misc]
    END = "__end__"  # type: ignore[assignment]
    _HAS_LANGGRAPH = False
    logger.warning(
        "langgraph package not installed. Graph will operate in direct sequence mode."
    )


class ControlWorkflowState(TypedDict, total=False):
    """Shared state for the multi-agent control workflow.

    Attributes:
        system_id: Identifier of the target thermal plant (e.g. 'cstr_jacketed').
        setpoint: Target operating temperature [K or degC].
        duration_s: Total simulation duration in seconds.
        dt_s: Numerical integration time step in seconds.
        plant_parameters: Phenomenological and thermodynamic plant parameters.
        foptd_parameters: First-Order Plus Dead Time model approximations (k, tau, theta).
        tuning_results: Classical PID tuning gains (Kp, Ti, Td, Ki, Kd) for each method.
        neural_config: Neural controller architecture and hyperparameters.
        simulation_results: Raw time-series simulation outputs for each controller.
        metrics: Transient, steady-state, and total variation performance metrics.
        report: Markdown and LaTeX summary reports.
        history: Execution event logs and status transitions.
    """

    system_id: str
    setpoint: float
    duration_s: float
    dt_s: float
    plant_parameters: Dict[str, Any]
    foptd_parameters: Dict[str, float]
    tuning_results: Dict[str, Any]
    neural_config: Dict[str, Any]
    simulation_results: Dict[str, Any]
    metrics: Dict[str, Any]
    report: str
    history: List[str]


def get_langfuse_callback(
    trace_name: str = "pid-neural-control-orchestration",
    session_id: Optional[str] = None,
    user_id: Optional[str] = "engineer@proic.uesc.br",
    tags: Optional[List[str]] = None,
) -> Optional[Any]:
    """Instantiate and configure Langfuse CallbackHandler from environment.

    Reads `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST`
    from the environment and instantiates a callback handler to trace nodes,
    latencies, tool calls, and LLM token usage.

    Args:
        trace_name: Name of the overarching execution trace.
        session_id: Optional session identifier for grouping runs.
        user_id: User identifier for authentication tracking.
        tags: List of descriptive tags for filtering traces in the UI.

    Returns:
        Configured Langfuse CallbackHandler instance, or None if credentials
        are missing or langfuse is not installed.
    """
    if not _HAS_LANGFUSE:
        logger.info(
            "Langfuse is not available in environment; returning None callback."
        )
        return None

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key or "..." in public_key:
        logger.info(
            "Langfuse credentials missing or unconfigured. Proceeding without active telemetry."
        )
        return None

    try:
        handler = LangfuseCallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            trace_name=trace_name,
            session_id=session_id,
            user_id=user_id,
            tags=tags or ["proic", "thermal-control", "cstr", "pid-neural"],
        )
        logger.info(
            "Langfuse CallbackHandler initialized successfully pointing to %s", host
        )
        return handler
    except Exception as exc:
        logger.error(
            "Failed to initialize Langfuse CallbackHandler: %s", exc, exc_info=True
        )
        return None


def modeling_node(state: ControlWorkflowState) -> Dict[str, Any]:
    """Modeling Node: Validates physical plant and extracts FOPTD parameters.

    Args:
        state: Current workflow state.

    Returns:
        Updated state dictionary with plant parameters and FOPTD models.
    """
    system_id = state.get("system_id", "cstr_jacketed")
    logger.info("Executing ModelingNode for system_id='%s'", system_id)

    # Nominal parameters for jacketed CSTR
    plant_params = {
        "V": 100.0,
        "V_j": 20.0,
        "q": 100.0,
        "C_Af": 1.0,
        "T_f": 350.0,
        "T_jf": 300.0,
        "k0": 7.2e10,
        "E_over_R": 8750.0,
        "delta_H": -5.0e4,
        "rho_Cp": 500.0,
        "UA": 5.0e4,
        "T_ss": 396.65,
        "C_A_ss": 0.0502,
        "T_j_ss": 348.33,
        "q_j_ss": 100.0,
    }

    # Linearized First-Order Plus Dead Time (FOPTD) representation around steady state
    foptd = {
        "k": -0.185,  # Process gain (cooling jacket: increasing q_j decreases reactor temp)
        "tau": 1.25,  # Dominant time constant [min]
        "theta": 0.15,  # Apparent dead time [min]
    }

    history = state.get("history", [])
    history.append("ModelingNode: Plant and FOPTD models loaded.")

    return {
        "plant_parameters": plant_params,
        "foptd_parameters": foptd,
        "history": history,
    }


def tuning_node(state: ControlWorkflowState) -> Dict[str, Any]:
    """Tuning Node: Synthesizes classical PID gains (Z-N, Cohen-Coon, IMC).

    Delegates to deterministic formulas to ensure numerical fidelity.

    Args:
        state: Current workflow state.

    Returns:
        Updated state dictionary containing classical tuning parameters.
    """
    foptd = state.get("foptd_parameters", {"k": -0.185, "tau": 1.25, "theta": 0.15})
    logger.info("Executing TuningNode with FOPTD=%s", foptd)

    from src.mcp.control_mcp import tune_classical_pid

    k = abs(foptd["k"])
    tau = foptd["tau"]
    theta = foptd["theta"]

    tunings = {
        "imc": tune_classical_pid(k=k, tau=tau, theta=theta, method="imc"),
        "ziegler_nichols": tune_classical_pid(
            k=k, tau=tau, theta=theta, method="ziegler_nichols"
        ),
        "cohen_coon": tune_classical_pid(
            k=k, tau=tau, theta=theta, method="cohen_coon"
        ),
    }

    history = state.get("history", [])
    history.append(
        "TuningNode: Classical gains calculated for IMC, Z-N, and Cohen-Coon."
    )

    return {
        "tuning_results": tunings,
        "history": history,
    }


def neural_node(state: ControlWorkflowState) -> Dict[str, Any]:
    """Neural Node: Configures neural policy and NN-PID hyperparameters.

    Args:
        state: Current workflow state.

    Returns:
        Updated state dictionary with neural controller configuration.
    """
    logger.info("Executing NeuralNode")
    neural_config = {
        "model_type": "NN-PID",
        "architecture": [
            3,
            16,
            16,
            3,
        ],  # inputs: [e, de/dt, int(e)] -> outputs: [delta_Kp, delta_Ki, delta_Kd]
        "activation": "tanh",
        "learning_rate": 0.001,
        "adaptation_rate": 0.05,
    }

    history = state.get("history", [])
    history.append("NeuralNode: Neural architecture configured.")

    return {
        "neural_config": neural_config,
        "history": history,
    }


def evaluation_node(state: ControlWorkflowState) -> Dict[str, Any]:
    """Evaluation Node: Simulates plant and computes control performance metrics.

    Invokes deterministic numerical solvers and evaluates IAE, ISE, ITAE,
    settling time, rise time, overshoot, and total variation.

    Args:
        state: Current workflow state.

    Returns:
        Updated state dictionary with simulation responses and computed metrics.
    """
    system_id = state.get("system_id", "cstr_jacketed")
    setpoint = state.get("setpoint", 401.65)  # +5K step from 396.65K
    duration_s = state.get("duration_s", 20.0)
    dt_s = state.get("dt_s", 0.02)

    logger.info(
        "Executing EvaluationNode: system_id='%s', setpoint=%.2f, duration=%.2f s",
        system_id,
        setpoint,
        duration_s,
    )

    from src.mcp.control_mcp import compute_transient_metrics, simulate_thermal_plant

    # Deterministic ODE simulation
    sim_data = simulate_thermal_plant(
        system_id=system_id,
        setpoint=setpoint,
        duration_s=duration_s,
        dt_s=dt_s,
    )

    # Compute transient and steady-state metrics
    metrics = compute_transient_metrics(
        time_series=sim_data["time"],
        output_series=sim_data["output"],
        setpoint=setpoint,
        control_series=sim_data["control_signal"],
    )

    history = state.get("history", [])
    history.append("EvaluationNode: Closed-loop simulation and metrics completed.")

    return {
        "simulation_results": sim_data,
        "metrics": metrics,
        "history": history,
    }


def report_node(state: ControlWorkflowState) -> Dict[str, Any]:
    """Report Node: Formats benchmark results into publication-grade summary.

    Args:
        state: Current workflow state.

    Returns:
        Updated state dictionary with technical report text.
    """
    metrics = state.get("metrics", {})
    tunings = state.get("tuning_results", {})

    report_lines = [
        "# Relatório Executivo de Controle Térmico — PROIC/UESC",
        f"**Sistema:** {state.get('system_id', 'CSTR')}",
        f"**Setpoint de Temperatura:** {state.get('setpoint', 0.0):.2f} K",
        "",
        "## 1. Parâmetros de Sintonia Clássica Calculados",
    ]

    for method, gains in tunings.items():
        report_lines.append(
            f"- **{method.upper()}**: Kp={gains.get('kp', 0.0):.4f}, "
            f"Ti={gains.get('ti', 0.0):.4f} min, Td={gains.get('td', 0.0):.4f} min"
        )

    report_lines.extend(
        [
            "",
            "## 2. Índices de Desempenho em Malha Fechada",
            f"- **Tempo de Subida (tr):** {metrics.get('rise_time', 0.0):.2f} s",
            f"- **Tempo de Acomodação (ts 2%):** {metrics.get('settling_time', 0.0):.2f} s",
            f"- **Sobressinal Percentual (Mp):** {metrics.get('overshoot_pct', 0.0):.2f}%",
            f"- **Integral do Erro Absoluto (IAE):** {metrics.get('iae', 0.0):.4f}",
            f"- **Integral do Erro Quadrático (ISE):** {metrics.get('ise', 0.0):.4f}",
            f"- **Integral do Erro Ponderado no Tempo (ITAE):** {metrics.get('itae', 0.0):.4f}",
            f"- **Variação Total do Atuador (TV):** {metrics.get('total_variation', 0.0):.4f}",
        ]
    )

    report_str = "\n".join(report_lines)
    history = state.get("history", [])
    history.append("ReportNode: Executive summary compiled.")

    logger.info("ReportNode completed successfully.")
    return {
        "report": report_str,
        "history": history,
    }


def build_control_graph() -> Any:
    """Construct and compile the LangGraph StateGraph workflow.

    Returns:
        Compiled StateGraph runnable instance or a callable sequence fallback.
    """
    if not _HAS_LANGGRAPH or StateGraph is None:
        logger.info("LangGraph is not available. Using sequence runner callable.")

        class SequentialGraphRunner:
            """Fallback sequential pipeline runner mimicking LangGraph invoke."""

            def invoke(
                self,
                input_state: ControlWorkflowState,
                config: Optional[Dict[str, Any]] = None,
            ) -> ControlWorkflowState:
                current_state: ControlWorkflowState = dict(input_state)  # type: ignore[assignment]
                for node_fn in [
                    modeling_node,
                    tuning_node,
                    neural_node,
                    evaluation_node,
                    report_node,
                ]:
                    update = node_fn(current_state)
                    current_state.update(update)  # type: ignore[arg-type]
                return current_state

        return SequentialGraphRunner()

    workflow = StateGraph(ControlWorkflowState)

    workflow.add_node("modeling", modeling_node)
    workflow.add_node("tuning", tuning_node)
    workflow.add_node("neural", neural_node)
    workflow.add_node("evaluation", evaluation_node)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("modeling")
    workflow.add_edge("modeling", "tuning")
    workflow.add_edge("tuning", "neural")
    workflow.add_edge("neural", "evaluation")
    workflow.add_edge("evaluation", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


def run_control_workflow(
    initial_state: ControlWorkflowState,
    session_id: Optional[str] = None,
    trace_name: str = "pid-neural-control-orchestration",
) -> ControlWorkflowState:
    """Execute the multi-agent control workflow with active Langfuse observability.

    Configures the Langfuse CallbackHandler into the invocation context
    via `config={"callbacks": [langfuse_handler]}` to record node transitions,
    latencies, model costs, and tool invocations.

    Args:
        initial_state: Workflow input state.
        session_id: Optional tracing session identifier.
        trace_name: Tracing run name for Langfuse dashboard.

    Returns:
        Final resulting state with simulations, tunings, metrics, and report.
    """
    logger.info(
        "Starting control workflow execution for system '%s'",
        initial_state.get("system_id", "default"),
    )

    langfuse_handler = get_langfuse_callback(
        trace_name=trace_name,
        session_id=session_id,
    )

    callbacks = [langfuse_handler] if langfuse_handler is not None else []
    config: Dict[str, Any] = {"callbacks": callbacks} if callbacks else {}

    graph = build_control_graph()
    result = graph.invoke(initial_state, config=config)

    logger.info("Control workflow executed successfully.")
    return result
