# -*- coding: utf-8 -*-
"""Utilitário de estilização gráfica para publicações IEEE Transactions em Matplotlib.

Configura tipografia, dimensões, tamanhos de fonte, estilos de linha e paletas de cores
para conformidade estrita com os padrões editoriais do IEEE para figuras de coluna única
(3.5 polegadas / single-column) e coluna dupla (7.16 polegadas / double-column),
garantindo incorporação vetorial de fontes TrueType (Type 42) aceitas pelo IEEE Xplore.
"""

# %% [1. Importações e Configuração de Logging]
import logging
from typing import Dict, Tuple

import matplotlib as mpl

# Configuração de logger específico para o módulo de estilos
logger = logging.getLogger(__name__)

# %% [2. Constantes de Dimensões IEEE Transactions]
# Larguras de referência oficiais da folha de estilo IEEEtran (em polegadas)
IEEE_SINGLE_COL_WIDTH_INCHES: float = 3.50  # Largura para figura em 1 coluna (3.50 in)
IEEE_DOUBLE_COL_WIDTH_INCHES: float = 7.16  # Largura para figura em 2 colunas (7.16 in)

# %% [3. Paleta de Cores e Estilos de Linha (Acessibilidade e Alto Contraste)]
# Paleta compatível com daltonismo (ColorBrewer / Okabe-Ito), permitindo distinção
# imediata tanto em telas coloridas quanto em impressões em escala de cinza.
IEEE_PALETTE: Dict[str, str] = {
    "Ziegler-Nichols": "#D95F02",        # Laranja-avermelhado (Vermilion)
    "Cohen-Coon": "#7570B3",             # Roxo-ardósia (Purple-Slate)
    "Skogestad SIMC": "#1B9E77",         # Verde-azulado escuro (Teal)
    "Optimal ITAE (Ms<=1.6)": "#1F78B4", # Azul marinho clássico (Navy Blue)
    "Optimal ITAE ($M_s \\leq 1.6$)": "#1F78B4",
    "Optimal ITAE ($M_s \\le 1.6$)": "#1F78B4",
    "Optimal ITAE": "#1F78B4",
    "Setpoint": "#222222",               # Quase preto para linha de referência
    "Constraint": "#E41A1C",             # Vermelho alerta para limites físicos
    "Pareto": "#252525",                 # Carvão escuro para curvas de fronteira
}

# Estilos de linha distintos para garantir legibilidade mesmo em publicações monocromáticas
IEEE_LINESTYLES: Dict[str, str] = {
    "Ziegler-Nichols": "--",             # Tracejado
    "Cohen-Coon": "-.",                  # Traço-ponto
    "Skogestad SIMC": "-",               # Contínuo sólido
    "Optimal ITAE (Ms<=1.6)": "-",       # Contínuo sólido
    "Optimal ITAE ($M_s \\leq 1.6$)": "-",
    "Optimal ITAE ($M_s \\le 1.6$)": "-",
    "Optimal ITAE": "-",
    "Setpoint": ":",                     # Pontilhado para setpoint
    "Constraint": ":",                   # Pontilhado para limites de saturação
}

# Marcadores para diagramas de dispersão e pontos discretos de sintonia
IEEE_MARKERS: Dict[str, str] = {
    "Ziegler-Nichols": "x",
    "Cohen-Coon": "^",
    "Skogestad SIMC": "s",
    "Optimal ITAE (Ms<=1.6)": "o",
    "Optimal ITAE ($M_s \\leq 1.6$)": "o",
    "Optimal ITAE ($M_s \\le 1.6$)": "o",
    "Optimal ITAE": "o",
}

# %% [4. Configuração Global de rcParams do Matplotlib]
def setup_ieee_style(single_column: bool = False) -> None:
    """Configura os parâmetros globais do Matplotlib (rcParams) no padrão IEEE Transactions.

    Garante que:
    1. As fontes sejam embutidas como TrueType Tipo 42 (obrigatório para IEEE Xplore).
    2. A tipografia utilize famílias serifadas (Times New Roman / Computer Modern).
    3. As fontes mantenham legibilidade estrita nas dimensões finais impressas.
    4. As grades sejam sutis e não obstruam as curvas de resposta temporal.

    Args:
        single_column: Se True, reduz ligeiramente o tamanho das fontes para caber
                       em figuras compactas de 3.5 polegadas sem truncamento.
    """
    logger.info(
        "Aplicando estilo de publicação IEEE Transactions (single_column=%s)", single_column
    )

    # Definição de hierarquia tipográfica proporcional à largura da figura
    base_font_size = 8.0 if single_column else 8.5
    label_font_size = 8.5 if single_column else 9.0
    title_font_size = 9.0 if single_column else 9.5

    params = {
        # Backend e embutimento vetorial de fontes compatível com IEEE Xplore
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Família serifada idêntica à do LaTeX do artigo
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
            "Computer Modern Roman",
            "serif",
        ],
        "font.size": base_font_size,

        # Títulos e rótulos de eixos
        "axes.titlesize": title_font_size,
        "axes.labelsize": label_font_size,
        "axes.titleweight": "bold",
        "axes.labelweight": "normal",
        "axes.linewidth": 0.8,

        # Marcações de eixos (Ticks) apontando para dentro
        "xtick.labelsize": base_font_size,
        "ytick.labelsize": base_font_size,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.top": True,
        "ytick.right": True,

        # Caixas de legenda discretas e translúcidas
        "legend.fontsize": base_font_size - 0.5,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": False,

        # Linhas de grade sutis para facilitar a leitura de valores
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
        "grid.color": "#AAAAAA",

        # Espessura de traço e resolução de exportação (300 DPI mínimo editorial)
        "lines.linewidth": 1.2,
        "lines.markersize": 5.0,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }

    mpl.rcParams.update(params)

# %% [5. Cálculo Didático de Dimensões de Figuras]
def get_figure_dimensions(
    columns: int = 1,
    aspect_ratio: float = 0.75,
    height_override: float = 0.0,
) -> Tuple[float, float]:
    """Calcula dimensões exatas (largura e altura em polegadas) para o layout IEEEtran.

    Passo a passo didático:
    1. Identifica se a figura ocupará 1 coluna (3.5 in) ou 2 colunas (7.16 in).
    2. Se uma altura explícita for informada via height_override, utiliza-a diretamente.
    3. Caso contrário, multiplica a largura pela razão de aspecto (padrão 4:3 -> 0.75).

    Args:
        columns: 1 para coluna simples (3.50 in) ou 2 para coluna dupla (7.16 in).
        aspect_ratio: Razão altura/largura (padrão 0.75 para proporção áurea/4:3).
        height_override: Se > 0.0, força altura fixa em polegadas.

    Returns:
        Tupla com (largura_em_polegadas, altura_em_polegadas).
    """
    width = IEEE_SINGLE_COL_WIDTH_INCHES if columns == 1 else IEEE_DOUBLE_COL_WIDTH_INCHES
    height = height_override if height_override > 0.0 else width * aspect_ratio
    return (float(width), float(height))
