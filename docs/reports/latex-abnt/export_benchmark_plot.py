"""
Gera o arquivo LaTeX com o grafico em PGFPlots/TikZ para o Capitulo 6.
"""
import numpy as np
from simulate_benchmark import T_cc, T_itae, T_simc, T_zn, q_cc, q_itae, q_simc, q_zn, t

# Amostragem para PGFPlots (a cada 5 pontos = 0.1 min, 201 pontos)
step = 5
idx = np.arange(0, len(t), step)

def format_coords(x, y):
    lines = []
    for i in idx:
        lines.append(f"({x[i]:.2f}, {y[i]:.2f})")
    return " ".join(lines)

# Setpoint curve
sp_coords = "(0, 401.65) (20, 401.65)"
sat_max_coords = "(0, 300.0) (20, 300.0)"
sat_min_coords = "(0, 0.0) (20, 0.0)"

simc_T = format_coords(t, T_simc)
itae_T = format_coords(t, T_itae)
zn_T = format_coords(t, T_zn)
cc_T = format_coords(t, T_cc)

simc_q = format_coords(t, q_simc)
itae_q = format_coords(t, q_itae)
zn_q = format_coords(t, q_zn)
cc_q = format_coords(t, q_cc)

latex_code = f"""% fig_benchmark.tex - Grafico comparativo de transientes
\\begin{{figure}}[htbp]
\\centering
\\begin{{subfigure}}{{\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    width=0.92\\textwidth,
    height=5.2cm,
    ylabel={{Temperatura $T$ [\\si{{\\kelvin}}],}},
    xmin=0, xmax=20,
    ymin=390, ymax=445,
    grid=both,
    grid style={{line width=.1pt, draw=gray!25}},
    major grid style={{line width=.2pt, draw=gray!40}},
    legend pos=north east,
    legend cell align={{left}},
    legend style={{font=\\scriptsize, fill=white, fill opacity=0.9, draw=gray!40}},
    tick label style={{font=\\footnotesize}},
    label style={{font=\\small}}
]
    % Setpoint
    \\addplot[line width=1.0pt, dashed, color=black] coordinates {{{sp_coords}}};
    \\addlegendentry{{Referência $T_{{\\text{{sp}}}}$}}

    % SIMC
    \\addplot[line width=1.3pt, color=NavyBlue] coordinates {{{simc_T}}};
    \\addlegendentry{{SIMC ($\\tau_c = 0{{,}}40\\,\\text{{min}}$)}}

    % ITAE
    \\addplot[line width=1.2pt, color=ForestGreen, dashdotted] coordinates {{{itae_T}}};
    \\addlegendentry{{ITAE Ótimo}}

    % ZN
    \\addplot[line width=1.1pt, color=BrickRed, dotted] coordinates {{{zn_T}}};
    \\addlegendentry{{Ziegler-Nichols}}

    % CC
    \\addplot[line width=1.0pt, color=BurntOrange, dashed] coordinates {{{cc_T}}};
    \\addlegendentry{{Cohen-Coon}}

    % Marcador de perturbacao
    \\node[draw=gray!60, fill=white, rounded corners=2pt, font=\\tiny, align=center] 
        at (axis cs:10, 428) {{Perturbação em $T_f$\\\\($+10\\,\\text{{K}}$ em $t=10\\,\\text{{min}}$)}};
    \\draw[->, line width=0.6pt, gray!70] (axis cs:10, 422) -- (axis cs:10, 403);

\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Resposta temporal da temperatura do reator $T(t)$ sob degrau de setpoint ($+5\\,\\si{{\\kelvin}}$) e perturbação ($+10\\,\\si{{\\kelvin}}$ em $t=10\\,\\si{{\\minute}}$).}}
\\label{{fig:benchmark_temperatura}}
\\end{{subfigure}}

\\vspace{{0.4cm}}

\\begin{{subfigure}}{{\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    width=0.92\\textwidth,
    height=4.8cm,
    xlabel={{Tempo $t$ [\\si{{\\minute}}]}},
    ylabel={{Vazão $q_j$ [\\si{{\\liter\\per\\minute}}]}},
    xmin=0, xmax=20,
    ymin=-10, ymax=330,
    grid=both,
    grid style={{line width=.1pt, draw=gray!25}},
    major grid style={{line width=.2pt, draw=gray!40}},
    legend pos=north east,
    legend cell align={{left}},
    legend style={{font=\\scriptsize, fill=white, fill opacity=0.9, draw=gray!40}},
    tick label style={{font=\\footnotesize}},
    label style={{font=\\small}}
]
    % Limites de saturacao
    \\addplot[line width=1.0pt, dashed, color=red!80!black] coordinates {{{sat_max_coords}}};
    \\addlegendentry{{Limite $u_{{\\max}} = 300\\,\\text{{L/min}}$}}

    % SIMC
    \\addplot[line width=1.3pt, color=NavyBlue] coordinates {{{simc_q}}};
    \\addlegendentry{{SIMC ($\\tau_c = 0{{,}}40\\,\\text{{min}}$)}}

    % ITAE
    \\addplot[line width=1.2pt, color=ForestGreen, dashdotted] coordinates {{{itae_q}}};
    \\addlegendentry{{ITAE Ótimo}}

    % ZN
    \\addplot[line width=1.1pt, color=BrickRed, dotted] coordinates {{{zn_q}}};
    \\addlegendentry{{Ziegler-Nichols}}

    % CC
    \\addplot[line width=1.0pt, color=BurntOrange, dashed] coordinates {{{cc_q}}};
    \\addlegendentry{{Cohen-Coon}}

\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Esforço de controle (vazão da camisa $q_j(t)$) evidenciando a saturação crônica nos métodos heurísticos e modulação suave em SIMC e ITAE.}}
\\label{{fig:benchmark_atuador}}
\\end{{subfigure}}

\\caption{{Comportamento dinâmico comparativo em malha fechada do CSTR sob os quatro métodos de sintonia avaliados, demonstrando a estabilização por SIMC e ITAE contra a divergência catastrófica por Ziegler-Nichols e Cohen-Coon.}}
\\label{{fig:benchmark_completo}}
\\fonte{{Elaborada pelos autores (2026).}}
\\end{{figure}}
"""

with open("secoes/fig_benchmark.tex", "w", encoding="utf-8") as f:
    f.write(latex_code)

print("secoes/fig_benchmark.tex generated successfully.")
