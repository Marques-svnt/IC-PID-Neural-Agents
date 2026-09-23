# Relatório Técnico — Laboratório 02: Controle de Temperatura em um Trocador de Calor
**Disciplina:** Controle de Processo Aplicado Computacional  
**Docente:** Prof. Dr. E.R. Edwards  
**Instituição:** Universidade Estadual de Santa Cruz — UESC  
**Departamento:** Departamento de Engenharias e Computação (DEC) — Curso de Engenharia Química  

---

## 1. Contextualização e Descrição Fenomenológica

O controle de temperatura em trocadores de calor industriais através da manipulação de vazão de vapor saturado é um dos problemas clássicos e fundamentais em automação e controle de plantas químicas.

O sistema em estudo opera segundo a primeira lei da termodinâmica aplicada a um sistema com capacitância térmica concentrada:

$$C \frac{dT}{dt} = \dot{Q}_{liq} = \dot{Q}_v - \dot{Q}_{processo} - \dot{Q}_{perdas}$$

Onde:
- $\dot{Q}_v = \frac{\dot{m}_v}{60} \lambda_v$ é a taxa de calor fornecida pela condensação de vapor $[\text{W}]$;
- $\dot{Q}_{processo} = \frac{\dot{m}_{in}}{60} c_p (T_{PV} - T_{in})$ é a taxa de transferência de calor necessária para aquecer a corrente de processo $[\text{W}]$;
- $\dot{Q}_{perdas} = U_L (T_{PV} - T_{amb})$ representa a dissipação de calor convectivo/radiativo para o ambiente $[\text{W}]$;
- $C$ é a capacidade calorífica global do trocador $[\text{J/}^\circ\text{C}]$.

A malha de controle em análise implementa um **Controlador Proporcional puro (P-only)**:
$$e_k = T_{SP} - T_{PV,k}$$
$$u_k = K_p \cdot e_k$$
$$u_{v,k} = \text{sat}_{[0, 100]}(u_0 + u_k)$$
$$\dot{m}_{v,k} = K_v \cdot u_{v,k}$$

---

## 2. Parâmetros e Condições de Operação

| Grandeza | Símbolo | Valor | Unidade |
| :--- | :---: | :---: | :---: |
| Setpoint de temperatura | $T_{SP}$ | 80.0 | $^\circ\text{C}$ |
| Ganho proporcional | $K_p$ | 2.0 | $\%/\,^\circ\text{C}$ |
| Abertura nominal (bias) | $u_0$ | 33.3 | $\%$ |
| Constante da válvula | $K_v$ | 0.01 | $\text{kg/(min}\cdot\%)$ |
| Calor latente de vaporização | $\lambda_v$ | $2.0 \times 10^6$ | $\text{J/kg}$ |
| Calor específico do fluido | $c_p$ | 4000.0 | $\text{J/(kg}\cdot\,^\circ\text{C)}$ |
| Temperatura de entrada | $T_{in}$ | 30.0 | $^\circ\text{C}$ |
| Temperatura ambiente | $T_{amb}$ | 25.0 | $^\circ\text{C}$ |
| Coeficiente global de perdas | $U_L$ | 20.0 | $\text{W/}^\circ\text{C}$ |
| Capacidade térmica do sistema | $C$ | $1.0 \times 10^5$ | $\text{J/}^\circ\text{C}$ |
| Vazão inicial ($t < 10\text{ min}$) | $\dot{m}_{in}$ | 3.0 | $\text{kg/min}$ |
| Vazão perturbada ($t \ge 10\text{ min}$) | $\dot{m}_{in}$ | 4.0 | $\text{kg/min}$ |

---

## 3. Discussão de Engenharia (Respostas Fundamentadas)

### Questão 1: Qual é o comportamento da temperatura durante os primeiros minutos de operação?
**Resposta:**  
Nos minutos iniciais ($t = 0$ a $t = 10\,\text{min}$), a temperatura parte de uma condição inicial fria ($T_{PV} = 70.000\,^\circ\text{C}$) e apresenta uma trajetória ascendente exponencial assimptótica típica de um sistema de primeira ordem estável em malha fechada. A taxa de aquecimento é máxima no início ($\approx 4.0\,^\circ\text{C/min}$ em $t = 0$) devido ao grande erro inicial ($10\,^\circ\text{C}$), que força a válvula a abrir até $53.3\%$. À medida que o calor se acumula no trocador, a derivada $\frac{dT}{dt}$ desacelera progressivamente até que o sistema atinge praticamente o regime permanente em $t = 10\,\text{min}$, com $T_{PV} = 79.995\,^\circ\text{C} \approx T_{SP}$.

---

### Questão 2: O que acontece com o erro de controle à medida que $T_{PV}$ se aproxima do setpoint?
**Resposta:**  
Pela definição matemática do erro de controle atuante:
$$e_k = T_{SP} - T_{PV,k}$$
À medida que $T_{PV}$ aumenta e se aproxima de $80.0\,^\circ\text{C}$, a magnitude do erro $e_k$ decresce monotonicamente, partindo de $e_0 = +10.0\,^\circ\text{C}$ em $t=0$, caindo para $+0.48\,^\circ\text{C}$ em $t=4\,\text{min}$ e convergindo para apenas $+0.005\,^\circ\text{C}$ em $t=10\,\text{min}$. O erro decai na mesma proporção em que o sistema se aproxima do equilíbrio térmico.

---

### Questão 3: Como a redução do erro modifica a abertura da válvula e a vazão de vapor?
**Resposta:**  
O controlador proporcional calcula a correção na variável manipulada de forma linear ao erro:
$$u_k = K_p \cdot e_k \implies u_{v,k} = u_0 + K_p \cdot e_k$$
Como o erro $e_k$ diminui, a ação $u_k$ é proporcionalmente reduzida (de $+20\%$ em $t=0$ para $+0.01\%$ em $t=10$). Consequentemente:
1. A abertura total da válvula $u_{v}$ recua de $53.3\%$ até o patamar nominal de operação $u_0 = 33.3\%$;
2. A vazão de vapor, governada pela relação de fluxo $\dot{m}_{v} = K_v \cdot u_v$, diminui de $0.533\,\text{kg/min}$ para $0.333\,\text{kg/min}$;
3. A potência térmica fornecida $\dot{Q}_v$ reduz-se de $17.767\,\text{W}$ para aproximadamente $11.100\,\text{W}$, dosando o suprimento de energia para coincidir estritamente com a demanda de regime permanente da corrente de $3.0\,\text{kg/min}$.

---

### Questão 4: O que ocorre com o balanço de energia quando a vazão da corrente de processo aumenta de 3.0 kg/min para 4.0 kg/min?
**Resposta:**  
Em $t = 10\,\text{min}$, ocorre um distúrbio de carga térmica tipo degrau (+33.3% na vazão de alimentação). Como a potência absorvida pelo processo é governada por:
$$\dot{Q}_{processo} = \frac{\dot{m}_{in}}{60} c_p (T_{PV} - T_{in})$$
No instante exato do distúrbio (com $T_{PV} \approx 80^\circ\text{C}$), a carga térmica salta bruscamente de $\approx 10.000\,\text{W}$ para $\approx 13.333\,\text{W}$. Uma vez que a válvula ainda se encontrava na abertura de regime anterior ($u_v \approx 33.3\%$, fornecendo $\approx 11.111\,\text{W}$), o balanço instantâneo de energia entra em déficit agudo:
$$\dot{Q}_{liq} = 11.111 - 13.333 - 1.100 = -3.322\,\text{W} < 0$$
O trocador passa a perder significativamente mais calor do que recebe.

---

### Questão 5: Por que a temperatura começa a diminuir após a perturbação?
**Resposta:**  
A resposta decorre diretamente do princípio de conservação de energia transiente (Primeira Lei da Termodinâmica para volume de controle):
$$\frac{dT}{dt} = \frac{\dot{Q}_{liq}}{C}$$
Como a vazão aumentada de líquido frio absorve mais energia do que a capacidade instantânea fornecida pelo vapor condensante, $\dot{Q}_{liq}$ torna-se fortemente negativo. Sendo a capacidade calorífica $C$ uma constante positiva, a derivada temporal $\frac{dT}{dt}$ torna-se negativa. A perda líquida de energia interna estocada na massa do trocador e no fluido resulta inevitavelmente na queda contínua da temperatura de saída $T_{PV}$.

---

### Questão 6: Como o controlador proporcional reage à redução da temperatura?
**Resposta:**  
À medida que $T_{PV}$ cai (passando de $79.995^\circ\text{C}$ para $76.505^\circ\text{C}$), o desvio em relação ao setpoint volta a aumentar:
$$e = 80.0 - T_{PV} > 0$$
O controlador proporcional detecta o aumento do erro positivo e reage instantaneamente amplificando a abertura:
$$u = K_p \cdot e = 2.0 \cdot (80.0 - T_{PV})$$
O sinal $u$ sobe de praticamente $0\%$ até cerca de $+6.99\%$, forçando a válvula de vapor a abrir para $u_v = 33.3 + 6.99 \approx 40.29\%$. Essa abertura extra eleva a vazão mássica de vapor para $\dot{m}_v \approx 0.403\,\text{kg/min}$ ($\dot{Q}_v \approx 13.430\,\text{W}$), desacelerando a queda de temperatura e conduzindo o trocador a um novo estado de equilíbrio térmico dinâmico.

---

### Questão 7: A temperatura retorna exatamente ao setpoint após a perturbação? Justifique o resultado observado com base nas características de um controlador proporcional.
**Resposta:**  
**Não.** A temperatura se estabiliza em $76.505\,^\circ\text{C}$, restando um desvio permanente de regime de:
$$\text{Offset} = e_{ss} = 80.0 - 76.505 = 3.495\,^\circ\text{C}$$

**Justificativa Teórica:**  
Esta é a limitação clássica intrínseca e estrutural do **Controlador Proporcional puro (P)**.
Para que o sistema opere estavelmente com a nova vazão de $4.0\,\text{kg/min}$, é necessária uma vazão contínua de vapor de aproximadamente $0.403\,\text{kg/min}$, o que exige que a válvula permaneça aberta em $u_v \approx 40.3\%$.
Como a abertura da válvula é calculada por:
$$u_v = u_0 + K_p \cdot e$$
Se a temperatura retornasse exatamente a $80\,^\circ\text{C}$, o erro $e$ seria igual a zero. Mas, se $e = 0$, a abertura da válvula retornaria ao bias nominal:
$$u_v = u_0 + 0 = 33.3\%$$
Com $33.3\%$ de abertura, o vapor fornece apenas $11.111\,\text{W}$, o que é insuficiente para manter $4.0\,\text{kg/min}$ de processo a $80\,^\circ\text{C}$ (demanda de $13.333\,\text{W}$). O sistema imediatamente se resfriaria novamente.  
Logo, **é matematicamente e fisicamente obrigatório que exista um erro residual permanente ($e \neq 0$)** para fornecer a ação de sustentação $u = K_p \cdot e$ que mantém a válvula aberta na nova posição requerida.  
*Nota de Engenharia:* Para eliminar completamente esse erro de regime (*offset*), é indispensável introduzir a **ação Integral (controlador PI ou PID)**, cujo termo $\frac{1}{T_i}\int e\,dt$ é capaz de sustentar o sinal de controle sem necessitar de erro não nulo em regime estacionário.

---

### Questão 8: Qual é a relação observada entre o sinal de $\dot{Q}_{liq}$ e o sinal de $dT/dt$?
**Resposta:**  
O sinal de $\dot{Q}_{liq}$ e o sinal de $\frac{dT}{dt}$ são rigorosamente idênticos em todos os instantes temporais:
$$\text{sgn}\left(\frac{dT}{dt}\right) \equiv \text{sgn}(\dot{Q}_{liq})$$
Isso decorre da relação direta de proporcionalidade estabelecida pela capacitância térmica $C > 0$:
$$\frac{dT}{dt} = \frac{1}{C} \dot{Q}_{liq}$$
- **Se $\dot{Q}_{liq} > 0$:** há acúmulo de entalpia no volume de controle $\implies \frac{dT}{dt} > 0$ (processo de aquecimento);
- **Se $\dot{Q}_{liq} < 0$:** há depleção de entalpia interna $\implies \frac{dT}{dt} < 0$ (processo de resfriamento);
- **Se $\dot{Q}_{liq} = 0$:** as taxas de calor fornecidas e retiradas se anulam perfeitamente $\implies \frac{dT}{dt} = 0$ (regime estacionário / equilíbrio térmico).

---

### Questão 9: Em quais regiões do gráfico o sistema pode ser considerado próximo do equilíbrio térmico?
**Resposta:**  
O sistema encontra-se próximo do equilíbrio térmico ($\dot{Q}_{liq} \approx 0$ e $\frac{dT}{dt} \approx 0$) em duas regiões temporais distintas:
1. **Região 1 ($t = 8$ a $10\,\text{min}$):** Primeiro regime permanente sob carga nominal ($\dot{m}_{in} = 3.0\,\text{kg/min}$), com $T_{PV} \approx 79.995\,^\circ\text{C}$, $\dot{Q}_{liq} \approx 0.05\,\text{W}$ e $\frac{dT}{dt} \approx 0.00003\,^\circ\text{C/s}$.
2. **Região 2 ($t = 16$ a $19\,\text{min}$):** Segundo regime permanente após acomodação da perturbação de carga ($\dot{m}_{in} = 4.0\,\text{kg/min}$), onde $T_{PV} \approx 76.505\,^\circ\text{C}$, $\dot{Q}_{liq} \approx 0.3\,\text{W}$ e $\frac{dT}{dt} \to 0$.

---

### Questão 10: Explicação da Sequência Física Completa:
$$T_{PV} \to e \to u \to u_v \to \dot{m}_v \to \dot{Q}_v \to \dot{Q}_{liq} \to \frac{dT}{dt}$$

**Resposta:**  
A sequência sintetiza a física da causalidade em uma malha fechada com atuador termofluidodinâmico:
1. **$T_{PV}$ (Variável de Processo):** A temperatura do fluido aquecido é mensurada na saída do trocador de calor pelo transmissor de temperatura.
2. **$e$ (Erro de Controle):** O elemento de comparação calcula a discrepância instantânea entre a meta de projeto e a condição medida: $e = T_{SP} - T_{PV}$.
3. **$u$ (Ação de Controle):** O algoritmo de controle proporcional traduz essa discrepância em uma solicitação de correção percentual: $u = K_p \cdot e$.
4. **$u_v$ (Posição da Válvula):** O sinal é somado ao bias operacional da válvula e passa pela saturação física mecânica ($0 \le u_v \le 100\%$).
5. **$\dot{m}_v$ (Vazão de Vapor):** O curso da haste da válvula altera a área útil de escoamento, modulando hidraulicamente o fluxo mássico de vapor de aquecimento: $\dot{m}_v = K_v u_v$.
6. **$\dot{Q}_v$ (Calor Fornecido):** Ao entrar em contato com as superfícies frias do trocador, o vapor condensa a pressão constante, liberando seu calor latente de vaporização como fluxo térmico para a carcaça/tubos: $\dot{Q}_v = \frac{\dot{m}_v}{60} \lambda_v$.
7. **$\dot{Q}_{liq}$ (Potência Líquida Acumulada):** Realiza-se o balanço térmico no volume de controle, descontando do calor recebido o calor absorvido pelo fluido que escoa continuamente ($\dot{Q}_{processo}$) e as perdas externas ($\dot{Q}_{perdas}$).
8. **$\frac{dT}{dt}$ (Inércia e Dinâmica Térmica):** A energia térmica líquida retida ou drenada altera a temperatura interna de acordo com a capacitância calorífica do equipamento ($C \frac{dT}{dt} = \dot{Q}_{liq}$), fechando a dinâmica temporal do processo contínuo.
