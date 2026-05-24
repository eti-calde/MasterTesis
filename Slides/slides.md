---
theme: default
title: Surrogates Physics-Informed para Modelos Hidrodinámicos
info: Brainstorming — Líneas de investigación
transition: slide-left
mdc: true
math: katex
---

# Surrogates Physics-Informed para Modelos Hidrodinámicos

**Brainstorming — Líneas de investigación**

Marzo 2026

---

## Motivación

Los solvers hidrodinámicos (TELEMAC, ADCIRC, EFDC...) resuelven las ecuaciones numéricamente con un alto costo computacional.

<v-clicks>

- Cada simulación puede tomar **horas a días** de CPU
- Calibración requiere **cientos de evaluaciones** del modelo
- Análisis de escenarios → multiplicar el costo × $N$
- Alto costo computacional y pronóstico en tiempo real no van de la mano

</v-clicks>

<!--
Contextualizar con nuestros casos: corrientes costeras, dispersión, oleaje. El cuello de botella es el mismo en todos.
-->

---

## Mini-intro: Deep Learning en 2 minutos

<div class="grid grid-cols-2 gap-6">
<div>

<img src="/nn-diagram.png" class="h-72 rounded shadow" />

</div>
<div>

- **Input layer**: datos de entrada (coordenadas, parámetros)
- **Hidden layers**: capas ocultas con neuronas conectadas
- **Pesos** $W$: las conexiones — lo que se aprende
- **Activación** $\sigma$: no linealidad ($\tanh$, ReLU) en cada neurona
- **Loss function** $\mathcal{L}$: mide el error de la predicción
- **Backpropagation**: calcula gradientes $\nabla_W \mathcal{L}$ para ajustar los pesos

</div>
</div>

<br>

**Diferenciación automática (AD)**: el mismo mecanismo del backprop puede calcular derivadas de la solución $\partial u / \partial x$, $\partial u / \partial t$ → la clave de todo lo que sigue

<!--
No profundizar en backprop. El punto clave es: (1) la red aproxima funciones, (2) AD da derivadas gratis.
-->

---

## De datos a física

<div class="grid grid-cols-2 gap-8">
<div>

### Data-driven (ML clásico)

$$\mathcal{L} = \|u_\theta - u_{\text{datos}}\|^2$$

- Necesita **muchos** datos
- No sabe física
- Puede predecir cosas no físicas

</div>
<div>

### Physics-informed (PINNs)

$$\mathcal{L} = \underbrace{\|u_\theta - u_{\text{datos}}\|^2}_{\text{datos}} + \underbrace{\left\|\frac{\partial u}{\partial t} + \mathcal{N}[u]\right\|^2}_{\text{residuo PDE}}$$

- Necesita **pocos datos** (o ninguno)
- La PDE actúa como regularizador

</div>
</div>

<!--
Raissi 2019 — paper fundacional. Con 100 puntos + ecuación de Burgers obtiene errores de 0.07%.
-->

---

## PINNs: la idea central

Una sola red resuelve la PDE **sin malla**:

<div class="grid grid-cols-2 gap-8">
<div>

**Input**: $(x, y, t)$

**Output**: $(h, u, v)$

**Loss**: que se cumpla la PDE + ICs/BCs

</div>
<div>

**Ventajas**:
- Mesh-free
- Pocos datos
- Robustez al ruido
- Problema inverso "gratis"

**Limitación clave**:
- 1 PINN = 1 configuración
- Cambiar $Q$, $n$, BCs → **reentrenar**

</div>
</div>

<!--
Esto motiva el salto a operator learning — queremos entrenar 1 vez y predecir N escenarios.
-->

---

## El salto conceptual: Operator Learning

<div class="grid grid-cols-2 gap-12">
<div>

### PINNs
Aprende una **función**

$$f: (x, y, t) \to (h, u, v)$$

Para **una** configuración fija

</div>
<div>

### Neural Operator
Aprende un **operador**

$$\mathcal{G}: Q(t) \mapsto h(x, y, t)$$

Para **cualquier** input nuevo

</div>
</div>

<br>

<div class="text-center text-xl">

Entrenas **1 vez** con $N$ escenarios → predices el escenario $N+1$ **sin reentrenar**

</div>

<!--
El operador mapea funciones a funciones. El input puede ser un hidrograma Q(t), un campo de Manning n(x,y), condiciones de frontera — cualquier función.
-->

---

## Zoo de arquitecturas

| Arquitectura | Cómo funciona | Mallas irregulares | Physics loss |
|---|---|---|---|
| **DeepONet** | Branch (input) × Trunk (coords) | Si | Si (PI-DeepONet) |
| **FNO** | Capas de Fourier (FFT) | No (grid regular) | Si (PINO) |
| **GNN-based** | Message passing en grafos | Si | Parcial |
| **Latent-NO** | DeepONet en espacio latente | Si (via encoder) | Si (PI-Latent-NO) |

<br>

**Speedups reportados**: $100\times$ a $10.000\times$ vs solvers clásicos

**Precisión**: errores relativos $0.3\%$ - $10\%$ dependiendo del problema

<!--
No hay "mejor" arquitectura universal — depende del problema. DeepONet es más flexible para mallas irregulares, FNO es más rápido en grids regulares. La tendencia actual (2025-2026) es combinar latent space + physics.
-->

---

## Aplicaciones para nuestro contexto

<v-clicks>

1. **Surrogate para simulación rápida**: reemplazar el solver por un neural operator para predicción en tiempo real de corrientes, oleaje, niveles

2. **Calibración automática**: usar el surrogate dentro de Bayesian Optimization → calibrar parámetros en minutos en vez de semanas

3. **Análisis de escenarios**: evaluar miles de combinaciones de caudal/oleaje/viento — análisis de riesgo, planificación

4. **Problema inverso**: inferir parámetros del modelo (fricción, batimetría) directamente de observaciones de campo

</v-clicks>

<!--
La metodología es agnóstica al solver específico. Si funciona con TELEMAC, funciona con cualquier solver que resuelva SWE o N-S.
-->
