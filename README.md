# Cat Swarm Optimization (CSO)

Implementación del algoritmo **Cat Swarm Optimization (CSO)** y comparación con otros algoritmos bio-inspirados (PSO y Firefly).

## 📖 Descripción

Este proyecto implementa el algoritmo CSO basado en el paper:
> "Cat Swarm Optimization" por Shu-Chuan Chu, Pei-Wei Tsai, and Jeng-Shyang Pan (2006)

El algoritmo simula el comportamiento de los gatos que pasan la mayor parte del tiempo descansando y observando (**Seeking Mode**) y un pequeño porcentaje cazando (**Tracing Mode**).

## 🎯 Características

- **CSO.py**: Implementación completa del algoritmo Cat Swarm Optimization
- **CSO_comparison.py**: Comparación entre CSO, PSO y Firefly Algorithm
- Optimización de la función de benchmark **Shekel** con múltiples máximos locales
- Visualizaciones interactivas en 2D y 3D
- Métricas de rendimiento y análisis comparativo

## 📋 Requisitos

- Python >= 3.11
- Dependencias (se instalan automáticamente):
  - numpy >= 1.24.0
  - deap >= 1.4.0
  - matplotlib >= 3.7.0

## 🚀 Instalación

1. Clona o descarga este repositorio
2. Instala las dependencias usando `uv` o `pip`:

```bash
# Con uv (recomendado)
uv pip install -e .

# Con pip
pip install -e .
```

## 💻 Uso

### Ejecutar CSO

```bash
python CSO.py
```

Este script ejecuta el algoritmo CSO y genera:
- Gráfica de convergencia
- Visualización 3D del paisaje de optimización
- Animación del movimiento de los gatos (opcional)

### Ejecutar comparación de algoritmos

```bash
python CSO_comparison.py
```

Este script compara el rendimiento de:
- Cat Swarm Optimization (CSO)
- Particle Swarm Optimization (PSO)
- Firefly Algorithm

Genera visualizaciones comparativas de convergencia y rendimiento.

## ⚙️ Parámetros del algoritmo CSO

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `N_CATS` | 50 | Número de gatos (población) |
| `N_ITERATIONS` | 100 | Número de iteraciones |
| `MR` | 0.20 | Mixture Ratio: proporción en Tracing Mode |
| `SMP` | 5 | Seeking Memory Pool: copias generadas en seeking mode |
| `SRD` | 0.2 | Seeking Range: rango de mutación (20% del rango) |
| `CDC` | 0.8 | Counts of Dimension to Change (80% de dimensiones) |
| `C1` | 2.0 | Constante para tracing mode |

## 📊 Funciones de benchmark

El proyecto utiliza la **función Shekel** con tres máximos locales en 2 dimensiones, ideal para probar algoritmos de optimización en problemas multimodales.

## 🏗️ Estructura del proyecto

```
cso/
├── CSO.py                 # Implementación del algoritmo CSO
├── CSO_comparison.py      # Comparación con otros algoritmos
├── main.py                # Script principal (placeholder)
├── pyproject.toml         # Configuración del proyecto
└── README.md              # Este archivo
```

## 📚 Referencias

- Chu, S. C., Tsai, P. W., & Pan, J. S. (2006). Cat swarm optimization. In Pacific Rim international conference on artificial intelligence (pp. 854-858). Springer, Berlin, Heidelberg.

## 👨‍💻 Autor

Implementación para el Máster en Inteligencia Artificial  
Fecha: Diciembre 2025

## 📄 Licencia

Este proyecto es para fines educativos.
