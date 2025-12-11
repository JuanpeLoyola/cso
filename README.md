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
- Visualizaciones interactivas en 2D y 3D con matplotlib
- Métricas de rendimiento y análisis comparativo
- Resultados pre-generados incluidos en el repositorio
- Documentación técnica en formato PDF

## 📋 Requisitos

- Python >= 3.11
- **uv** - Gestor de paquetes y entornos Python (recomendado)
- Dependencias (se instalan automáticamente con `uv sync`):
  - numpy >= 1.24.0
  - deap >= 1.4.0
  - matplotlib >= 3.7.0
  - ipykernel >= 7.1.0

## 🚀 Instalación

### Opción 1: Con uv (Recomendado)

1. Clona el repositorio:
```bash
git clone https://github.com/JuanpeLoyola/cso.git
cd cso
```

2. Sincroniza el entorno y las dependencias:
```bash
uv sync
```

Esto creará automáticamente un entorno virtual y instalará todas las dependencias especificadas en `pyproject.toml`.

### Opción 2: Con pip tradicional

```bash
pip install -e .
```

## 💻 Uso

### Con uv (Recomendado)

#### Ejecutar CSO
```bash
uv run python CSO.py
```

Este script ejecuta el algoritmo CSO y genera:
- Gráfica de convergencia (`CSO_results.png`)
- Visualización 3D del paisaje de optimización
- Animación del movimiento de los gatos (opcional)
- Métricas de rendimiento en consola

#### Ejecutar comparación de algoritmos
```bash
uv run python CSO_comparison.py
```

Este script compara el rendimiento de tres algoritmos:
- **Cat Swarm Optimization (CSO)**
- **Particle Swarm Optimization (PSO)**
- **Firefly Algorithm**

Genera visualizaciones comparativas:
- `algorithms_comparison.png` - Comparación general
- `algorithms_comparison_shekel.png` - Comparación específica en función Shekel

### Con Python tradicional

Si instalaste con pip, puedes ejecutar directamente:
```bash
python CSO.py
python CSO_comparison.py
```

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
├── CSO.py                              # Implementación del algoritmo CSO
├── CSO_comparison.py                   # Comparación con PSO y Firefly
├── main.py                             # Script principal (placeholder)
├── pyproject.toml                      # Configuración del proyecto y dependencias
├── uv.lock                             # Lock file de uv para reproducibilidad
├── README.md                           # Este archivo
├── .python-version                     # Versión de Python del proyecto
├── .gitignore                          # Archivos ignorados por git
│
├── 📄 Documentación:
│   └── Cat_Swarm_Optimization.pdf     # Paper de referencia del algoritmo
│
└── 📊 Resultados generados:
    ├── CSO_results.png                 # Visualización de resultados CSO
    ├── algorithms_comparison.png       # Comparación de algoritmos
    └── algorithms_comparison_shekel.png # Comparación en función Shekel
```

## 📊 Resultados

El repositorio incluye resultados pre-generados de las ejecuciones:

- **CSO_results.png**: Gráficas de convergencia y exploración del algoritmo CSO
- **algorithms_comparison.png**: Comparación visual del rendimiento de CSO vs PSO vs Firefly
- **algorithms_comparison_shekel.png**: Análisis específico en la función de benchmark Shekel

Estos archivos se sobrescriben cada vez que ejecutas los scripts.

## 📚 Referencias

- Chu, S. C., Tsai, P. W., & Pan, J. S. (2006). Cat swarm optimization. In Pacific Rim international conference on artificial intelligence (pp. 854-858). Springer, Berlin, Heidelberg.
- Paper incluido en el repositorio: `Cat_Swarm_Optimization.pdf`

## 🛠️ Tecnologías utilizadas

- **Python 3.11+**: Lenguaje de programación
- **uv**: Gestor de paquetes y entornos Python ultrarrápido
- **DEAP**: Framework para algoritmos evolutivos
- **NumPy**: Computación numérica
- **Matplotlib**: Visualización de datos

## 📝 Notas

- El proyecto utiliza `uv` como gestor de dependencias para garantizar reproducibilidad y velocidad
- El archivo `uv.lock` asegura que todos instalen exactamente las mismas versiones de dependencias
- Los scripts pueden ejecutarse como notebooks Jupyter gracias a `ipykernel`
- Los gráficos se generan automáticamente y se guardan en el directorio raíz

## 👨‍💻 Autor

Implementación para el Máster en Inteligencia Artificial  
Fecha: Diciembre 2025

## 📄 Licencia

Este proyecto es para fines educativos.
