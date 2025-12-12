# Cat Swarm Optimization (CSO)

Comparación completa de algoritmos de enjambre bio-inspirados: **CSO**, **PSO**, **ACO** y **Firefly**.

## 📖 Descripción

Este proyecto implementa y compara cuatro algoritmos de optimización basados en el comportamiento de enjambres:

1. **Cat Swarm Optimization (CSO)** - Basado en el paper de Chu, Tsai & Pan (2006)
   - Simula gatos en modo descanso/observación (**Seeking Mode**) y caza (**Tracing Mode**)
   
2. **Particle Swarm Optimization (PSO)** - Optimización por enjambre de partículas
   
3. **Ant Colony Optimization (ACO)** - Optimización por colonia de hormigas (versión continua)
   
4. **Firefly Algorithm (FA)** - Algoritmo de luciérnagas

Todos los algoritmos están implementados usando el framework **DEAP** y se comparan en dos funciones de benchmark:
- **Rosenbrock** (minimización)
- **H1** (maximización)

## 🎯 Características

- Implementación completa de 4 algoritmos bio-inspirados
- Comparación justa con los mismos parámetros de población e iteraciones
- Optimización de funciones benchmark clásicas
- Visualizaciones separadas para cada función objetivo
- Métricas de rendimiento comparativas
- Resultados guardados automáticamente como imágenes PNG
- Documentación técnica del paper original incluida

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

```bash
uv run python main.py
```

### Con Python tradicional

Si instalaste con pip:
```bash
python main.py
```

### ¿Qué hace el script?

El script `main.py` ejecuta los **4 algoritmos** (CSO, PSO, ACO, Firefly) en **2 funciones de benchmark**:

1. **Rosenbrock** (minimización): Función clásica con valle estrecho
2. **H1** (maximización): Función multimodal con múltiples óptimos

**Salida generada:**
- `images/cso_pso_aco_rosenbrock.png` - Comparación en función Rosenbrock
- `images/cso_pso_aco_h1.png` - Comparación en función H1
- Tabla comparativa de resultados en consola

## ⚙️ Parámetros de configuración

### Parámetros generales
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `N_AGENTS` | 30 | Tamaño de población (común para todos los algoritmos) |
| `N_ITERATIONS` | 100 | Número de iteraciones |
| `DIMENSIONS` | 2 | Dimensiones del problema |
| `SEED` | 42 | Semilla para reproducibilidad |

### Parámetros específicos de CSO
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `MR` | 0.2 | Mixture Ratio: proporción en Tracing Mode |
| `SMP` | 5 | Seeking Memory Pool: copias generadas |
| `SRD` | 0.2 | Seeking Range: rango de mutación |
| `CDC` | 0.8 | Counts of Dimension to Change |
| `C1` | 2.0 | Constante para tracing mode |

### Parámetros específicos de PSO
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `w` | 0.5 | Factor de inercia |
| `c1` | 1.5 | Coeficiente cognitivo |
| `c2` | 1.5 | Coeficiente social |

### Parámetros específicos de ACO
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `evaporation` | 0.5 | Tasa de evaporación de feromonas |
| `Q` | 1.0 | Constante de depósito de feromonas |

### Parámetros específicos de Firefly
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `alpha` | 0.2 | Paso aleatorio |
| `beta0` | 1.0 | Atracción máxima |
| `gamma` | 1.0 | Coeficiente de absorción de luz |

## 📊 Funciones de benchmark

### Función Rosenbrock (Minimización)
- Función clásica con valle estrecho en forma de banana
- Óptimo global: f(1, 1) = 0
- Rango: [-30, 30]
- Ideal para probar convergencia fina

### Función H1 (Maximización)  
- Función multimodal con múltiples máximos locales
- Rango: [-100, 100]
- Ideal para probar exploración y escape de óptimos locales

## 🏗️ Estructura del proyecto

```
cso/
├── main.py                             # Script principal con implementación y comparación
├── pyproject.toml                      # Configuración del proyecto y dependencias
├── uv.lock                             # Lock file de uv para reproducibilidad
├── README.md                           # Este archivo
├── .python-version                     # Versión de Python del proyecto (3.11)
├── .gitignore                          # Archivos ignorados por git
│
├── 📄 Documentación:
│   └── Cat_Swarm_Optimization.pdf     # Paper original de referencia del CSO
│
└── 📊 images/                          # Resultados generados
    ├── cso_pso_aco_rosenbrock.png     # Comparación en función Rosenbrock
    └── cso_pso_aco_h1.png             # Comparación en función H1
```

## 📊 Resultados

El script genera automáticamente dos gráficas en el directorio `images/`:

- **cso_pso_aco_rosenbrock.png**: Comparación de convergencia en función Rosenbrock (minimización)
  - Muestra curvas de convergencia de los 4 algoritmos
  - Escala logarítmica en eje Y para mejor visualización
  
- **cso_pso_aco_h1.png**: Comparación de convergencia en función H1 (maximización)
  - Muestra la capacidad de exploración de cada algoritmo
  - Ideal para ver escape de óptimos locales

Además, el script imprime una tabla comparativa con los mejores resultados encontrados por cada algoritmo.

## � Algoritmos implementados

### 1. Cat Swarm Optimization (CSO)
Simula el comportamiento de los gatos con dos modos:
- **Seeking Mode**: Los gatos descansan y observan (exploración local)
- **Tracing Mode**: Los gatos persiguen objetivos (explotación)

### 2. Particle Swarm Optimization (PSO)
Basado en el comportamiento social de bandadas de aves:
- Cada partícula recuerda su mejor posición personal
- Las partículas son atraídas hacia el mejor global del enjambre

### 3. Ant Colony Optimization (ACO)
Inspirado en el comportamiento de hormigas buscando comida:
- Las hormigas depositan feromonas en buenos caminos
- Versión continua adaptada para optimización numérica

### 4. Firefly Algorithm (FA)
Basado en el patrón de destello de las luciérnagas:
- Luciérnagas menos brillantes se mueven hacia las más brillantes
- La intensidad depende de la distancia y la calidad de la solución

## �📚 Referencias

- Chu, S. C., Tsai, P. W., & Pan, J. S. (2006). Cat swarm optimization. In Pacific Rim international conference on artificial intelligence (pp. 854-858). Springer, Berlin, Heidelberg.
- Paper incluido en el repositorio: `Cat_Swarm_Optimization.pdf`

## 🛠️ Tecnologías utilizadas

- **Python 3.11+**: Lenguaje de programación
- **uv**: Gestor de paquetes y entornos Python ultrarrápido
- **DEAP**: Framework para algoritmos evolutivos y optimización
- **NumPy**: Computación numérica y operaciones con arrays
- **Matplotlib**: Visualización de datos y generación de gráficas

## 📝 Notas técnicas

- El proyecto utiliza `uv` como gestor de dependencias para garantizar reproducibilidad y velocidad de instalación
- El archivo `uv.lock` asegura que todos instalen exactamente las mismas versiones de dependencias
- Los scripts pueden ejecutarse como notebooks Jupyter gracias a `ipykernel`
- Todos los algoritmos usan la misma semilla aleatoria (42) para comparación justa y reproducible
- Los gráficos se generan automáticamente en alta resolución (300 DPI) en el directorio `images/`
- El código del algoritmo CSO está completamente comentado para facilitar su comprensión

## 🎓 Ejemplo de salida

Al ejecutar el script, verás:
1. Progreso de ejecución de cada algoritmo en cada benchmark
2. Dos ventanas con las gráficas de convergencia
3. Tabla comparativa en consola:

```
Resultados Finales (Mejor encontrado):
--------------------------------------------------
Algoritmo  | Rosenbrock (Min)     | H1 (Max)            
--------------------------------------------------
CSO        | 1.2345e-02           | 8.5432
PSO        | 2.3456e-02           | 8.1234
ACO        | 3.4567e-02           | 7.9876
Firefly    | 1.5678e-02           | 8.3456
--------------------------------------------------
```

## 👨‍💻 Autores

**Juan Pedro García Sanz** y **Adolfo Peña Marín**

Implementación para el Máster en Inteligencia Artificial  
Universidad: [Tu Universidad]  
Fecha: Diciembre 2025

## 📄 Licencia

Este proyecto es para fines educativos y académicos.
