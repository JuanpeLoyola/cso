"""
Cat Swarm Optimization (CSO) Algorithm

Implementación del algoritmo Cat Swarm Optimization basado en el paper:
"Cat Swarm Optimization" por Shu-Chuan Chu, Pei-Wei Tsai, and Jeng-Shyang Pan (2006)

El algoritmo simula el comportamiento de los gatos que pasan la mayor parte del tiempo
descansando y observando (Seeking Mode) y un pequeño porcentaje cazando (Tracing Mode).

Parámetros principales:
- MR (Mixture Ratio): Proporción de gatos en Tracing Mode vs Seeking Mode
- SMP (Seeking Memory Pool): Número de copias en modo búsqueda
- SRD (Seeking Range of the Selected Dimension): Rango de mutación
- CDC (Counts of Dimension to Change): Número de dimensiones a modificar
- c1: Constante para tracing mode (similar a PSO)

Author: Implementación para el Máster en IA
Date: Diciembre 2025
"""

#%%
import random
import numpy as np
from deap import base, creator, tools, benchmarks
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# PARÁMETROS DEL ALGORITMO CSO
# ============================================================================

N_CATS = 50              # Número de gatos (población)
N_ITERATIONS = 100       # Número de iteraciones
MR = 0.20                # Mixture Ratio: proporción en Tracing Mode (20% típico)
SMP = 5                  # Seeking Memory Pool: copias generadas en seeking mode
SRD = 0.2                # Seeking Range: rango de mutación (20% del rango)
CDC = 0.8                # Counts of Dimension to Change (80% de dimensiones)
C1 = 2.0                 # Constante para tracing mode
DIMENSIONS = 2           # Dimensiones del problema

# Límites del espacio de búsqueda
BOUNDS_MIN = -100.0
BOUNDS_MAX = 100.0

# Semillas para reproducibilidad
random.seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURACIÓN DE DEAP
# ============================================================================

# Crear tipos para MAXIMIZACIÓN (función H1 busca máximos)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Cat", list, fitness=creator.FitnessMax, 
               velocity=list, mode=str, flag=bool)

toolbox = base.Toolbox()

def h1_function(individual):
    """
    Función benchmark H1 de DEAP.
    Es una función bidimensional con múltiples máximos locales.
    Utilizada para probar algoritmos de optimización global.
    """
    result = benchmarks.h1(individual)[0]
    return (result,)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def init_cat():
    """
    Inicializa un gato con:
    - Posición aleatoria en el espacio de búsqueda
    - Velocidad inicial aleatoria pequeña
    - Modo aleatorio (seeking o tracing según MR)
    - Flag para marcar el mejor gato
    """
    cat = creator.Cat(
        random.uniform(BOUNDS_MIN, BOUNDS_MAX) for _ in range(DIMENSIONS)
    )
    # Velocidad inicial pequeña
    cat.velocity = [random.uniform(-1, 1) for _ in range(DIMENSIONS)]
    # Asignar modo según Mixture Ratio
    cat.mode = "tracing" if random.random() < MR else "seeking"
    cat.flag = False  # Marca si es el mejor gato
    return cat

def evaluate_cat(cat):
    """Evalúa la función fitness del gato"""
    fitness = h1_function(cat)
    cat.fitness.values = fitness
    return cat

def bound_position(value):
    """Mantiene los valores dentro de los límites del espacio de búsqueda"""
    return max(BOUNDS_MIN, min(BOUNDS_MAX, value))

# ============================================================================
# SEEKING MODE (MODO BÚSQUEDA)
# ============================================================================

def seeking_mode(cat, best_position):
    """
    Seeking Mode: El gato está descansando y observando.
    
    Proceso:
    1. Crear SMP copias del gato actual
    2. Para cada copia, modificar CDC dimensiones aleatoriamente
    3. Calcular fitness de todas las copias
    4. Seleccionar una copia mediante selección probabilística
    5. Actualizar posición del gato con la copia seleccionada
    
    Args:
        cat: El gato en modo seeking
        best_position: Mejor posición global encontrada
    
    Returns:
        Nueva posición del gato
    """
    # Crear copias del gato (Seeking Memory Pool)
    copies = []
    
    for _ in range(SMP):
        # Crear una copia de la posición actual
        copy = list(cat)
        
        # Determinar cuántas dimensiones cambiar
        num_dims_to_change = max(1, int(CDC * DIMENSIONS))
        dims_to_change = random.sample(range(DIMENSIONS), num_dims_to_change)
        
        # Modificar las dimensiones seleccionadas
        for dim in dims_to_change:
            # Calcular el rango de mutación
            mutation_range = SRD * (BOUNDS_MAX - BOUNDS_MIN)
            # Aplicar mutación
            mutation = random.uniform(-mutation_range, mutation_range)
            copy[dim] = bound_position(copy[dim] + mutation)
        
        copies.append(copy)
    
    # Evaluar fitness de todas las copias
    fitness_values = [h1_function(copy)[0] for copy in copies]
    
    # Normalizar fitness para selección probabilística
    # Ajustar para que todos los valores sean positivos
    min_fitness = min(fitness_values)
    if min_fitness < 0:
        fitness_values = [f - min_fitness + 1 for f in fitness_values]
    
    # Calcular probabilidades (proporcionales al fitness)
    total_fitness = sum(fitness_values)
    if total_fitness > 0:
        probabilities = [f / total_fitness for f in fitness_values]
    else:
        probabilities = [1.0 / len(copies) for _ in copies]
    
    # Seleccionar una copia mediante ruleta
    selected_idx = np.random.choice(len(copies), p=probabilities)
    new_position = copies[selected_idx]
    
    return new_position

# ============================================================================
# TRACING MODE (MODO SEGUIMIENTO/CAZA)
# ============================================================================

def tracing_mode(cat, best_position):
    """
    Tracing Mode: El gato está persiguiendo/cazando un objetivo.
    
    Actualización similar a PSO:
    v_new = v_old + c1 * r * (x_best - x_current)
    x_new = x_current + v_new
    
    Args:
        cat: El gato en modo tracing
        best_position: Mejor posición global encontrada
    
    Returns:
        Nueva posición y velocidad del gato
    """
    new_velocity = []
    new_position = []
    
    for dim in range(DIMENSIONS):
        # Actualizar velocidad hacia la mejor posición
        r = random.random()
        v_new = cat.velocity[dim] + C1 * r * (best_position[dim] - cat[dim])
        
        # Limitar velocidad máxima
        v_max = (BOUNDS_MAX - BOUNDS_MIN) * 0.2  # 20% del rango
        v_new = max(-v_max, min(v_max, v_new))
        
        new_velocity.append(v_new)
        
        # Actualizar posición
        x_new = bound_position(cat[dim] + v_new)
        new_position.append(x_new)
    
    return new_position, new_velocity

# ============================================================================
# ALGORITMO PRINCIPAL CSO
# ============================================================================

def cat_swarm_optimization():
    """
    Implementación principal del algoritmo Cat Swarm Optimization.
    
    Returns:
        best_cat: Mejor solución encontrada
        best_fitness_history: Historial del mejor fitness
        mean_fitness_history: Historial del fitness promedio
        positions_history: Posiciones de los gatos para visualización
    """
    # Inicializar población de gatos
    cats = [init_cat() for _ in range(N_CATS)]
    
    # Evaluar población inicial
    for cat in cats:
        evaluate_cat(cat)
    
    # Encontrar el mejor gato inicial
    best_cat = max(cats, key=lambda c: c.fitness.values[0])
    best_position = list(best_cat)
    
    # Estadísticas
    best_fitness_history = [best_cat.fitness.values[0]]
    mean_fitness_history = [np.mean([c.fitness.values[0] for c in cats])]
    positions_history = []
    mode_history = {"seeking": [0], "tracing": [0]}  # Contador de modos
    
    print(f"Generación 0:")
    print(f"  Mejor fitness: {best_cat.fitness.values[0]:.6f}")
    print(f"  Mejor posición: [{best_cat[0]:.4f}, {best_cat[1]:.4f}]")
    print(f"  Fitness promedio: {mean_fitness_history[0]:.6f}")
    print(f"  Gatos en Seeking Mode: {sum(1 for c in cats if c.mode == 'seeking')}")
    print(f"  Gatos en Tracing Mode: {sum(1 for c in cats if c.mode == 'tracing')}")
    
    # Evolución
    for iteration in range(1, N_ITERATIONS + 1):
        # Actualizar cada gato según su modo
        for cat in cats:
            if cat.mode == "seeking":
                # Aplicar Seeking Mode
                new_position = seeking_mode(cat, best_position)
                # Actualizar posición del gato
                for dim in range(DIMENSIONS):
                    cat[dim] = new_position[dim]
                
            else:  # tracing mode
                # Aplicar Tracing Mode
                new_position, new_velocity = tracing_mode(cat, best_position)
                # Actualizar posición y velocidad
                for dim in range(DIMENSIONS):
                    cat[dim] = new_position[dim]
                    cat.velocity[dim] = new_velocity[dim]
            
            # Evaluar nueva posición
            evaluate_cat(cat)
        
        # Actualizar mejor solución global
        current_best = max(cats, key=lambda c: c.fitness.values[0])
        if current_best.fitness.values[0] > best_cat.fitness.values[0]:
            best_cat = creator.Cat(current_best)
            best_cat.fitness.values = current_best.fitness.values
            best_cat.velocity = list(current_best.velocity)
            best_position = list(best_cat)
        
        # Re-asignar modos aleatoriamente (mantener MR ratio)
        # Seleccionar cuáles gatos estarán en tracing mode
        num_tracing = int(N_CATS * MR)
        indices = list(range(N_CATS))
        random.shuffle(indices)
        
        for i, cat in enumerate(cats):
            cat.mode = "tracing" if i in indices[:num_tracing] else "seeking"
        
        # Registrar estadísticas
        best_fitness_history.append(best_cat.fitness.values[0])
        mean_fitness = np.mean([c.fitness.values[0] for c in cats])
        mean_fitness_history.append(mean_fitness)
        
        # Contar modos
        seeking_count = sum(1 for c in cats if c.mode == "seeking")
        tracing_count = N_CATS - seeking_count
        mode_history["seeking"].append(seeking_count)
        mode_history["tracing"].append(tracing_count)
        
        # Guardar posiciones para visualización (cada 10 iteraciones)
        if iteration % 10 == 0:
            positions_history.append([(c[0], c[1]) for c in cats])
        
        # Mostrar progreso
        if iteration % 20 == 0:
            print(f"\nGeneración {iteration}:")
            print(f"  Mejor fitness: {best_cat.fitness.values[0]:.6f}")
            print(f"  Mejor posición: [{best_cat[0]:.4f}, {best_cat[1]:.4f}]")
            print(f"  Fitness promedio: {mean_fitness:.6f}")
            print(f"  Seeking/Tracing: {seeking_count}/{tracing_count}")
    
    return best_cat, best_fitness_history, mean_fitness_history, positions_history, mode_history

# ============================================================================
# EJECUCIÓN Y RESULTADOS
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("CAT SWARM OPTIMIZATION (CSO) - Función Benchmark H1")
    print("="*70)
    print(f"Parámetros del algoritmo:")
    print(f"  Número de gatos: {N_CATS}")
    print(f"  Iteraciones: {N_ITERATIONS}")
    print(f"  Mixture Ratio (MR): {MR:.2%} en Tracing Mode")
    print(f"  Seeking Memory Pool (SMP): {SMP}")
    print(f"  Seeking Range Dimension (SRD): {SRD:.2%}")
    print(f"  Counts Dimension to Change (CDC): {CDC:.2%}")
    print(f"  Constante C1 (Tracing): {C1}")
    print(f"  Dimensiones: {DIMENSIONS}")
    print(f"  Límites: [{BOUNDS_MIN}, {BOUNDS_MAX}]")
    print("="*70)
    print()
    
    # Ejecutar algoritmo
    best_solution, best_history, mean_history, positions_hist, mode_hist = cat_swarm_optimization()
    
    # ========================================================================
    # RESULTADOS FINALES
    # ========================================================================
    
    print("\n" + "="*70)
    print("RESULTADOS FINALES")
    print("="*70)
    print(f"Mejor fitness encontrado: {best_solution.fitness.values[0]:.6f}")
    print(f"Mejor posición: [{best_solution[0]:.6f}, {best_solution[1]:.6f}]")
    print("="*70)
    
    # ========================================================================
    # VISUALIZACIONES
    # ========================================================================
    
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Gráfico de convergencia
    ax1 = fig.add_subplot(231)
    generations = range(len(best_history))
    ax1.plot(generations, best_history, 'b-', linewidth=2, label='Mejor Fitness')
    ax1.plot(generations, mean_history, 'r--', linewidth=1.5, label='Fitness Promedio')
    ax1.set_xlabel('Generación', fontsize=11)
    ax1.set_ylabel('Fitness', fontsize=11)
    ax1.set_title('Convergencia del Algoritmo CSO', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribución de modos
    ax2 = fig.add_subplot(232)
    ax2.plot(generations, mode_hist["seeking"], 'g-', linewidth=2, label='Seeking Mode')
    ax2.plot(generations, mode_hist["tracing"], 'm-', linewidth=2, label='Tracing Mode')
    ax2.axhline(y=N_CATS*MR, color='m', linestyle='--', alpha=0.5, label=f'MR objetivo ({MR:.0%})')
    ax2.set_xlabel('Generación', fontsize=11)
    ax2.set_ylabel('Número de Gatos', fontsize=11)
    ax2.set_title('Distribución de Modos CSO', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Superficie 3D de la función H1
    ax3 = fig.add_subplot(233, projection='3d')
    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = benchmarks.h1([X[i, j], Y[i, j]])[0]
    
    ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
    ax3.scatter([best_solution[0]], [best_solution[1]], [best_solution.fitness.values[0]], 
               c='red', s=150, marker='*', label='Mejor Solución')
    ax3.set_xlabel('X', fontsize=10)
    ax3.set_ylabel('Y', fontsize=10)
    ax3.set_zlabel('H1(X, Y)', fontsize=10)
    ax3.set_title('Función H1 con Mejor Solución', fontsize=12, fontweight='bold')
    ax3.legend()
    
    # 4. Contorno con trayectorias
    ax4 = fig.add_subplot(234)
    contour = ax4.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax4.clabel(contour, inline=True, fontsize=8)
    
    # Dibujar trayectorias de algunos gatos
    if positions_hist:
        for cat_idx in range(min(5, N_CATS)):
            x_traj = [positions_hist[i][cat_idx][0] for i in range(len(positions_hist))]
            y_traj = [positions_hist[i][cat_idx][1] for i in range(len(positions_hist))]
            ax4.plot(x_traj, y_traj, 'o-', alpha=0.5, markersize=3, linewidth=1)
    
    ax4.scatter([best_solution[0]], [best_solution[1]], 
               c='red', s=200, marker='*', edgecolors='black', linewidth=2,
               label='Mejor Solución', zorder=5)
    ax4.set_xlabel('X', fontsize=11)
    ax4.set_ylabel('Y', fontsize=11)
    ax4.set_title('Contorno H1 con Trayectorias de Gatos', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(BOUNDS_MIN, BOUNDS_MAX)
    ax4.set_ylim(BOUNDS_MIN, BOUNDS_MAX)
    
    # 5. Histograma de fitness final
    ax5 = fig.add_subplot(235)
    final_fitness = [best_history[-1], mean_history[-1]]
    labels = ['Mejor', 'Promedio']
    colors = ['blue', 'red']
    ax5.bar(labels, final_fitness, color=colors, alpha=0.7)
    ax5.set_ylabel('Fitness', fontsize=11)
    ax5.set_title('Fitness Final', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Comparación con valor óptimo conocido
    ax6 = fig.add_subplot(236)
    # El valor óptimo de H1 es aproximadamente 2.0
    optimal_value = 2.0
    error = abs(best_solution.fitness.values[0] - optimal_value)
    ax6.barh(['Mejor Encontrado', 'Óptimo Conocido', 'Error'], 
             [best_solution.fitness.values[0], optimal_value, error],
             color=['green', 'blue', 'orange'], alpha=0.7)
    ax6.set_xlabel('Valor', fontsize=11)
    ax6.set_title('Comparación con Óptimo', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('/home/juanpe/master/mlmuia/oca/codigos_clase6/CSO_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Visualizaciones generadas y guardadas correctamente.")
    print("✓ Archivo guardado: CSO_results.png")
    print("\n" + "="*70)
    print("ANÁLISIS DEL ALGORITMO CSO")
    print("="*70)
    print("\nCaracterísticas principales:")
    print("1. Seeking Mode: Exploración local mediante copias y mutaciones")
    print("2. Tracing Mode: Explotación dirigida hacia la mejor solución")
    print(f"3. Balance: {(1-MR)*100:.0f}% exploración, {MR*100:.0f}% explotación")
    print("\nVentajas observadas:")
    print("- Buena capacidad de exploración global")
    print("- Convergencia estable hacia el óptimo")
    print("- Balance automático entre exploración y explotación")
    print("="*70)

# %%
