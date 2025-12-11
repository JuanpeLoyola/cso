# -*- coding: utf-8 -*-
"""
Comparación entre algoritmos de optimización bio-inspirados
CSO vs PSO vs Firefly

Este script compara el rendimiento de tres algoritmos:
1. Cat Swarm Optimization (CSO)
2. Particle Swarm Optimization (PSO)
3. Firefly Algorithm

Todos optimizan la misma función (Shekel) con parámetros comparables.
"""

#%%
import random
import numpy as np
from deap import base, creator, tools, benchmarks
import matplotlib.pyplot as plt
import time

# Semilla para reproducibilidad
random.seed(42)
np.random.seed(42)

# ============================================================================
# FUNCIÓN SHEKEL
# ============================================================================

def shekel_three_maxima_2d(individual):
    """
    Función Shekel con tres máximos locales en 2 dimensiones.
    
    Para MAXIMIZACIÓN, invertimos el signo.
    """
    # Definir parámetros para 3 máximos en 2D
    A = [
        [4.0, 4.0],   # Centro del máximo 1
        [1.0, 1.0],   # Centro del máximo 2
        [8.0, 8.0],   # Centro del máximo 3
    ]
    
    # Vector C: parámetros que controlan altura/agudeza
    C = [0.1, 0.2, 0.2]
    
    # Usar la función shekel de DEAP con nuestros parámetros
    # Para MAXIMIZACIÓN, invertimos el signo
    result = -benchmarks.shekel(individual, A, C)[0]
    
    return (result,)

# ============================================================================
# PARÁMETROS COMUNES
# ============================================================================

N_AGENTS = 50           # Número de agentes (gatos/partículas/luciérnagas)
N_ITERATIONS = 100      # Número de iteraciones
DIMENSIONS = 2          # Dimensiones del problema
BOUNDS_MIN = 0.0        # Ajustado para Shekel
BOUNDS_MAX = 10.0       # Ajustado para Shekel
N_RUNS = 5              # Número de ejecuciones para promediar

# ============================================================================
# CONFIGURACIÓN DEAP
# ============================================================================

if not hasattr(creator, 'FitnessMax'):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, 'Agent'):
    creator.create("Agent", list, fitness=creator.FitnessMax, velocity=list)

def bound_position(value):
    """Mantiene valores dentro de límites"""
    return max(BOUNDS_MIN, min(BOUNDS_MAX, value))

# ============================================================================
# IMPLEMENTACIÓN CSO SIMPLIFICADA
# ============================================================================

def cso_algorithm(verbose=False):
    """Cat Swarm Optimization"""
    MR = 0.20
    SMP = 5
    SRD = 0.2
    CDC = 0.8
    C1 = 2.0
    
    # Inicializar población
    cats = []
    for _ in range(N_AGENTS):
        cat = creator.Agent([random.uniform(BOUNDS_MIN, BOUNDS_MAX) for _ in range(DIMENSIONS)])
        cat.velocity = [random.uniform(-1, 1) for _ in range(DIMENSIONS)]
        cat.fitness.values = shekel_three_maxima_2d(cat)
        cats.append(cat)
    
    best_cat = max(cats, key=lambda c: c.fitness.values[0])
    best_position = list(best_cat)
    history = [best_cat.fitness.values[0]]
    
    for iteration in range(N_ITERATIONS):
        # Asignar modos
        num_tracing = int(N_AGENTS * MR)
        indices = list(range(N_AGENTS))
        random.shuffle(indices)
        modes = ['tracing' if i in indices[:num_tracing] else 'seeking' for i in range(N_AGENTS)]
        
        for i, cat in enumerate(cats):
            if modes[i] == "seeking":
                # Seeking Mode
                copies = []
                for _ in range(SMP):
                    copy = list(cat)
                    num_dims = max(1, int(CDC * DIMENSIONS))
                    dims = random.sample(range(DIMENSIONS), num_dims)
                    for dim in dims:
                        mutation = random.uniform(-SRD * (BOUNDS_MAX - BOUNDS_MIN), 
                                                 SRD * (BOUNDS_MAX - BOUNDS_MIN))
                        copy[dim] = bound_position(copy[dim] + mutation)
                    copies.append(copy)
                
                fitness_values = [shekel_three_maxima_2d(c)[0] for c in copies]
                min_fit = min(fitness_values)
                if min_fit < 0:
                    fitness_values = [f - min_fit + 1 for f in fitness_values]
                
                total = sum(fitness_values)
                if total > 0:
                    probs = [f / total for f in fitness_values]
                else:
                    probs = [1.0 / len(copies)] * len(copies)
                
                selected = np.random.choice(len(copies), p=probs)
                for dim in range(DIMENSIONS):
                    cat[dim] = copies[selected][dim]
            
            else:  # Tracing Mode
                for dim in range(DIMENSIONS):
                    r = random.random()
                    cat.velocity[dim] = cat.velocity[dim] + C1 * r * (best_position[dim] - cat[dim])
                    v_max = (BOUNDS_MAX - BOUNDS_MIN) * 0.2
                    cat.velocity[dim] = max(-v_max, min(v_max, cat.velocity[dim]))
                    cat[dim] = bound_position(cat[dim] + cat.velocity[dim])
            
            cat.fitness.values = shekel_three_maxima_2d(cat)
        
        current_best = max(cats, key=lambda c: c.fitness.values[0])
        if current_best.fitness.values[0] > best_cat.fitness.values[0]:
            best_cat = creator.Agent(current_best)
            best_cat.fitness.values = current_best.fitness.values
            best_position = list(best_cat)
        
        history.append(best_cat.fitness.values[0])
        
        if verbose and (iteration + 1) % 20 == 0:
            print(f"  Iteración {iteration+1}: Mejor = {best_cat.fitness.values[0]:.6f}")
    
    return best_cat.fitness.values[0], history

# ============================================================================
# IMPLEMENTACIÓN PSO SIMPLIFICADA
# ============================================================================

def pso_algorithm(verbose=False):
    """Particle Swarm Optimization"""
    W = 0.7   # Inercia
    C1 = 1.5  # Cognitivo
    C2 = 1.5  # Social
    
    # Inicializar población
    particles = []
    for _ in range(N_AGENTS):
        particle = creator.Agent([random.uniform(BOUNDS_MIN, BOUNDS_MAX) for _ in range(DIMENSIONS)])
        particle.velocity = [random.uniform(-1, 1) for _ in range(DIMENSIONS)]
        particle.fitness.values = shekel_three_maxima_2d(particle)
        particle.best = list(particle)
        particle.best_fitness = particle.fitness.values[0]
        particles.append(particle)
    
    best_particle = max(particles, key=lambda p: p.fitness.values[0])
    global_best = list(best_particle)
    history = [best_particle.fitness.values[0]]
    
    for iteration in range(N_ITERATIONS):
        for particle in particles:
            for dim in range(DIMENSIONS):
                r1, r2 = random.random(), random.random()
                cognitive = C1 * r1 * (particle.best[dim] - particle[dim])
                social = C2 * r2 * (global_best[dim] - particle[dim])
                particle.velocity[dim] = W * particle.velocity[dim] + cognitive + social
                
                v_max = (BOUNDS_MAX - BOUNDS_MIN) * 0.2
                particle.velocity[dim] = max(-v_max, min(v_max, particle.velocity[dim]))
                particle[dim] = bound_position(particle[dim] + particle.velocity[dim])
            
            particle.fitness.values = shekel_three_maxima_2d(particle)
            
            if particle.fitness.values[0] > particle.best_fitness:
                particle.best = list(particle)
                particle.best_fitness = particle.fitness.values[0]
        
        current_best = max(particles, key=lambda p: p.fitness.values[0])
        if current_best.fitness.values[0] > history[-1]:
            global_best = list(current_best.best)
        
        history.append(max(p.fitness.values[0] for p in particles))
        
        if verbose and (iteration + 1) % 20 == 0:
            print(f"  Iteración {iteration+1}: Mejor = {history[-1]:.6f}")
    
    return history[-1], history

# ============================================================================
# IMPLEMENTACIÓN FIREFLY SIMPLIFICADA
# ============================================================================

def firefly_algorithm(verbose=False):
    """Firefly Algorithm"""
    ALPHA = 0.2
    BETA_0 = 1.0
    GAMMA = 0.005
    
    # Inicializar población
    fireflies = []
    for _ in range(N_AGENTS):
        firefly = creator.Agent([random.uniform(BOUNDS_MIN, BOUNDS_MAX) for _ in range(DIMENSIONS)])
        firefly.fitness.values = shekel_three_maxima_2d(firefly)
        fireflies.append(firefly)
    
    best_firefly = max(fireflies, key=lambda f: f.fitness.values[0])
    history = [best_firefly.fitness.values[0]]
    
    for iteration in range(N_ITERATIONS):
        fireflies.sort(key=lambda f: f.fitness.values[0])
        
        new_fireflies = []
        for i in range(N_AGENTS):
            moved = False
            new_firefly = creator.Agent(fireflies[i])
            
            for j in range(i + 1, N_AGENTS):
                if fireflies[j].fitness.values[0] > fireflies[i].fitness.values[0]:
                    distance = np.sqrt(sum((fireflies[i][k] - fireflies[j][k])**2 
                                         for k in range(DIMENSIONS)))
                    beta = BETA_0 * np.exp(-GAMMA * distance**2)
                    
                    for k in range(DIMENSIONS):
                        attraction = beta * (fireflies[j][k] - fireflies[i][k])
                        randomization = ALPHA * (random.random() - 0.5) * (BOUNDS_MAX - BOUNDS_MIN)
                        new_firefly[k] = bound_position(fireflies[i][k] + attraction + randomization)
                    moved = True
                    break
            
            if not moved:
                for k in range(DIMENSIONS):
                    new_firefly[k] = bound_position(fireflies[i][k] + 
                                                   ALPHA * (random.random() - 0.5) * (BOUNDS_MAX - BOUNDS_MIN))
            
            new_firefly.fitness.values = shekel_three_maxima_2d(new_firefly)
            new_fireflies.append(new_firefly)
        
        fireflies = new_fireflies
        
        current_best = max(fireflies, key=lambda f: f.fitness.values[0])
        if current_best.fitness.values[0] > best_firefly.fitness.values[0]:
            best_firefly = creator.Agent(current_best)
            best_firefly.fitness.values = current_best.fitness.values
        
        history.append(best_firefly.fitness.values[0])
        
        if verbose and (iteration + 1) % 20 == 0:
            print(f"  Iteración {iteration+1}: Mejor = {best_firefly.fitness.values[0]:.6f}")
    
    return best_firefly.fitness.values[0], history

# ============================================================================
# EXPERIMENTOS Y COMPARACIÓN
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("COMPARACIÓN DE ALGORITMOS BIO-INSPIRADOS")
    print("="*70)
    print(f"Función objetivo: Shekel (3 máximos)")
    print(f"Agentes/Población: {N_AGENTS}")
    print(f"Iteraciones: {N_ITERATIONS}")
    print(f"Ejecuciones por algoritmo: {N_RUNS}")
    print(f"Dimensiones: {DIMENSIONS}")
    print(f"Límites: [{BOUNDS_MIN}, {BOUNDS_MAX}]")
    print("="*70)
    print()
    
    algorithms = {
        'CSO': cso_algorithm,
        'PSO': pso_algorithm,
        'Firefly': firefly_algorithm
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"\n{'='*70}")
        print(f"Ejecutando {name}...")
        print('='*70)
        
        best_values = []
        all_histories = []
        times = []
        
        for run in range(N_RUNS):
            print(f"\nEjecución {run + 1}/{N_RUNS}")
            start_time = time.time()
            best_val, history = algorithm(verbose=True)
            elapsed = time.time() - start_time
            
            best_values.append(best_val)
            all_histories.append(history)
            times.append(elapsed)
            
            print(f"  Mejor fitness: {best_val:.6f}")
            print(f"  Tiempo: {elapsed:.3f}s")
        
        results[name] = {
            'best_values': best_values,
            'histories': all_histories,
            'times': times,
            'mean_best': np.mean(best_values),
            'std_best': np.std(best_values),
            'mean_time': np.mean(times)
        }
    
    # ========================================================================
    # RESULTADOS
    # ========================================================================
    
    print("\n" + "="*70)
    print("RESUMEN DE RESULTADOS")
    print("="*70)
    print(f"\n{'Algoritmo':<15} {'Media':<12} {'Std':<12} {'Mejor':<12} {'Tiempo (s)':<12}")
    print("-"*70)
    
    for name in algorithms.keys():
        res = results[name]
        print(f"{name:<15} {res['mean_best']:<12.6f} {res['std_best']:<12.6f} "
              f"{max(res['best_values']):<12.6f} {res['mean_time']:<12.3f}")
    
    print("="*70)
    
    # Determinar el mejor algoritmo
    best_algorithm = max(results.items(), key=lambda x: x[1]['mean_best'])
    print(f"\n🏆 Mejor algoritmo: {best_algorithm[0]}")
    print(f"   Media de fitness: {best_algorithm[1]['mean_best']:.6f}")
    print(f"   Desviación estándar: {best_algorithm[1]['std_best']:.6f}")
    
    # ========================================================================
    # VISUALIZACIÓN SIMPLE: SOLO CONVERGENCIA
    # ========================================================================
    
    plt.figure(figsize=(10, 6))
    
    colors = {'CSO': 'blue', 'PSO': 'green', 'Firefly': 'red'}
    
    for name in algorithms.keys():
        histories = results[name]['histories']
        mean_history = np.mean(histories, axis=0)
        std_history = np.std(histories, axis=0)
        generations = range(len(mean_history))
        
        plt.plot(generations, mean_history, color=colors[name], 
                linewidth=2.5, label=f'{name}', marker='o', markersize=4, 
                markevery=10)
        plt.fill_between(generations, 
                         mean_history - std_history,
                         mean_history + std_history,
                         color=colors[name], alpha=0.15)
    
    plt.xlabel('Iteración', fontsize=13, fontweight='bold')
    plt.ylabel('Fitness', fontsize=13, fontweight='bold')
    plt.title('Comparación de Convergencia: CSO vs PSO vs Firefly\nFunción Shekel', 
              fontsize=14, fontweight='bold', pad=15)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('algorithms_comparison_shekel.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Comparación completada.")
    print("✓ Gráfico guardado en: algorithms_comparison_shekel.png")
