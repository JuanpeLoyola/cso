"""
Comparación Completa de Algoritmos de Enjambre con DEAP
CSO vs PSO vs ACO vs Firefly

Incluye:
1. Cat Swarm Optimization (CSO)
2. Particle Swarm Optimization (PSO)
3. Ant Colony Optimization (ACO)
4. Firefly Algorithm (FA)
"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, benchmarks

# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_AGENTS = 30           # Población
N_ITERATIONS = 100      # Iteraciones
DIMENSIONS = 2          # Dimensiones (H1 es 2D)

# =============================================================================
# CONFIGURACIÓN DEAP
# =============================================================================

# Definimos objetivos: Minimización (-) y Maximización (+)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) 
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

toolbox = base.Toolbox()

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def bound_position(pos, min_val, max_val):
    return max(min_val, min(max_val, pos))

def get_evaluate_func(func_name):
    if func_name == "rosenbrock":
        return benchmarks.rosenbrock
    else:
        return benchmarks.h1

# =============================================================================
# 1. ALGORITMO CSO (CAT SWARM OPTIMIZATION)
# =============================================================================

def run_cso(func_name, bounds_min, bounds_max, problem_type="min"):
    """
    Implementa Cat Swarm Optimization
    Los gatos alternan entre dos modos: Seeking (exploración) y Tracing (explotación)
    """
    # Parámetros CSO
    MR = 0.2    # Mixture Ratio: 20% de gatos en modo Tracing (cazando)
    SMP = 5     # Seeking Memory Pool: número de copias a generar por gato en Seeking
    SRD = 0.2   # Seeking Range of selected Dimension: rango de mutación (20% del espacio)
    CDC = 0.8   # Counts of Dimension to Change: probabilidad de mutar dimensión (80%)
    C1 = 2.0    # Constante de velocidad para Tracing Mode (similar a PSO)
    
    # Seleccionar clase de fitness según si es minimización o maximización
    FitnessClass = creator.FitnessMin if problem_type == "min" else creator.FitnessMax

    # Eliminar clase anterior si existe (para evitar conflictos en múltiples ejecuciones)
    if "AgentCSO" in dir(creator):
        del creator.AgentCSO
    # Crear clase de agente con posición, fitness, velocidad y modo
    creator.create("AgentCSO", list, fitness=FitnessClass, velocity=list, mode=str)
    
    # Obtener función de evaluación (rosenbrock o h1)
    evaluate = get_evaluate_func(func_name)

    # =========================================================================
    # INICIALIZACIÓN DE LA POBLACIÓN
    # =========================================================================
    cats = []  # Lista para almacenar todos los gatos
    for _ in range(N_AGENTS):  # Crear N_AGENTS gatos
        # Crear gato con posición aleatoria en cada dimensión
        cat = creator.AgentCSO(random.uniform(bounds_min, bounds_max) for _ in range(DIMENSIONS))
        # Inicializar velocidad aleatoria entre -1 y 1
        cat.velocity = [random.uniform(-1, 1) for _ in range(DIMENSIONS)]
        # Inicialmente todos están en modo "seeking" (descansando/observando)
        cat.mode = "seeking"
        # Evaluar fitness del gato en su posición inicial
        cat.fitness.values = evaluate(cat)
        # Añadir gato a la población
        cats.append(cat)
    
    # Encontrar el mejor gato inicial (max para maximización, min para minimización)
    best_cat = (max(cats, key=lambda c: c.fitness.values[0]) if problem_type == "max" 
                else min(cats, key=lambda c: c.fitness.values[0]))
    # Guardar posición del mejor gato
    best_pos = list(best_cat)
    # Historial de convergencia (guarda el mejor fitness por iteración)
    history = [best_cat.fitness.values[0]]
    
    # =========================================================================
    # BUCLE PRINCIPAL DE ITERACIONES
    # =========================================================================
    for _ in range(N_ITERATIONS):
        # =====================================================================
        # ASIGNACIÓN DE MODOS (Seeking vs Tracing)
        # =====================================================================
        # Calcular cuántos gatos estarán en modo Tracing (MR = 20%)
        num_tracing = int(N_AGENTS * MR)
        # Crear lista de índices y mezclarlos aleatoriamente
        indices = list(range(N_AGENTS))
        random.shuffle(indices)
        # Asignar modos: primeros num_tracing → tracing, resto → seeking
        for i, idx in enumerate(indices):
            cats[idx].mode = "tracing" if i < num_tracing else "seeking"
        
        # =====================================================================
        # ACTUALIZACIÓN DE CADA GATO SEGÚN SU MODO
        # =====================================================================
        for cat in cats:
            # =================================================================
            # SEEKING MODE (Modo descanso/observación - EXPLORACIÓN)
            # =================================================================
            if cat.mode == "seeking":
                copies = []  # Lista para almacenar copias del gato
                
                # Crear SMP copias del gato actual
                for _ in range(SMP):
                    # Crear copia del gato
                    copy_cat = creator.AgentCSO(cat)
                    
                    # Con probabilidad CDC, mutar las dimensiones
                    if random.random() < CDC:
                        for d in range(DIMENSIONS):
                            # Calcular mutación: SRD * rango_total * valor_aleatorio_[-1,1]
                            mutation = SRD * (bounds_max - bounds_min) * (random.random() - 0.5) * 2
                            # Aplicar mutación y asegurar que esté dentro de los límites
                            copy_cat[d] = bound_position(copy_cat[d] + mutation, bounds_min, bounds_max)
                    
                    # Añadir copia a la lista
                    copies.append(copy_cat)
                
                # Evaluar fitness de todas las copias
                fits = [evaluate(c)[0] for c in copies]
                # Seleccionar la mejor copia (max para maximizar, min para minimizar)
                best_idx = np.argmax(fits) if problem_type == "max" else np.argmin(fits)
                
                # Actualizar posición del gato con la mejor copia
                cat[:] = copies[best_idx][:]
                # Actualizar fitness del gato
                cat.fitness.values = (fits[best_idx],)

            # =================================================================
            # TRACING MODE (Modo caza - EXPLOTACIÓN)
            # =================================================================
            else:
                # Actualizar cada dimensión usando ecuación de velocidad
                for d in range(DIMENSIONS):
                    # Número aleatorio para componente estocástica
                    r = random.random()
                    # Actualizar velocidad: v = v + C1 * r * (mejor_posición - posición_actual)
                    # Similar a PSO pero solo con componente global
                    cat.velocity[d] += C1 * r * (best_pos[d] - cat[d])
                    # Limitar velocidad al 10% del rango del espacio de búsqueda
                    v_limit = (bounds_max - bounds_min) * 0.1
                    cat.velocity[d] = max(-v_limit, min(v_limit, cat.velocity[d]))
                    # Actualizar posición: x = x + v
                    cat[d] = bound_position(cat[d] + cat.velocity[d], bounds_min, bounds_max)
                
                # Evaluar fitness en la nueva posición
                cat.fitness.values = evaluate(cat)

            # =================================================================
            # ACTUALIZACIÓN DEL MEJOR GLOBAL
            # =================================================================
            # Verificar si el gato actual es mejor que el mejor global
            better = (cat.fitness.values[0] > best_cat.fitness.values[0]) if problem_type == "max" \
                     else (cat.fitness.values[0] < best_cat.fitness.values[0])
            
            # Si es mejor, actualizar el mejor global
            if better:
                best_cat = creator.AgentCSO(cat)    # Crear copia del gato
                best_cat.fitness.values = cat.fitness.values  # Copiar fitness
                best_pos = list(cat)  # Guardar posición del mejor
        
        # Guardar mejor fitness de esta iteración en el historial
        history.append(best_cat.fitness.values[0])
    
    # Retornar historial de convergencia
    return history

# =============================================================================
# 2. ALGORITMO PSO (PARTICLE SWARM)
# =============================================================================

def run_pso(func_name, bounds_min, bounds_max, problem_type="min"):
    W, C1, C2 = 0.7, 1.5, 1.5
    FitnessClass = creator.FitnessMin if problem_type == "min" else creator.FitnessMax
    creator.create("AgentPSO", list, fitness=FitnessClass, velocity=list, best=list, best_fit=float)
    evaluate = get_evaluate_func(func_name)

    pop = []
    for _ in range(N_AGENTS):
        p = creator.AgentPSO(random.uniform(bounds_min, bounds_max) for _ in range(DIMENSIONS))
        p.velocity = [random.uniform(-1, 1) for _ in range(DIMENSIONS)]
        p.fitness.values = evaluate(p)
        p.best = list(p)
        p.best_fit = p.fitness.values[0]
        pop.append(p)
        
    g_best = (max(pop, key=lambda x: x.best_fit) if problem_type == "max" 
              else min(pop, key=lambda x: x.best_fit))
    g_best_pos = list(g_best.best)
    g_best_fit = g_best.best_fit
    history = [g_best_fit]
    
    for _ in range(N_ITERATIONS):
        for part in pop:
            for d in range(DIMENSIONS):
                r1, r2 = random.random(), random.random()
                v_cog = C1 * r1 * (part.best[d] - part[d])
                v_soc = C2 * r2 * (g_best_pos[d] - part[d])
                part.velocity[d] = W * part.velocity[d] + v_cog + v_soc
                part[d] = bound_position(part[d] + part.velocity[d], bounds_min, bounds_max)
            
            part.fitness.values = evaluate(part)
            fit = part.fitness.values[0]
            
            better_p = (fit > part.best_fit) if problem_type == "max" else (fit < part.best_fit)
            if better_p:
                part.best_fit = fit
                part.best = list(part)
                better_g = (fit > g_best_fit) if problem_type == "max" else (fit < g_best_fit)
                if better_g:
                    g_best_fit = fit
                    g_best_pos = list(part)
        history.append(g_best_fit)
    return history

# =============================================================================
# 3. ALGORITMO ACO (ANT COLONY)
# =============================================================================

def run_aco(func_name, bounds_min, bounds_max, problem_type="min"):
    ARCHIVE_SIZE, Q, ZETA = 10, 0.5, 1.0
    FitnessClass = creator.FitnessMin if problem_type == "min" else creator.FitnessMax
    creator.create("AgentACO", list, fitness=FitnessClass)
    evaluate = get_evaluate_func(func_name)
    
    archive = []
    for _ in range(N_AGENTS):
        ant = creator.AgentACO(random.uniform(bounds_min, bounds_max) for _ in range(DIMENSIONS))
        ant.fitness.values = evaluate(ant)
        archive.append(ant)
        
    archive.sort(key=lambda x: x.fitness.values[0], reverse=(problem_type == "max"))
    archive = archive[:ARCHIVE_SIZE]
    history = [archive[0].fitness.values[0]]
    
    for _ in range(N_ITERATIONS):
        new_pop = []
        weights = [1/(ZETA*np.sqrt(2*np.pi))*np.exp(-0.5*((r-1)/(ZETA*ARCHIVE_SIZE))**2) 
                   for r in range(1, ARCHIVE_SIZE+1)]
        total_w = sum(weights)
        probs = [w/total_w for w in weights]
        
        for _ in range(N_AGENTS):
            idx = np.random.choice(range(ARCHIVE_SIZE), p=probs)
            guide = archive[idx]
            sigma = [np.std([ind[d] for ind in archive])*Q for d in range(DIMENSIONS)]
            
            new_ant = creator.AgentACO()
            for d in range(DIMENSIONS):
                val = np.random.normal(guide[d], sigma[d])
                new_ant.append(bound_position(val, bounds_min, bounds_max))
            new_ant.fitness.values = evaluate(new_ant)
            new_pop.append(new_ant)
            
        total = archive + new_pop
        total.sort(key=lambda x: x.fitness.values[0], reverse=(problem_type == "max"))
        archive = total[:ARCHIVE_SIZE]
        history.append(archive[0].fitness.values[0])
    return history

# =============================================================================
# 4. ALGORITMO FIREFLY (FA)
# =============================================================================

def run_firefly(func_name, bounds_min, bounds_max, problem_type="min"):

    ALPHA = 0.2     # Aleatoriedad
    BETA_0 = 1.0    # Atractivo base
    GAMMA = 0.005   # Absorción de luz
    
    FitnessClass = creator.FitnessMin if problem_type == "min" else creator.FitnessMax
    creator.create("AgentFirefly", list, fitness=FitnessClass)
    evaluate = get_evaluate_func(func_name)
    
    # Inicialización
    fireflies = []
    for _ in range(N_AGENTS):
        ff = creator.AgentFirefly(random.uniform(bounds_min, bounds_max) for _ in range(DIMENSIONS))
        ff.fitness.values = evaluate(ff)
        fireflies.append(ff)
        
    # Mejor inicial
    best_ff = (max(fireflies, key=lambda x: x.fitness.values[0]) if problem_type == "max" 
               else min(fireflies, key=lambda x: x.fitness.values[0]))
    history = [best_ff.fitness.values[0]]
    
    for _ in range(N_ITERATIONS):
    
        fireflies.sort(key=lambda x: x.fitness.values[0], reverse=(problem_type == "max"))
        
        for i in range(N_AGENTS):
            moved = False
            for j in range(N_AGENTS):
                # Determinar si j es "más brillante" (mejor fitness) que i
                # Si es Max: fit_j > fit_i. Si es Min: fit_j < fit_i
                is_brighter = (fireflies[j].fitness.values[0] > fireflies[i].fitness.values[0]) if problem_type == "max" \
                              else (fireflies[j].fitness.values[0] < fireflies[i].fitness.values[0])
                
                if is_brighter:
                    # Calcular distancia Euclidiana
                    dist = math.sqrt(sum((fireflies[i][d] - fireflies[j][d])**2 for d in range(DIMENSIONS)))
                    
                    # Calcular Atractivo: beta = beta0 * exp(-gamma * r^2)
                    beta = BETA_0 * math.exp(-GAMMA * dist**2)
                    
                    # Mover luciérnaga i hacia j
                    for d in range(DIMENSIONS):
                        attraction = beta * (fireflies[j][d] - fireflies[i][d])
                        randomness = ALPHA * (random.random() - 0.5) * (bounds_max - bounds_min)
                        
                        new_val = fireflies[i][d] + attraction + randomness
                        fireflies[i][d] = bound_position(new_val, bounds_min, bounds_max)
                    
                    moved = True
            
            # Si no se sintió atraída por nadie, movimiento aleatorio (Random Walk)
            if not moved:
                for d in range(DIMENSIONS):
                    rand_step = ALPHA * (random.random() - 0.5) * (bounds_max - bounds_min)
                    fireflies[i][d] = bound_position(fireflies[i][d] + rand_step, bounds_min, bounds_max)
            
            # Evaluar nueva posición
            fireflies[i].fitness.values = evaluate(fireflies[i])
            
        # Actualizar mejor global
        current_best = (max(fireflies, key=lambda x: x.fitness.values[0]) if problem_type == "max" 
                        else min(fireflies, key=lambda x: x.fitness.values[0]))
        
        # Comparar con el histórico
        is_new_record = (current_best.fitness.values[0] > best_ff.fitness.values[0]) if problem_type == "max" \
                        else (current_best.fitness.values[0] < best_ff.fitness.values[0])
                        
        if is_new_record:
            best_ff = creator.AgentFirefly(current_best)
            best_ff.fitness.values = current_best.fitness.values
            
        history.append(best_ff.fitness.values[0])
        
    return history

# =============================================================================
# EJECUCIÓN Y COMPARATIVA
# =============================================================================

print("="*70)
print("COMPARATIVA: CSO vs PSO vs ACO vs FIREFLY")
print("="*70)

# --- 1. ROSENBROCK (Minimización) ---
print("\n[1/2] Benchmark Rosenbrock (Minimización)")
h_cso_ros = run_cso("rosenbrock", -30, 30, "min")
h_pso_ros = run_pso("rosenbrock", -30, 30, "min")
h_aco_ros = run_aco("rosenbrock", -30, 30, "min")
h_fir_ros = run_firefly("rosenbrock", -30, 30, "min")

# --- 2. H1 (Maximización) ---
print("[2/2] Benchmark H1 (Maximización)")
h_cso_h1 = run_cso("h1", -100, 100, "max")
h_pso_h1 = run_pso("h1", -100, 100, "max")
h_aco_h1 = run_aco("h1", -100, 100, "max")
h_fir_h1 = run_firefly("h1", -100, 100, "max")

# =============================================================================
# GRÁFICOS
# =============================================================================

# Gráfica Rosenbrock
plt.figure(figsize=(10, 6))
plt.plot(h_cso_ros, label='CSO (Gatos)', linewidth=2)
plt.plot(h_pso_ros, label='PSO (Partículas)', linestyle='--')
plt.plot(h_aco_ros, label='ACO (Hormigas)', linestyle=':')
plt.plot(h_fir_ros, label='Firefly (Luciérnagas)', linestyle='-.', color='red')
plt.title("Rosenbrock (Minimizar)")
plt.xlabel("Iteraciones")
plt.ylabel("Fitness (Log)")
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('cso_pso_aco_rosenbrock.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfica H1
plt.figure(figsize=(10, 6))
plt.plot(h_cso_h1, label='CSO (Gatos)', linewidth=2)
plt.plot(h_pso_h1, label='PSO (Partículas)', linestyle='--')
plt.plot(h_aco_h1, label='ACO (Hormigas)', linestyle=':')
plt.plot(h_fir_h1, label='Firefly (Luciérnagas)', linestyle='-.', color='red')
plt.title("Función H1 (Maximizar)")
plt.xlabel("Iteraciones")
plt.ylabel("Fitness")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('cso_pso_aco_h1.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResultados Finales (Mejor encontrado):")
print("-" * 50)
print(f"{'Algoritmo':<10} | {'Rosenbrock (Min)':<20} | {'H1 (Max)':<20}")
print("-" * 50)
print(f"CSO        | {h_cso_ros[-1]:.4e}           | {h_cso_h1[-1]:.4f}")
print(f"PSO        | {h_pso_ros[-1]:.4e}           | {h_pso_h1[-1]:.4f}")
print(f"ACO        | {h_aco_ros[-1]:.4e}           | {h_aco_h1[-1]:.4f}")
print(f"Firefly    | {h_fir_ros[-1]:.4e}           | {h_fir_h1[-1]:.4f}")
print("-" * 50)