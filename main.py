"""
Comparación Completa de Algoritmos de Enjambre con DEAP
CSO vs PSO vs ACO vs Firefly

Incluye:
1. Cat Swarm Optimization (CSO)
2. Particle Swarm Optimization (PSO)
3. Ant Colony Optimization (ACO Continuo)
4. Firefly Algorithm (FA) - Basado en el código del profesor
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
    # Parámetros CSO
    MR, SMP, SRD, CDC, C1 = 0.2, 5, 0.2, 0.8, 2.0
    
    FitnessClass = creator.FitnessMin if problem_type == "min" else creator.FitnessMax
    # Importante: Aseguramos que la clase exista o se sobrescriba correctamente
    if "AgentCSO" in dir(creator):
        del creator.AgentCSO
    creator.create("AgentCSO", list, fitness=FitnessClass, velocity=list, mode=str)
    
    evaluate = get_evaluate_func(func_name)

    # Inicialización
    cats = []
    for _ in range(N_AGENTS):
        cat = creator.AgentCSO(random.uniform(bounds_min, bounds_max) for _ in range(DIMENSIONS))
        cat.velocity = [random.uniform(-1, 1) for _ in range(DIMENSIONS)]
        cat.mode = "seeking"
        cat.fitness.values = evaluate(cat)
        cats.append(cat)
        
    best_cat = (max(cats, key=lambda c: c.fitness.values[0]) if problem_type == "max" 
                else min(cats, key=lambda c: c.fitness.values[0]))
    best_pos = list(best_cat)
    history = [best_cat.fitness.values[0]]
    
    for _ in range(N_ITERATIONS):
        # Asignar modos
        num_tracing = int(N_AGENTS * MR)
        indices = list(range(N_AGENTS))
        random.shuffle(indices)
        for i, idx in enumerate(indices):
            cats[idx].mode = "tracing" if i < num_tracing else "seeking"
            
        for cat in cats:
            if cat.mode == "seeking":
                copies = []
                for _ in range(SMP):
                    copy_cat = creator.AgentCSO(cat)
                    if random.random() < CDC:
                        for d in range(DIMENSIONS):
                            mutation = SRD * (bounds_max - bounds_min) * (random.random() - 0.5) * 2
                            copy_cat[d] = bound_position(copy_cat[d] + mutation, bounds_min, bounds_max)
                    copies.append(copy_cat)
                
                fits = [evaluate(c)[0] for c in copies]
                best_idx = np.argmax(fits) if problem_type == "max" else np.argmin(fits)
                
                cat[:] = copies[best_idx][:]
                cat.fitness.values = (fits[best_idx],)

            else: # Tracing
                for d in range(DIMENSIONS):
                    r = random.random()
                    cat.velocity[d] += C1 * r * (best_pos[d] - cat[d])
                    v_limit = (bounds_max - bounds_min) * 0.1
                    cat.velocity[d] = max(-v_limit, min(v_limit, cat.velocity[d]))
                    cat[d] = bound_position(cat[d] + cat.velocity[d], bounds_min, bounds_max)
                cat.fitness.values = evaluate(cat)

            # --- CORRECCIÓN AQUÍ ---
            # Actualizar Global
            better = (cat.fitness.values[0] > best_cat.fitness.values[0]) if problem_type == "max" \
                     else (cat.fitness.values[0] < best_cat.fitness.values[0])
            
            if better:
                best_cat = creator.AgentCSO(cat)       # Copia la posición
                best_cat.fitness.values = cat.fitness.values  # <--- ESTA LÍNEA FALTABA
                best_pos = list(cat)
        
        history.append(best_cat.fitness.values[0])
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
# 3. ALGORITMO ACO (ANT COLONY - CONTINUOUS)
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
# 4. ALGORITMO FIREFLY (FA) - INTEGRADO
# =============================================================================

def run_firefly(func_name, bounds_min, bounds_max, problem_type="min"):
    """
    Implementación basada en Firefly.py del profesor.
    Lógica: Las luciérnagas se mueven hacia las más brillantes.
    """
    # Parámetros del profesor
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
        # En el código del profesor, se ordenan primero (opcional pero ayuda a la convergencia)
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
            # Tal como aparece en el código del profesor
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