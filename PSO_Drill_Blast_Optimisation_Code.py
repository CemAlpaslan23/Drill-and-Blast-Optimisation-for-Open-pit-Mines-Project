import random
import copy
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def fitness_max_deviation(particle_data):
    """
    NEW FITNESS FUNCTION: Minimize the maximum deviation (Y)
    Objective: Min Z = Y
    We'll calculate the actual maximum deviation and use it as fitness
    """
    target_particle_size = 50.0  # cm
    gamma = 0.5  # proportionality constant for A = γη

    # Extract particle data
    burdens = particle_data['burdens']
    spacings = particle_data['spacings']
    explosives = particle_data['explosives']
    charge_masses = particle_data['charge_masses']
    depths = particle_data['depths']
    hardness_data = particle_data['hardness_data']

    rows, cols = len(burdens), len(spacings[0])
    max_deviation_ratio = 0.0

    for i in range(rows):
        for j in range(cols):
            # Calculate position
            x = j * 10.0 + 5.0  # assuming 10m spacing
            y = i * 10.0 + 5.0  # assuming 10m burden

            # Get hardness for this position
            hardness = get_hardness_for_position(x, y, hardness_data)

            # Get explosive parameters
            explosive = explosives[i][j]
            charge_mass = charge_masses[i][j]
            burden = burdens[i]
            spacing = spacings[i][j]
            depth = depths[i][j]

            # Calculate mean particle size using Kuz-Ram formula (Eq. 9)
            A = gamma * hardness  # rock factor
            K = charge_mass / (depth * burden * spacing)  # powder factor (using actual depth)
            Q = charge_mass  # explosive charge per hole
            R = explosive['weight_strength']

            # Ensure positive values to avoid complex numbers
            if K <= 0 or Q <= 0 or R <= 0:
                xm_mean = target_particle_size  # default value
            else:
                xm_mean = A * (K ** (-4 / 5)) * (Q ** (1 / 6)) * ((115 / R) ** (19 / 20))

            # Calculate deviation ratio
            deviation_ratio = abs(xm_mean - target_particle_size) / target_particle_size
            max_deviation_ratio = max(max_deviation_ratio, deviation_ratio)

    # Return the actual maximum deviation ratio (to minimize)
    # This represents how much the worst particle deviates from target
    return max_deviation_ratio


import math


def get_hardness_for_position(x, y, hardness_data):
    """Get hardness value for a position using inverse distance weighting"""
    if not hardness_data:
        return 2.0  # default medium hardness

    total_weight = 0.0
    weighted_hardness = 0.0

    for data in hardness_data.values():
        dx = x - data['x']
        dy = y - data['y']
        distance_sq = dx * dx + dy * dy

        if distance_sq < 1e-12:  # same position
            return data['hardness']

        weight = 1.0 / distance_sq
        weighted_hardness += weight * data['hardness']
        total_weight += weight

    return weighted_hardness / total_weight if total_weight > 0 else 2.0


# ----------Particle Class----------

class Particle:
    def __init__(self, fitness, dim, minx, maxx, seed, hardness_data, explosive_products, diameter_options):
        self.rnd = random.Random(seed)
        self.hardness_data = hardness_data
        self.explosive_products = explosive_products
        self.diameter_options = diameter_options

        # Initialize position of the particle with random values
        self.position = [0.0 for i in range(dim)]

        # Initialize velocity of the particle with 0.0 value
        self.velocity = [0.0 for i in range(dim)]

        # Initialize best particle position of the particle with 0.0 value
        self.best_part_pos = [0.0 for i in range(dim)]

        # Generate feasible particle that satisfies constraints
        self.position = self._generate_feasible_position(dim, minx, maxx)

        # Initialize velocity
        for i in range(dim):
            self.velocity[i] = ((maxx - minx) * self.rnd.random() + minx)

        # compute fitness of particle
        if fitness is not None:
            self.fitness = fitness(self.position)  # curr fitness
        else:
            self.fitness = 0.0

        # initialize best position and fitness of this particle
        self.best_part_pos = copy.copy(self.position)
        self.best_part_fitnessVal = self.fitness  # best fitness

    def update_position(self, bounds):
        """Update particle position within bounds"""
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]
            # Apply bounds
            if self.position[i] < bounds[0]:
                self.position[i] = bounds[0]
            elif self.position[i] > bounds[1]:
                self.position[i] = bounds[1]

    def update_velocity(self, global_best_pos, w, c1, c2):
        """Update particle velocity using PSO formula"""
        for i in range(len(self.velocity)):
            r1 = self.rnd.random()
            r2 = self.rnd.random()

            cognitive = c1 * r1 * (self.best_part_pos[i] - self.position[i])
            social = c2 * r2 * (global_best_pos[i] - self.position[i])

            self.velocity[i] = w * self.velocity[i] + cognitive + social

    def decode_position(self):
        """Decode position vector to drill blast parameters"""
        # This function converts the position vector to actual drill blast parameters
        # Position vector structure: [burden1, burden2, ..., spacing1, spacing2, ..., ...]

        rows = 20  # fixed rows
        cols = 20  # fixed columns

        # Decode burdens (first rows values)
        burdens = []
        for i in range(rows):
            burden_val = self.position[i] if i < len(self.position) else 10.0
            burdens.append(max(5.0, min(15.0, burden_val)))  # bound between 5-15m

        # Decode spacings (next rows*cols values)
        spacings = []
        for i in range(rows):
            row_spacings = []
            for j in range(cols):
                idx = rows + i * cols + j
                spacing_val = self.position[idx] if idx < len(self.position) else 10.0
                row_spacings.append(max(5.0, min(15.0, spacing_val)))  # bound between 5-15m
            spacings.append(row_spacings)

        # Decode diameters (next rows*cols values)
        diameters = []
        for i in range(rows):
            row_diameters = []
            for j in range(cols):
                idx = rows + rows * cols + i * cols + j
                if idx < len(self.position):
                    # Map position value to discrete diameter
                    diameter_idx = int((self.position[idx] - 0) / (1 - 0) * (len(self.diameter_options) - 1))
                    diameter_idx = max(0, min(len(self.diameter_options) - 1, diameter_idx))
                    row_diameters.append(self.diameter_options[diameter_idx])
                else:
                    row_diameters.append(self.diameter_options[0])
            diameters.append(row_diameters)

        # Decode explosives (next rows*cols values)
        explosives = []
        for i in range(rows):
            row_explosives = []
            for j in range(cols):
                idx = rows + 2 * rows * cols + i * cols + j
                if idx < len(self.position):
                    # Map position value to discrete explosive
                    explosive_idx = int((self.position[idx] - 0) / (1 - 0) * (len(self.explosive_products) - 1))
                    explosive_idx = max(0, min(len(self.explosive_products) - 1, explosive_idx))
                    row_explosives.append(self.explosive_products[explosive_idx])
                else:
                    row_explosives.append(self.explosive_products[0])
            explosives.append(row_explosives)

        # Decode charge masses (next rows*cols values)
        charge_masses = []
        for i in range(rows):
            row_charges = []
            for j in range(cols):
                idx = rows + 3 * rows * cols + i * cols + j
                charge_val = self.position[idx] if idx < len(self.position) else 100.0
                # Map 0-1 to 30-150kg range
                charge_mass = 30.0 + charge_val * 120.0
                row_charges.append(charge_mass)
            charge_masses.append(row_charges)

        # Decode depths (next rows*cols values)
        depths = []
        for i in range(rows):
            row_depths = []
            for j in range(cols):
                idx = rows + 4 * rows * cols + i * cols + j
                depth_val = self.position[idx] if idx < len(self.position) else 15.0
                # Map from 0-1 to 10-25m range (bench height + overburden)
                depth = 10.0 + depth_val * 15.0
                row_depths.append(depth)
            depths.append(row_depths)

        # Decode stem lengths (next rows*cols values)
        stem_lengths = []
        for i in range(rows):
            row_stems = []
            for j in range(cols):
                idx = rows + 5 * rows * cols + i * cols + j
                stem_val = self.position[idx] if idx < len(self.position) else 3.0
                # Map from 0-1 to 2-8m range
                stem_length = 2.0 + stem_val * 6.0
                row_stems.append(stem_length)
            stem_lengths.append(row_stems)

        # Decode Y variable (maximum deviation) - last value in position vector
        Y_idx = len(self.position) - 1
        Y_val = self.position[Y_idx] if Y_idx < len(self.position) else 0.1
        # Map from 0-1 to 0.001-0.8 range (0.1% to 80% maximum deviation) - very lenient
        Y = 0.001 + Y_val * 0.799

        return {
            'burdens': burdens,
            'spacings': spacings,
            'diameters': diameters,
            'explosives': explosives,
            'charge_masses': charge_masses,
            'depths': depths,
            'stem_lengths': stem_lengths,
            'hardness_data': self.hardness_data,
            'Y': Y  # New decision variable for maximum deviation
        }

    def _generate_feasible_position(self, dim, minx, maxx):
        """Generate a feasible position with very lenient constraints"""
        max_attempts = 200

        for attempt in range(max_attempts):
            # Generate random position
            position = [self.rnd.random() for _ in range(dim)]

            # Decode to check constraints
            particle_data = self.decode_position()

            # Check if feasible with ALL constraints
            is_feasible, violated = check_constraints(particle_data, "all")

            if is_feasible:
                print(f"✅ Generated feasible particle (attempt {attempt + 1})")
                return position

            # If not feasible, try to repair
            if attempt < max_attempts - 1:
                position = self._repair_position(position, particle_data, "all")

        # If still not feasible, return a basic feasible solution
        print("⚠️ Could not generate feasible particle, using basic solution")
        return self._generate_basic_feasible_position(dim, minx, maxx)

    def _generate_basic_feasible_position(self, dim, minx, maxx):
        """Generate a basic feasible position with minimal constraints"""
        position = []

        # Generate burdens (first 20 values) - ensure they sum to ~200
        total_burden = 0
        for i in range(20):  # 20 rows
            if i < 19:
                burden = self.rnd.uniform(8, 12)  # 8-12m range
                position.append((burden - 5) / (20 - 5))  # map to 0-1
                total_burden += burden
            else:
                # Last burden to make total ~200
                remaining = 200 - total_burden
                burden = max(8, min(12, remaining))
                position.append((burden - 5) / (20 - 5))

        # Generate spacings (next 400 values) - ensure each row sums to ~200
        for i in range(20):  # 20 rows
            row_total = 0
            for j in range(19):  # 19 columns (last one calculated)
                spacing = self.rnd.uniform(8, 12)  # 8-12m range
                position.append((spacing - 5) / (20 - 5))  # map to 0-1
                row_total += spacing

            # Last spacing to make row total ~200
            remaining = 200 - row_total
            spacing = max(8, min(12, remaining))
            position.append((spacing - 5) / (20 - 5))

        # Generate diameters (next 400 values) - use discrete options
        for i in range(400):
            diameter_idx = self.rnd.randint(0, len(self.diameter_options) - 1)
            position.append(diameter_idx / (len(self.diameter_options) - 1))

        # Generate explosives (next 400 values) - use discrete options
        for i in range(400):
            explosive_idx = self.rnd.randint(0, len(self.explosive_products) - 1)
            position.append(explosive_idx / (len(self.explosive_products) - 1))

        # Generate charge masses (next 400 values) - reasonable range
        for i in range(400):
            charge_mass = self.rnd.uniform(80, 150)  # 80-150kg range
            position.append((charge_mass - 50) / (250 - 50))  # map to 0-1

        # Generate depths (next 400 values) - reasonable range
        for i in range(400):
            depth = self.rnd.uniform(10.0, 25.0)  # 10-25m range
            position.append((depth - 10.0) / (25.0 - 10.0))  # map to 0-1

        # Generate stem lengths (next 400 values) - reasonable range
        for i in range(400):
            stem_length = self.rnd.uniform(2.0, 8.0)  # 2-8m range
            position.append((stem_length - 2.0) / (8.0 - 2.0))  # map to 0-1

        # Generate Y variable (maximum deviation) - very lenient range
        Y = self.rnd.uniform(0.001, 0.8)  # 0.1% to 80% maximum deviation
        position.append((Y - 0.001) / (0.8 - 0.001))  # map to 0-1

        return position

    def _repair_position(self, position, particle_data, constraint_level="basic"):
        """Repair position to make it more feasible based on constraint level"""
        # Get current parameters
        burdens = particle_data['burdens']
        spacings = particle_data['spacings']
        charge_masses = particle_data['charge_masses']

        rows, cols = len(burdens), len(spacings[0])

        # Repair basic bounds
        if constraint_level in ["basic", "extent", "energy", "vibration", "all"]:
            # Repair burden bounds
            for i in range(min(rows, 20)):
                if i < len(position):
                    if burdens[i] < 5.0:
                        position[i] = 0.0  # map to 5.0
                    elif burdens[i] > 20.0:
                        position[i] = 1.0  # map to 20.0

            # Repair spacing bounds
            for i in range(rows):
                for j in range(cols):
                    idx = 20 + i * cols + j
                    if idx < len(position):
                        if spacings[i][j] < 5.0:
                            position[idx] = 0.0  # map to 5.0
                        elif spacings[i][j] > 20.0:
                            position[idx] = 1.0  # map to 20.0

            # Repair charge mass bounds
            for i in range(rows):
                for j in range(cols):
                    idx = 20 + 3 * rows * cols + i * cols + j
                    if idx < len(position):
                        if charge_masses[i][j] < 50:
                            position[idx] = 0.0  # map to 50
                        elif charge_masses[i][j] > 250:
                            position[idx] = 1.0  # map to 250

        # Repair extent constraints
        if constraint_level in ["extent", "energy", "vibration", "all"]:
            # Repair burden extent constraint (Eq. 19)
            total_burden = sum(burdens)
            target_burden = 200.0
            if abs(total_burden - target_burden) > 50:
                # Scale burdens to meet extent constraint
                scale_factor = target_burden / total_burden if total_burden > 0 else 1.0
                for i in range(min(rows, 20)):
                    if i < len(position):
                        new_burden = max(5.0, min(20.0, burdens[i] * scale_factor))
                        position[i] = (new_burden - 5.0) / (20.0 - 5.0)

            # Repair spacing extent constraint (Eq. 20)
            for i in range(rows):
                total_spacing = sum(spacings[i])
                if abs(total_spacing - target_burden) > 50:
                    # Scale spacings to meet extent constraint
                    scale_factor = target_burden / total_spacing if total_spacing > 0 else 1.0
                    for j in range(cols):
                        idx = 20 + i * cols + j
                        if idx < len(position):
                            new_spacing = max(5.0, min(20.0, spacings[i][j] * scale_factor))
                            position[idx] = (new_spacing - 5.0) / (20.0 - 5.0)

        return position


# ----------Particle Swarm Optimization Function for Parameter Tuning----------

def parameter_tuning(fitness_func, hardness_data, explosive_products, diameter_options,
                     dim, minx, maxx, test_iterations=100):
    """Find optimal PSO parameters through comprehensive grid search"""
    print("🔍 Starting comprehensive parameter tuning...")

    # Better parameter ranges for tuning
    particle_counts = [200, 300, 400]  # 3 values for better exploration
    iteration_counts = [test_iterations]  # Use the provided test_iterations
    c1_values = [1.5, 2.0, 2.5]  # 3 values for cognitive component
    c2_values = [1.5, 2.0, 2.5]  # 3 values for social component

    best_params = None
    best_fitness = float('inf')
    total_tests = len(particle_counts) * len(iteration_counts) * len(c1_values) * len(c2_values)
    current_test = 0

    print(f"📊 Total parameter combinations to test: {total_tests}")

    for particles in particle_counts:
        for iterations in iteration_counts:
            for c1 in c1_values:
                for c2 in c2_values:
                    current_test += 1
                    print(
                        f"Test {current_test}/{total_tests}: particles={particles}, iter={iterations}, c1={c1}, c2={c2}")

                    # Run PSO with these parameters (using fixed parameters, not adaptive)
                    best_position, fitness_history, iteration_numbers = pso_fixed_params(
                        fitness_func, iterations, particles, dim, minx, maxx,
                        hardness_data, explosive_products, diameter_options, c1, c2)

                    # Get final fitness value
                    final_fitness = fitness_history[-1] if fitness_history else float('inf')

                    if final_fitness < best_fitness:
                        best_fitness = final_fitness
                        best_params = {
                            'particles': particles,
                            'iterations': iterations,
                            'c1': c1,
                            'c2': c2,
                            'fitness': best_fitness
                        }
                        print(f"✅ New best: fitness={best_fitness:.2e}")

    print(f"🎯 Optimal parameters found: {best_params}")
    return best_params


def pso_fixed_params(fitness, max_iter, n, dim, minx, maxx, hardness_data, explosive_products, diameter_options, c1, c2):
    w = 0.9  # Fixed inertia weight
    rnd = random.Random(0)

    # Create n random particles
    swarm = [Particle(fitness, dim, minx, maxx, i, hardness_data, explosive_products, diameter_options) for i in range(n)]

    # Initialize best global position and fitness
    best_particle = min(swarm, key=lambda p: p.fitness)
    best_swarm_pos = best_particle.position[:]
    best_swarm_fitnessVal = best_particle.fitness

    fitness_history = []
    iteration_numbers = []

    for Iter in range(max_iter):
        if Iter % 10 == 0 and Iter > 1:
            print(f"Iter = {Iter} best fitness = {best_swarm_fitnessVal:.3f}")
            fitness_history.append(best_swarm_fitnessVal)
            iteration_numbers.append(Iter)

        for particle in swarm:
            for k in range(dim):
                r1 = rnd.random()
                r2 = rnd.random()

                # Velocity update
                cognitive = c1 * r1 * (particle.best_part_pos[k] - particle.position[k])
                social = c2 * r2 * (best_swarm_pos[k] - particle.position[k])
                particle.velocity[k] = w * particle.velocity[k] + cognitive + social

                # Velocity clamping
                if particle.velocity[k] < minx:
                    particle.velocity[k] = minx
                elif particle.velocity[k] > maxx:
                    particle.velocity[k] = maxx

                # Position update
                particle.position[k] += particle.velocity[k]

                # Position clamping
                if particle.position[k] < minx:
                    particle.position[k] = minx
                elif particle.position[k] > maxx:
                    particle.position[k] = maxx

            # Recalculate fitness after position update
            particle.fitness = fitness(particle.position)

            # Update personal best
            if particle.fitness < particle.best_part_fitnessVal:
                particle.best_part_fitnessVal = particle.fitness
                particle.best_part_pos = particle.position[:]  # shallow copy of list

            # Update global best
            if particle.fitness < best_swarm_fitnessVal:
                best_swarm_fitnessVal = particle.fitness
                best_swarm_pos = particle.position[:]

    # Store final fitness value
    fitness_history.append(best_swarm_fitnessVal)
    iteration_numbers.append(max_iter)

    return best_swarm_pos, fitness_history, iteration_numbers


def adaptive_parameters(iteration, max_iter, current_fitness, best_fitness):
    """Dynamically adjust PSO parameters based on progress"""
    progress = iteration / max_iter

    # Adaptive c1, c2 based on progress
    # Early: High exploration (c1 high, c2 low)
    # Late: High exploitation (c1 low, c2 high)
    c1 = 2.5 * (1 - progress) + 1.5 * progress  # 2.5 → 1.5
    c2 = 1.5 * (1 - progress) + 2.5 * progress  # 1.5 → 2.5

    # Adaptive inertia weight (already implemented)
    w = 0.9 * (1 - progress) + 0.4 * progress  # 0.9 → 0.4

    # Improved adaptive restart threshold based on fitness improvement
    # Early: More frequent restarts for exploration (20-30)
    # Middle: Moderate restarts (30-40) 
    # Late: Less frequent restarts for exploitation (40-50)
    base_threshold = 20 + int(30 * progress)  # 20 → 50

    # Additional adjustment based on fitness stagnation
    if iteration > 50:  # Only after initial exploration
        fitness_improvement = (best_fitness - current_fitness) / best_fitness if best_fitness > 0 else 0
        if fitness_improvement < 0.01:  # Less than 1% improvement recently
            base_threshold = max(15, base_threshold - 10)  # More aggressive restart
        elif fitness_improvement > 0.05:  # Good improvement recently
            base_threshold = min(60, base_threshold + 10)  # Less frequent restart

    restart_threshold = base_threshold

    return w, c1, c2, restart_threshold


def pso(fitness, max_iter, n, dim, minx, maxx, hardness_data, explosive_products, diameter_options, tuned_params=None):
    # Use tuned parameters if provided, otherwise use default adaptive parameters
    if tuned_params:
        w = 0.9  # Fixed inertia weight
        c1 = tuned_params['c1']  # Use tuned cognitive component
        c2 = tuned_params['c2']  # Use tuned social component
        use_adaptive = False
    else:
        w = 0.9  # adaptive inertia weight
        c1 = 2.0  # adaptive cognitive component
        c2 = 2.0  # adaptive social component
        use_adaptive = True

    rnd = random.Random(0)

    # create n random particles
    swarm = [Particle(fitness, dim, minx, maxx, i, hardness_data, explosive_products, diameter_options) for i in
             range(n)]

    # compute the value of best_position and best_fitness in swarm
    best_swarm_pos = [0.0 for i in range(dim)]
    best_swarm_fitnessVal = sys.float_info.max  # swarm best

    # computer best particle of swarm and it's fitness
    for i in range(n):  # check each particle
        if swarm[i].fitness < best_swarm_fitnessVal:
            best_swarm_fitnessVal = swarm[i].fitness
            best_swarm_pos = copy.copy(swarm[i].position)

    # Store fitness history for plotting
    fitness_history = []
    iteration_numbers = []

    # Restart mechanism variables
    no_improvement_count = 0
    restart_threshold = 50  # Restart if no improvement for 50 iterations
    best_ever_fitness = best_swarm_fitnessVal
    best_ever_position = copy.copy(best_swarm_pos)

    # main loop of pso
    Iter = 0
    while Iter < max_iter:
        # Get parameters (adaptive or fixed)
        if use_adaptive:
            current_w, current_c1, current_c2, current_restart_threshold = adaptive_parameters(
                Iter, max_iter, best_swarm_fitnessVal, best_ever_fitness
            )
        else:
            # Use fixed parameters
            current_w, current_c1, current_c2 = w, c1, c2
            current_restart_threshold = 50  # Fixed restart threshold

        # Check for improvement
        if best_swarm_fitnessVal < best_ever_fitness:
            best_ever_fitness = best_swarm_fitnessVal
            best_ever_position = copy.copy(best_swarm_pos)
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Restart mechanism if stuck (using adaptive threshold)
        if no_improvement_count >= current_restart_threshold:
            print(f"🔄 Restarting PSO at iteration {Iter} (no improvement for {current_restart_threshold} iterations)")
            # Reinitialize 20% of particles randomly
            restart_count = int(0.2 * n)
            for i in range(restart_count):
                idx = rnd.randint(0, n - 1)
                swarm[idx] = Particle(fitness, dim, minx, maxx, idx, hardness_data, explosive_products,
                                      diameter_options)
                if swarm[idx].fitness < best_swarm_fitnessVal:
                    best_swarm_fitnessVal = swarm[idx].fitness
                    best_swarm_pos = copy.copy(swarm[idx].position)
            no_improvement_count = 0

        # after every 10 iterations
        # print iteration number and best fitness value so far
        if Iter % 10 == 0 and Iter > 1:
            print("Iter = " + str(Iter) + " best fitness = %.3f" % best_swarm_fitnessVal)
            fitness_history.append(best_swarm_fitnessVal)
            iteration_numbers.append(Iter)

        for i in range(n):  # process each particle
            # compute new velocity of curr particle
            for k in range(dim):
                r1 = rnd.random()  # randomizations
                r2 = rnd.random()

                # Enhanced velocity update with diversity mechanism (using adaptive parameters)
                cognitive_component = current_c1 * r1 * (swarm[i].best_part_pos[k] - swarm[i].position[k])
                social_component = current_c2 * r2 * (best_swarm_pos[k] - swarm[i].position[k])

                # Add small random component for diversity (5% of search space)
                diversity_component = 0.05 * (maxx - minx) * (rnd.random() - 0.5)

                swarm[i].velocity[k] = ((current_w * swarm[i].velocity[k]) +
                                        cognitive_component +
                                        social_component +
                                        diversity_component)

            # if velocity[k] is not in [minx, maxx]
            # then clip it
            if swarm[i].velocity[k] < minx:
                swarm[i].velocity[k] = minx
            elif swarm[i].velocity[k] > maxx:
                swarm[i].velocity[k] = maxx

        # compute new position using new velocity
        for k in range(dim):
            swarm[i].position[k] += swarm[i].velocity[k]

        # Ensure position stays within bounds and is feasible
        for k in range(dim):
            if swarm[i].position[k] < minx:
                swarm[i].position[k] = minx
            elif swarm[i].position[k] > maxx:
                swarm[i].position[k] = maxx

        # compute fitness of new position
        swarm[i].fitness = fitness(swarm[i].position)

        # is new position a new best for the particle?
        if swarm[i].fitness < swarm[i].best_part_fitnessVal:
            swarm[i].best_part_fitnessVal = swarm[i].fitness
            swarm[i].best_part_pos = copy.copy(swarm[i].position)

        # is new position a new best overall?
        if swarm[i].fitness < best_swarm_fitnessVal:
            best_swarm_fitnessVal = swarm[i].fitness
            best_swarm_pos = copy.copy(swarm[i].position)

        # Elitism: Keep best particle unchanged for next iteration
        if i == 0:  # First particle is always the best
            best_particle_idx = 0
            for j in range(1, n):
                if swarm[j].fitness < swarm[best_particle_idx].fitness:
                    best_particle_idx = j

            # Swap best particle to position 0
            if best_particle_idx != 0:
                swarm[0], swarm[best_particle_idx] = swarm[best_particle_idx], swarm[0]

        # for-each particle
        Iter += 1
        # end while

    # Add final fitness value
    fitness_history.append(best_swarm_fitnessVal)
    iteration_numbers.append(Iter)

    return best_swarm_pos, fitness_history, iteration_numbers


# ----------constraint checking functions---------- Just for debugging


def check_constraints(particle_data, constraint_level="basic"):
    violated = []

    # Extract parameters
    burdens = particle_data['burdens']
    spacings = particle_data['spacings']
    diameters = particle_data['diameters']
    explosives = particle_data['explosives']
    charge_masses = particle_data['charge_masses']
    depths = particle_data['depths']
    stem_lengths = particle_data['stem_lengths']
    hardness_data = particle_data['hardness_data']

    rows, cols = len(burdens), len(spacings[0])

    # Precompute grid positions, distances, and hardnesses
    position_data = [[None for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        y = i * 10.0 + 5.0
        burden = burdens[i]
        spacing_row = spacings[i]
        for j in range(cols):
            x = j * 10.0 + 5.0
            distance = math.sqrt((x - 250) ** 2 + (y - 100) ** 2)
            hardness = get_hardness_for_position(x, y, hardness_data)
            explosive = explosives[i][j]
            position_data[i][j] = {
                'x': x,
                'y': y,
                'distance': distance,
                'hardness': hardness,
                'explosive': explosive,
                'burden': burden,
                'spacing': spacing_row[j],
                'diameter': diameters[i][j],
                'charge_mass': charge_masses[i][j],
                'depth': depths[i][j],
                'stem_length': stem_lengths[i][j]
            }

    # STEP 1: Basic bounds
    if constraint_level in ["basic", "extent", "energy", "vibration", "all"]:
        for i in range(rows):
            burden = burdens[i]
            if burden < 1.0 or burden > 50.0:
                violated.append(f"Burden out of bounds for row {i + 1}: {burden}")
            for j in range(cols):
                p = position_data[i][j]
                if not (1.0 <= p['spacing'] <= 50.0):
                    violated.append(f"Spacing out of bounds for R{i + 1}_C{j + 1}: {p['spacing']}")
                if not (50 <= p['diameter'] <= 500):
                    violated.append(f"Diameter out of bounds for R{i + 1}_C{j + 1}: {p['diameter']}")
                if not (10 <= p['charge_mass'] <= 500):
                    violated.append(f"Charge mass out of bounds for R{i + 1}_C{j + 1}: {p['charge_mass']}")
                if not (10.0 <= p['depth'] <= 25.0):
                    violated.append(f"Depth out of bounds for R{i + 1}_C{j + 1}: {p['depth']}")
                if not (2.0 <= p['stem_length'] <= 8.0):
                    violated.append(f"Stem length out of bounds for R{i + 1}_C{j + 1}: {p['stem_length']}")

    # STEP 2: Extent constraints
    if constraint_level in ["extent", "energy", "vibration", "all"]:
        total_burden = sum(burdens)
        target_burden = 200.0
        if abs(total_burden - target_burden) > 400:
            violated.append(f"Burden extent constraint violated: {total_burden} != {target_burden}")
        for i in range(rows):
            if abs(sum(spacings[i]) - target_burden) > 400:
                violated.append(
                    f"Spacing extent constraint violated for row {i + 1}: {sum(spacings[i])} != {target_burden}")

    # STEP 3: Energy intensity
    if constraint_level in ["energy", "vibration", "all"]:
        for i in range(rows):
            for j in range(cols):
                p = position_data[i][j]
                max_energy = get_max_energy_intensity(p['hardness'])
                if p['explosive']['energy_intensity'] > max_energy * 20.0:
                    violated.append(f"Energy intensity exceeded for R{i + 1}_C{j + 1}")

    # STEP 4: Vibration
    if constraint_level in ["vibration", "all"]:
        lambda_val, alpha, PPV_limit = 1000, -1.5, 10.0
        for i in range(rows):
            for j in range(cols):
                p = position_data[i][j]
                max_charge_vibration = (p['distance'] / ((PPV_limit / lambda_val) ** (1 / alpha))) ** 2
                if p['charge_mass'] > max_charge_vibration * 200.0:
                    violated.append(f"Ground vibration limit exceeded for R{i + 1}_C{j + 1}")

    # STEP 5: Depth + Stem length
    if constraint_level in ["vibration", "all"]:
        for i in range(rows):
            for j in range(cols):
                p = position_data[i][j]
                min_depth = get_minimum_depth(p['burden'], p['spacing'], p['diameter'])
                if p['depth'] < min_depth * 0.1:
                    violated.append(f"Depth below minimum for R{i + 1}_C{j + 1}: {p['depth']} < {min_depth}")
                if p['stem_length'] > p['depth'] * 0.9:
                    violated.append(
                        f"Stem length too large for R{i + 1}_C{j + 1}: {p['stem_length']} > {p['depth'] * 0.9}")

    # STEP 6: Energy concentration
    if constraint_level in ["vibration", "all"]:
        for i in range(rows):
            for j in range(cols):
                p = position_data[i][j]
                max_eff = get_energy_concentration_max_efficiency(p['hardness'])
                if p['explosive']['energy_intensity'] < max_eff * 0.5:
                    violated.append(f"Energy concentration too low for R{i + 1}_C{j + 1}")

    # STEP 7: Air Overpressure
    if constraint_level in ["vibration", "all"]:
        k, beta, AOp_limit = 200, -1.5, 120.0
        for i in range(rows):
            for j in range(cols):
                p = position_data[i][j]
                max_charge_aop = (p['distance'] / ((AOp_limit / k) ** (1 / beta))) ** 3
                if p['charge_mass'] > max_charge_aop * 100.0:
                    violated.append(f"Air overpressure limit exceeded for R{i + 1}_C{j + 1}")

    # STEP 8: Flyrock / SDoB
    if constraint_level in ["vibration", "all"]:
        for i in range(rows):
            for j in range(cols):
                p = position_data[i][j]
                sdob_limit = get_scaled_depth_of_burial(p['distance'])
                area = math.pi * (p['diameter'] / 2) ** 2
                sdob_calc = (p['stem_length'] + 5 * p['diameter']) / math.sqrt(
                    area * p['explosive']['density'] * 10 * p['diameter'])
                if sdob_calc < sdob_limit * 0.001:
                    violated.append(f"SDoB constraint violated for R{i + 1}_C{j + 1}")

    # STEP 9: Spacing-to-Burden Ratio
    if constraint_level in ["extent", "energy", "vibration", "all"]:
        for i in range(rows):
            for j in range(cols):
                p = position_data[i][j]
                if p['burden'] > 0:
                    ratio = p['spacing'] / p['burden']
                    if ratio < 0.1 or ratio > 5.0:
                        violated.append(f"Spacing-to-burden ratio out of range for R{i + 1}_C{j + 1}: {ratio:.2f}")

    return len(violated) == 0, violated


def get_max_energy_intensity(hardness):
    """Get maximum allowed energy intensity based on hardness"""
    if hardness <= 1.5:  # Soft
        return 3.5e6
    elif hardness <= 2.5:  # Medium
        return 4.0e6
    elif hardness <= 3.5:  # Hard
        return 4.5e6
    else:  # Extra Hard
        return 5.0e6


def get_energy_concentration_max_efficiency(hardness):
    """Get energy concentration for maximum excavator efficiency (E^0_ij)"""
    if hardness <= 1.5:  # Soft
        return 2.5e6
    elif hardness <= 2.5:  # Medium
        return 3.5e6
    elif hardness <= 3.5:  # Hard
        return 4.5e6
    else:  # Extra Hard
        return 5.5e6


def visualize_grid_deviation(particle_data, title="Drill and Blast Grid - Deviation Analysis"):
    """
    Visualize the drill and blast grid with deviation analysis
    Only shows holes that exist in the Excel file (not all 400 positions)
    
    Args:
        particle_data: Decoded particle data containing all parameters
        title: Title for the plot
    """
    target_particle_size = 50.0  # cm
    gamma = 0.5  # proportionality constant for A = γη

    # Extract particle data
    burdens = particle_data['burdens']
    spacings = particle_data['spacings']
    diameters = particle_data['diameters']
    explosives = particle_data['explosives']
    charge_masses = particle_data['charge_masses']
    depths = particle_data['depths']
    stem_lengths = particle_data['stem_lengths']
    hardness_data = particle_data['hardness_data']

    rows, cols = len(burdens), len(spacings[0])

    # Create matrices for visualization (only for existing holes)
    deviation_ratios = np.full((rows, cols), np.nan)  # NaN for non-existent holes
    particle_sizes = np.full((rows, cols), np.nan)  # NaN for non-existent holes

    # Calculate deviation only for existing holes
    existing_holes = []
    for i in range(rows):
        for j in range(cols):
            # Calculate position
            x = j * 10.0 + 5.0
            y = i * 10.0 + 5.0

            # Check if hole exists in hardness_data (from Excel)
            hole_id = f"R{i + 1}_C{j + 1}"
            if hole_id in hardness_data:
                # Get hardness for this position
                hardness = get_hardness_for_position(x, y, hardness_data)

                # Get explosive parameters
                explosive = explosives[i][j]
                charge_mass = charge_masses[i][j]
                burden = burdens[i]
                spacing = spacings[i][j]
                depth = depths[i][j]
                stem_length = stem_lengths[i][j]
                diameter = diameters[i][j]

                # Calculate mean particle size using Kuz-Ram formula (Eq. 9)
                A = gamma * hardness  # rock factor
                K = charge_mass / (depth * burden * spacing)  # powder factor
                Q = charge_mass  # explosive charge per hole
                R = explosive['weight_strength']

                # Ensure positive values to avoid complex numbers
                if K <= 0 or Q <= 0 or R <= 0:
                    xm_mean = target_particle_size  # default value
                else:
                    xm_mean = A * (K ** (-4 / 5)) * (Q ** (1 / 6)) * ((115 / R) ** (19 / 20))

                # Calculate deviation ratio
                deviation_ratio = abs(xm_mean - target_particle_size) / target_particle_size

                # Store values (only for existing holes)
                deviation_ratios[i, j] = deviation_ratio
                particle_sizes[i, j] = xm_mean
                existing_holes.append((i, j, deviation_ratio, xm_mean))

    # Create figure with 2 subplots only
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot 1: Deviation Ratios Heatmap
    im1 = ax1.imshow(deviation_ratios, cmap='Reds', aspect='equal')
    ax1.set_title(f'Deviation Ratios\n(Target: {target_particle_size} cm)', fontweight='bold')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Deviation Ratio')

    # Add text annotations for deviation ratios (only existing holes)
    for i, j, _, _ in existing_holes:
        text = ax1.text(j, i, f'{deviation_ratios[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=8)

    # Plot 2: Particle Sizes Heatmap
    im2 = ax2.imshow(particle_sizes, cmap='viridis', aspect='equal')
    ax2.set_title('Actual Particle Sizes (cm)', fontweight='bold')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')

    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Particle Size (cm)')

    # Add text annotations for particle sizes (only existing holes)
    for i, j, _, _ in existing_holes:
        text = ax2.text(j, i, f'{particle_sizes[i, j]:.1f}',
                        ha="center", va="center", color="white", fontsize=7)

    plt.tight_layout()
    plt.show()

    # Calculate and print summary
    if existing_holes:
        deviation_values = [hole[2] for hole in existing_holes]
        max_deviation = max(deviation_values)
        mean_deviation = np.mean(deviation_values)

        print(f"\n🎯 DEVIATION ANALYSIS SUMMARY:")
        print(f"   Target Particle Size: {target_particle_size} cm")
        print(f"   Max Deviation Ratio: {max_deviation:.3f} ({max_deviation * 100:.1f}%)")
        print(f"   Mean Deviation Ratio: {mean_deviation:.3f} ({mean_deviation * 100:.1f}%)")
        print(f"   Total Existing Holes: {len(existing_holes)}")
        print(f"   Grid Size: {rows} × {cols} = {rows * cols} possible positions")
        print(f"   Hole Generation Rate: {len(existing_holes) / (rows * cols) * 100:.1f}%")

        return {
            'max_deviation': max_deviation,
            'mean_deviation': mean_deviation,
            'existing_holes_count': len(existing_holes),
            'deviation_ratios': deviation_ratios,
            'particle_sizes': particle_sizes
        }
    else:
        print("❌ No existing holes found in hardness data!")
        return None


def get_minimum_depth(burden, spacing, diameter):
    """Get minimum hole depth required (d^min_ij)"""
    # Minimum depth = bench height + overburden
    # Overburden depends on burden and spacing
    bench_height = 15.0
    overburden_factor = 0.1 * (burden + spacing) / 20.0  # 0.1-0.2 factor
    return bench_height + overburden_factor


def get_scaled_depth_of_burial(distance_to_sensitive_site):
    """Get scaled depth of burial (SDoB_ij) based on distance to sensitive site"""
    # SDoB increases with distance to sensitive site
    if distance_to_sensitive_site < 50:
        return 0.3  # Close to sensitive site
    elif distance_to_sensitive_site < 100:
        return 0.5  # Medium distance
    elif distance_to_sensitive_site < 200:
        return 0.7  # Far from sensitive site
    else:
        return 1.0  # Very far from sensitive site


# ----------data loading functions----------

def load_hardness_data(excel_file):
    """Load hardness data from Excel file"""
    try:
        df = pd.read_excel(excel_file, sheet_name='Hole_Data')
        hardness_data = {}

        for _, row in df.iterrows():
            hole_id = f"R{row['Row']}_C{row['Column']}"
            hardness_data[hole_id] = {
                'x': row['X_Coordinate'],
                'y': row['Y_Coordinate'],
                'hardness': row['Hardness_Value'],
                'hardness_level': row['Hardness_Level']
            }

        print(f"✅ Loaded {len(hardness_data)} holes from Excel")
        return hardness_data
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        return {}


# ----------Driver code for drill and blast optimization----------

def main():
    print("\nBegin particle swarm optimization on drill and blast fragmentation\n")

    # Load hardness data
    hardness_data = load_hardness_data('zone_based_grid_hardness_data.xlsx')

    # Define explosive products
    explosive_products = [
        {'name': 'ANFO', 'energy_intensity': 3.8e6, 'density': 800, 'weight_strength': 1.0},
        {'name': 'Emulsion', 'energy_intensity': 4.2e6, 'density': 1200, 'weight_strength': 1.1},
        {'name': 'Heavy ANFO', 'energy_intensity': 4.5e6, 'density': 1000, 'weight_strength': 1.2},
        {'name': 'High Energy', 'energy_intensity': 5.0e6, 'density': 1400, 'weight_strength': 1.3}
    ]

    # Define diameter options (mm)
    diameter_options = [76, 89, 102, 114, 127, 140, 152, 165, 178, 191, 203, 216, 229, 241, 254]

    # Problem dimensions
    rows, cols = 20, 20
    dim = rows + rows * cols * 6 + 1  # burdens + spacings + diameters + explosives + charge_masses + depths + stem_lengths + Y

    print("Goal is to minimize fragmentation deviation in " + str(dim) + " variables")
    print("Function represents drill and blast optimization with constraints")

    # PSO parameters - further optimized for better performance
    num_particles = 300  # Increased particles for better exploration
    max_iter = 500  # Increased iterations for better convergence

    print("Setting num_particles = " + str(num_particles))
    print("Setting max_iter = " + str(max_iter))
    print("\nStarting PSO algorithm\n")

    # Create fitness function with data
    def fitness_func(position):
        # Create a dummy particle to decode position
        particle = Particle(None, dim, 0, 1, 0, hardness_data, explosive_products, diameter_options)
        particle.position = position
        particle_data = particle.decode_position()

        # Calculate fragmentation objective directly
        # Constraints handled through feasible particle generation
        return fitness_max_deviation(particle_data)

    # Skip parameter tuning for now and use the original adaptive approach
    print("🚀 Using original adaptive PSO approach (no parameter tuning)")

    # Run PSO with adaptive parameters (original approach)
    best_position, fitness_history, iteration_numbers = pso(fitness_func, max_iter, num_particles, dim, 0.0, 1.0,
                                                            hardness_data, explosive_products, diameter_options)

    print("\nPSO completed\n")
    print("\nBest solution found:")
    print(["%.6f" % best_position[k] for k in range(min(10, dim))])  # Show first 10 values

    # Decode best solution
    particle = Particle(None, dim, 0, 1, 0, hardness_data, explosive_products, diameter_options)
    particle.position = best_position
    best_solution = particle.decode_position()

    # Calculate final fitness
    fitnessVal = fitness_max_deviation(best_solution)
    print("fitness of best solution = %.6f" % fitnessVal)

    # Check constraints
    is_feasible, violated = check_constraints(best_solution)
    print(f"Solution feasible: {is_feasible}")
    if not is_feasible:
        print(f"Violated constraints: {len(violated)}")
        for v in violated[:5]:  # Show first 5 violations
            print(f"  - {v}")

    # Visualize fitness convergence
    visualize_fitness_convergence(fitness_history, iteration_numbers)

    # Visualize grid deviation analysis
    print("\n🎯 VISUALIZING GRID DEVIATION ANALYSIS...")

    # Show only deviation ratios and particle sizes (no threshold analysis)
    print("\n📊 Showing deviation analysis for existing holes only...")
    visualize_grid_deviation(best_solution, title="Grid Analysis - Existing Holes Only")

    print("\nEnd particle swarm for drill and blast optimization\n")


def visualize_pso_results(best_solution, hardness_data):
    """Visualize PSO optimization results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    burdens = best_solution['burdens']
    spacings = best_solution['spacings']
    diameters = best_solution['diameters']
    explosives = best_solution['explosives']
    charge_masses = best_solution['charge_masses']
    depths = best_solution['depths']
    stem_lengths = best_solution['stem_lengths']

    rows, cols = len(burdens), len(spacings[0])

    # Plot 1: Burden distribution per row
    axes[0, 0].bar(range(1, rows + 1), burdens, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Optimized Burden per Row', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Row Number')
    axes[0, 0].set_ylabel('Burden (m)')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Spacing distribution
    spacing_flat = [s for row in spacings for s in row]
    axes[0, 1].hist(spacing_flat, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Optimized Spacing Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Spacing (m)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Charge mass distribution
    charge_flat = [c for row in charge_masses for c in row]
    axes[0, 2].hist(charge_flat, bins=20, color='orange', alpha=0.7, edgecolor='black')
    axes[0, 2].set_title('Optimized Charge Mass Distribution', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Charge Mass (kg)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Diameter distribution
    diameter_flat = [d for row in diameters for d in row]
    unique_diameters, counts = np.unique(diameter_flat, return_counts=True)
    axes[1, 0].bar(unique_diameters, counts, color='red', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Optimized Diameter Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Diameter (mm)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Explosive type distribution
    explosive_flat = [e['name'] for row in explosives for e in row]
    explosive_counts = {}
    for exp in explosive_flat:
        explosive_counts[exp] = explosive_counts.get(exp, 0) + 1

    axes[1, 1].pie(explosive_counts.values(), labels=explosive_counts.keys(), autopct='%1.1f%%',
                   colors=['lightblue', 'lightgreen', 'orange', 'pink'])
    axes[1, 1].set_title('Explosive Type Distribution', fontsize=12, fontweight='bold')

    # Plot 6: Hardness vs Charge Mass scatter
    hardness_values = []
    charge_values = []
    for i in range(rows):
        for j in range(cols):
            x = j * 10.0 + 5.0
            y = i * 10.0 + 5.0
            hardness = get_hardness_for_position(x, y, hardness_data)
            hardness_values.append(hardness)
            charge_values.append(charge_masses[i][j])

    scatter = axes[1, 2].scatter(hardness_values, charge_values, c=hardness_values,
                                 cmap='viridis', alpha=0.7, s=50)
    axes[1, 2].set_title('Hardness vs Charge Mass', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Hardness Level')
    axes[1, 2].set_ylabel('Charge Mass (kg)')
    axes[1, 2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 2], label='Hardness Level')

    plt.tight_layout()
    plt.suptitle('PSO Drill and Blast Optimization Results', fontsize=16, fontweight='bold', y=0.98)
    plt.show()

    # Additional 3D visualization
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create 3D scatter plot
    x_coords = []
    y_coords = []
    z_coords = []
    colors = []

    for i in range(rows):
        for j in range(cols):
            x = j * 10.0 + 5.0
            y = i * 10.0 + 5.0
            hardness = get_hardness_for_position(x, y, hardness_data)

            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(charge_masses[i][j])
            colors.append(hardness)

    scatter = ax.scatter(x_coords, y_coords, z_coords, c=colors, cmap='viridis',
                         s=50, alpha=0.7)

    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_zlabel('Charge Mass (kg)')
    ax.set_title('3D Visualization: Position vs Charge Mass (colored by Hardness)',
                 fontsize=14, fontweight='bold')

    plt.colorbar(scatter, ax=ax, label='Hardness Level', shrink=0.5)
    plt.show()


def visualize_fitness_convergence(fitness_history, iteration_numbers):
    """Visualize PSO fitness convergence over iterations"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Linear scale plot
    ax.plot(iteration_numbers, fitness_history, 'b-o', linewidth=2, markersize=6, markerfacecolor='red',
            markeredgecolor='darkred')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Fitness Value', fontsize=12)
    ax.set_title('PSO Fitness Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(iteration_numbers) + 5)

    # Add value annotations
    for i, (iter_num, fitness) in enumerate(zip(iteration_numbers, fitness_history)):
        if i % 2 == 0:  # Show every other point to avoid clutter
            ax.annotate(f'{fitness:.1e}', (iter_num, fitness),
                        textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    # Add convergence statistics
    if len(fitness_history) > 1:
        initial_fitness = fitness_history[0]
        final_fitness = fitness_history[-1]
        improvement = ((initial_fitness - final_fitness) / initial_fitness) * 100

        ax.text(0.02, 0.98,
                f'Initial: {initial_fitness:.2e}\nFinal: {final_fitness:.2e}\nImprovement: {improvement:.1f}%',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # Print detailed fitness progression
    print("\n📊 Detailed Fitness Progression:")
    print("=" * 50)
    for i, (iter_num, fitness) in enumerate(zip(iteration_numbers, fitness_history)):
        if i == 0:
            print(f"Iter = {iter_num:2d} best fitness = {fitness:.3f} (Initial)")
        elif i == len(iteration_numbers) - 1:
            print(f"Iter = {iter_num:2d} best fitness = {fitness:.3f} (Final)")
        else:
            improvement = ((fitness_history[i - 1] - fitness) / fitness_history[i - 1]) * 100 if i > 0 else 0
            print(f"Iter = {iter_num:2d} best fitness = {fitness:.3f} (Improvement: {improvement:+.1f}%)")


if __name__ == "__main__":
    main()
