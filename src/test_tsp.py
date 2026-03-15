import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from problems.discrete import TSPProblem

from algorithms.human.ca import CA_TSP
from algorithms.evolution.GA import GA_TSP
from algorithms.physics.SA import SA_TSP
from algorithms.biology.ACOR import ACO_TSP
from algorithms.classic.local_search import HillClimbing_TSP
from algorithms.classic.informed_search import A_STAR_TSP


# -------------------------------------------------
# Worker function (MUST be top-level for multiprocessing)
# -------------------------------------------------
def run_algorithm(task):
    name, dist_matrix = task

    if name == "CA_TSP":
        algo = CA_TSP(dist_matrix, pop_size=30, max_iter=50, seed=42)

    elif name == "GA_TSP":
        algo = GA_TSP(dist_matrix, pop_size=30, max_iter=50, seed=42)

    elif name == "SA_TSP":
        algo = SA_TSP(
            dist_matrix,
            initial_temperature=1000,
            cooling_rate=0.99,
            max_iterations=200,
            seed=42,
        )

    elif name == "ACO_TSP":
        algo = ACO_TSP(
            dist_matrix,
            n_ants=dist_matrix.shape[0],   # typical choice
            max_iterations=50,
            alpha=1.0,
            beta=2.0,
            rho=0.1,
            q=100.0,
            seed=42,
        )

    elif name == "HillClimbing_TSP":
        algo = HillClimbing_TSP(
            dist_matrix,
            max_restarts=20,
            seed=42,
        )

    elif name == "A_STAR_TSP":
        algo = A_STAR_TSP(dist_matrix, seed=42)

    else:
        raise ValueError(f"Unknown algorithm: {name}")

    algo.solve()

    # Unify result extraction
    if hasattr(algo, "get_results"):
        result = algo.get_results()
        best_fitness = result.get("best_fitness")
        exec_time = result.get("execution_time_seconds", 0)
        best_solution = result.get("best_solution")
        time_complexity = result.get("time_complexity")
        space_complexity = result.get("space_complexity")
        nodes_expanded = result.get("nodes_expanded")
    else:
        best_fitness = algo.best_fitness
        exec_time = algo.execution_time
        best_solution = getattr(algo, "best_solution", None)
        time_complexity = None
        space_complexity = None
        nodes_expanded = getattr(algo, "nodes_expanded", None)

    return {
        "name": name,
        "best_fitness": best_fitness,
        "execution_time": exec_time,
        "best_solution": best_solution,
        "time_complexity": time_complexity,
        "space_complexity": space_complexity,
        "nodes_expanded": nodes_expanded,
    }


# -------------------------------------------------
# Main Test
# -------------------------------------------------
def test_all():

    prob = TSPProblem("tests/TSP/knn/test_0.txt")

    print("\n" + "=" * 100)
    print(f"TSP Test Instance: {prob.n_cities} cities")
    print("=" * 100 + "\n")

    algorithms = [
        "CA_TSP",
        "GA_TSP",
        "SA_TSP",
        "ACO_TSP",
        "HillClimbing_TSP",
        "A_STAR_TSP"
    ]

    tasks = [(name, prob.distance_matrix) for name in algorithms]

    results = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_algorithm, task) for task in tasks]

        for future in as_completed(futures):
            result = future.result()

            print(f"[DONE] {result['name']}")
            print(f"  Best fitness: {result['best_fitness']:.2f}")
            print(f"  Time: {result['execution_time']:.3f}s")

            best_sol = result.get("best_solution")
            if best_sol is not None:
                # Avoid overly long tour prints
                sol_str = str(best_sol)
                if len(sol_str) > 120:
                    sol_str = sol_str[:117] + "..."
                print(f"  Best solution: {sol_str}")

            tc = result.get("time_complexity") or "N/A"
            sc = result.get("space_complexity") or "N/A"
            print(f"  Complexity: time={tc}, space={sc}")

            ne = result.get("nodes_expanded")
            if ne is not None:
                print(f"  Nodes expanded: {ne}")

            print("")

            results.append((result["name"], result["best_fitness"]))

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY (sorted by solution quality)")
    print("=" * 100)

    results.sort(key=lambda x: x[1])

    for i, (name, fitness) in enumerate(results, 1):
        print(f"{i}. {name:20s} : {fitness:.0f}")


if __name__ == "__main__":
    test_all()