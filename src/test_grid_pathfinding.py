#!/usr/bin/env python3
"""
Comprehensive Grid Shortest Path Benchmark Test
Compares classic, informed, and metaheuristic algorithms on SP problem.

Metrics analyzed:
- Computational Complexity: nodes expanded/stored in memory
- Solution Quality: path length optimality
- Exploration vs Exploitation: blind vs heuristic-guided search
"""

import sys
import numpy as np
from problems.discrete import SPProblem
from algorithms.classic.uninformed_search import BFS, DFS, UCS
from algorithms.classic.informed_search import GreedyBestFirst, AStarSearch
from algorithms.evolution.GA import GA_Grid
from algorithms.biology.ACOR import ACO_Grid


class AlgorithmBenchmark:
    """Runs comprehensive benchmark on all grid pathfinding algorithms."""

    def __init__(self, test_file='tests/SP/test_1.txt'):
        self.prob = SPProblem(test_file)
        self.grid_size = self.prob.n * self.prob.m
        self.results = {}
        self.optimal_path_length = None

    def run_all(self):
        """Execute all algorithms and collect metrics."""
        print(f"\n{'='*110}")
        print(f"Grid Shortest Path Benchmark Test")
        print(f"{'='*110}")
        print(f"Grid Size: {self.prob.n} x {self.prob.m} ({self.grid_size} cells)")
        print(f"Start: {self.prob.start_node}, Goal: {self.prob.end_node}")
        print(f"Blocked cells: {np.sum(self.prob.grid == 1)}, Open cells: {np.sum(self.prob.grid == 0)}")
        print(f"{'='*110}\n")

        # ===== CLASSIC UNINFORMED SEARCH (Blind Search) =====
        print("CLASSIC UNINFORMED SEARCH (Blind/Unguided)")
        print("-" * 110)

        print("[1/7] BFS (Breadth-First Search)...")
        bfs = BFS(self.prob.grid, self.prob.start_node, self.prob.end_node)
        bfs.solve()
        self.results['BFS'] = self._collect_metrics(bfs.get_results(), "BFS", is_graph=True)
        print(self.results['BFS']['summary'])

        print("[2/7] DFS (Depth-First Search)...")
        dfs = DFS(self.prob.grid, self.prob.start_node, self.prob.end_node)
        dfs.solve()
        self.results['DFS'] = self._collect_metrics(dfs.get_results(), "DFS", is_graph=True)
        print(self.results['DFS']['summary'])

        print("[3/7] UCS (Uniform Cost Search)...")
        ucs = UCS(self.prob.grid, self.prob.start_node, self.prob.end_node)
        ucs.solve()
        self.results['UCS'] = self._collect_metrics(ucs.get_results(), "UCS", is_graph=True)
        print(self.results['UCS']['summary'])

        # ===== CLASSIC INFORMED SEARCH (Heuristic-Guided) =====
        print("\nCLASSIC INFORMED SEARCH (Heuristic-Guided)")
        print("-" * 110)

        print("[4/7] Greedy Best-First Search...")
        greedy = GreedyBestFirst(self.prob.grid, self.prob.start_node, self.prob.end_node)
        greedy.solve()
        self.results['Greedy'] = self._collect_metrics(greedy.get_results(), "Greedy Best-First", is_graph=True)
        print(self.results['Greedy']['summary'])

        print("[5/7] A* Search...")
        astar = AStarSearch(self.prob.grid, self.prob.start_node, self.prob.end_node)
        astar.solve()
        self.results['A*'] = self._collect_metrics(astar.get_results(), "A* Search", is_graph=True)
        print(self.results['A*']['summary'])

        # Set A* as optimal reference
        self.optimal_path_length = self.results['A*']['best_fitness']

        # ===== METAHEURISTIC ALGORITHMS =====
        print("\nMETAHEURISTIC ALGORITHMS (Population/Agent-Based)")
        print("-" * 110)

        print("[6/7] GA_Grid (Genetic Algorithm)...")
        ga = GA_Grid(self.prob.grid, self.prob.start_node, self.prob.end_node,
                     pop_size=30, max_iter=100)
        ga.solve()
        self.results['GA_Grid'] = self._collect_metrics(ga.get_results(), "GA_Grid", is_graph=False,
                                                       is_metaheuristic=True, pop_size=30)
        print(self.results['GA_Grid']['summary'])

        print("[7/7] ACO_Grid (Ant Colony Optimization)...")
        aco = ACO_Grid(self.prob.grid, self.prob.start_node, self.prob.end_node,
                       n_ants=20, max_iterations=100)
        aco.solve()
        self.results['ACO_Grid'] = self._collect_metrics(aco.get_results(), "ACO_Grid", is_graph=False,
                                                        is_metaheuristic=True, pop_size=20)
        print(self.results['ACO_Grid']['summary'])

    def _collect_metrics(self, res, name, is_graph=False, is_metaheuristic=False, pop_size=None):
        """Collect comprehensive metrics for each algorithm."""
        metrics = {
            'name': name,
            'best_fitness': float(res['best_fitness']),
            'execution_time': res['execution_time_seconds'],
        }

        # === COMPUTATIONAL COMPLEXITY ===
        if is_graph:
            nodes_expanded = res.get('nodes_expanded', 0)
            metrics['nodes_expanded'] = nodes_expanded
            metrics['space_complexity'] = nodes_expanded  # nodes stored in memory
            metrics['expansion_ratio'] = nodes_expanded / self.grid_size if self.grid_size > 0 else 0
            search_type = "Uninformed"
        elif is_metaheuristic:
            # For metaheuristics, approximate space complexity by population size * iterations
            iterations = res.get('convergence_curve', np.array([])).shape[0]
            metrics['nodes_expanded'] = pop_size * iterations  # approximate nodes evaluated
            metrics['space_complexity'] = pop_size  # active population in memory
            metrics['expansion_ratio'] = (pop_size * iterations) / self.grid_size if self.grid_size > 0 else 0
            search_type = "Metaheuristic"

        # === SOLUTION QUALITY (Optimality) ===
        if self.optimal_path_length:
            suboptimality = metrics['best_fitness'] - self.optimal_path_length
            optimality_gap_pct = (suboptimality / self.optimal_path_length * 100) if self.optimal_path_length > 0 else 0
            metrics['optimality_gap'] = suboptimality
            metrics['optimality_gap_pct'] = optimality_gap_pct
        else:
            metrics['optimality_gap'] = 0
            metrics['optimality_gap_pct'] = 0

        # === EXPLORATION VS EXPLOITATION ===
        if is_metaheuristic and 'convergence_curve' in res:
            conv_curve = res['convergence_curve']
            # Check if algorithm converged (curve plateaued)
            if len(conv_curve) > 10:
                final_10pct = conv_curve[-len(conv_curve)//10:]
                early_10pct = conv_curve[:len(conv_curve)//10]
                convergence_improvement = early_10pct.mean() - final_10pct.mean()
            else:
                convergence_improvement = 0
            metrics['convergence_improvement'] = convergence_improvement
            metrics['search_strategy'] = "Adaptive (exploration→exploitation)"
        else:
            # For graph search, lower expansion_ratio = better exploitation (guided)
            if 'Uninformed' in search_type:
                metrics['search_strategy'] = "Unguided (full exploration)"
            else:
                metrics['search_strategy'] = "Heuristic-guided (directed exploration)"

        # === SUMMARY STRING ===
        if is_graph:
            summary = (f"  Path length: {metrics['best_fitness']:.0f} | "
                      f"Nodes expanded: {nodes_expanded} | "
                      f"Exp. ratio: {metrics['expansion_ratio']:.1%} | "
                      f"Time: {metrics['execution_time']:.4f}s")
        elif is_metaheuristic:
            summary = (f"  Path length: {metrics['best_fitness']:.0f} | "
                      f"Nodes evaluated: {metrics['nodes_expanded']} | "
                      f"Pop size: {pop_size} | "
                      f"Time: {metrics['execution_time']:.4f}s")

        if self.optimal_path_length and metrics['optimality_gap'] > 0:
            summary += f" | Subopt: +{metrics['optimality_gap']:.0f} steps ({metrics['optimality_gap_pct']:.1f}%)"

        metrics['summary'] = summary + "\n"

        return metrics

    def print_analysis(self):
        """Print comprehensive analysis comparing all algorithms."""
        if not self.results:
            print("No results to analyze. Run test_all() first.")
            return

        # ===== 1. COMPUTATIONAL COMPLEXITY ANALYSIS =====
        print(f"\n{'='*110}")
        print("1. COMPUTATIONAL COMPLEXITY: Memory & Search Space")
        print(f"{'='*110}")
        print(f"{'Algorithm':<20} {'Nodes Expanded/Eval':<20} {'Space Complexity':<20} {'Exploration Ratio':<15} {'Strategy':<25}")
        print("-" * 110)

        sorted_by_complexity = sorted(self.results.items(),
                                     key=lambda x: x[1]['space_complexity'])
        for name, metrics in sorted_by_complexity:
            exp_ratio = f"{metrics['expansion_ratio']:.1%}"
            print(f"{metrics['name']:<20} {metrics['nodes_expanded']:<20} "
                  f"{metrics['space_complexity']:<20} {exp_ratio:<15} {metrics['search_strategy']:<25}")

        # ===== 2. SOLUTION QUALITY ANALYSIS =====
        print(f"\n{'='*110}")
        print("2. SOLUTION QUALITY: Optimality Analysis")
        print(f"{'='*110}")
        print(f"Optimal path length (A*): {self.optimal_path_length:.0f} steps\n")
        print(f"{'Algorithm':<20} {'Path Length':<15} {'Optimality Gap':<20} {'% Suboptimal':<15} {'Classification':<25}")
        print("-" * 110)

        sorted_by_quality = sorted(self.results.items(),
                                   key=lambda x: x[1]['best_fitness'])
        for name, metrics in sorted_by_quality:
            gap_str = f"+{metrics['optimality_gap']:.0f}" if metrics['optimality_gap'] > 0 else "Optimal"
            gap_pct = f"{metrics['optimality_gap_pct']:.1f}%" if metrics['optimality_gap'] > 0 else "0%"
            if metrics['optimality_gap'] == 0:
                classification = "Optimal"
            elif metrics['optimality_gap_pct'] <= 5:
                classification = "Near-Optimal (≤5%)"
            elif metrics['optimality_gap_pct'] <= 15:
                classification = "Good (5-15%)"
            else:
                classification = "Acceptable (>15%)"
            print(f"{metrics['name']:<20} {metrics['best_fitness']:<15.0f} {gap_str:<20} "
                  f"{gap_pct:<15} {classification:<25}")

        # ===== 3. EXPLORATION VS EXPLOITATION ANALYSIS =====
        print(f"\n{'='*110}")
        print("3. EXPLORATION VS EXPLOITATION: Search Strategy Comparison")
        print(f"{'='*110}")
        print("\nBlind Search Algorithms (Full Exploration):")
        print("-" * 110)
        print(f"  BFS:  {self.results['BFS']['expansion_ratio']:.1%} of grid (exhaustive level-by-level)")
        print(f"  DFS:  {self.results['DFS']['expansion_ratio']:.1%} of grid (exhaustive depth-first)")
        print(f"  UCS:  {self.results['UCS']['expansion_ratio']:.1%} of grid (cost-guided exhaustive)")

        avg_blind = np.mean([self.results['BFS']['expansion_ratio'],
                             self.results['DFS']['expansion_ratio'],
                             self.results['UCS']['expansion_ratio']])
        print(f"  AVG:  {avg_blind:.1%} of grid\n")

        print("Heuristic-Guided Search (Directed Exploration):")
        print("-" * 110)
        greedy_exp = self.results['Greedy']['expansion_ratio']
        astar_exp = self.results['A*']['expansion_ratio']
        greedy_dist = self.results['Greedy']['optimality_gap_pct']
        astar_dist = self.results['A*']['optimality_gap_pct']

        print(f"  Greedy: {greedy_exp:.1%} of grid explored | Solution: +{greedy_dist:.1f}% suboptimal")
        print(f"  A*:     {astar_exp:.1%} of grid explored | Solution: {astar_dist:.1f}% (OPTIMAL)")
        improvement = ((greedy_exp - astar_exp) / greedy_exp * 100) if greedy_exp > 0 else 0
        print(f"  → A* reduces exploration by {improvement:.0f}% vs Greedy\n")

        print("Metaheuristic Algorithms (Adaptive Population Search):")
        print("-" * 110)
        if 'GA_Grid' in self.results:
            ga_exp = self.results['GA_Grid']['expansion_ratio']
            ga_qual = self.results['GA_Grid']['optimality_gap_pct']
            print(f"  GA_Grid:  {ga_exp:.1%} nodes evaluated | Solution: +{ga_qual:.1f}% suboptimal")
        if 'ACO_Grid' in self.results:
            aco_exp = self.results['ACO_Grid']['expansion_ratio']
            aco_qual = self.results['ACO_Grid']['optimality_gap_pct']
            print(f"  ACO_Grid: {aco_exp:.1%} nodes evaluated | Solution: +{aco_qual:.1f}% suboptimal")

        # ===== 4. FINAL RANKING =====
        print(f"\n{'='*110}")
        print("4. OVERALL RANKING")
        print(f"{'='*110}")
        print(f"{'Rank':<6} {'Algorithm':<20} {'Path Length':<15} {'Nodes Expanded':<20} {'Time (s)':<12} {'Efficiency':<15}")
        print("-" * 110)

        # Score: minimize (path_length + nodes_expanded/100 + time*1000)
        ranked = []
        for name, metrics in self.results.items():
            score = metrics['best_fitness'] + metrics['nodes_expanded']/100.0 + metrics['execution_time']*10.0
            ranked.append((name, metrics, score))

        ranked.sort(key=lambda x: x[2])
        for rank, (name, metrics, score) in enumerate(ranked, 1):
            efficiency = (self.optimal_path_length / metrics['best_fitness']) * 100 if self.optimal_path_length > 0 else 0
            print(f"{rank:<6} {metrics['name']:<20} {metrics['best_fitness']:<15.0f} "
                  f"{metrics['nodes_expanded']:<20} {metrics['execution_time']:<12.4f} {efficiency:<15.0f}%")

        print("\nNotes:")
        print("  • Efficiency = (Optimal Path / Algorithm Path) * 100  [100% = optimal]")
        print("  • Exploration ratio = Nodes expanded / Grid size")
        print("  • Lower nodes expanded = better space complexity")
        print("  • A* provides the gold standard: optimal solution with minimal exploration")


if __name__ == "__main__":
    benchmark = AlgorithmBenchmark('tests/SP/test_1.txt')
    benchmark.run_all()
    benchmark.print_analysis()
