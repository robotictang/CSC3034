"""Fast regression tests for the dependency-light lab algorithms."""

from __future__ import annotations

import os
from pathlib import Path
import random
import sys
import unittest

import numpy as np


os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "files"))

import isaac_ga_robot  # noqa: E402
import lab3_ga  # noqa: E402
import lab4_pso  # noqa: E402
import lab5_aco  # noqa: E402


class GeneticAlgorithmTests(unittest.TestCase):
    def test_binary_encoding_clamps_to_bit_range(self):
        self.assertEqual(lab3_ga.value2binary(-2, bits=6), "000000")
        self.assertEqual(lab3_ga.value2binary(100, bits=6), "111111")

    def test_mutation_repairs_problem_domain(self):
        random.seed(3)
        for _ in range(50):
            value = lab3_ga.mutate(63, p_mutation=1.0)
            self.assertGreaterEqual(value, 1)
            self.assertLessEqual(value, 10)

    def test_paper_fitness_is_total_used_area(self):
        self.assertEqual(lab3_ga.calculateFitness(5), 300.0)
        self.assertEqual(lab3_ga.calculateFitness(20), 0.0)

    def test_segment_collision_checks_between_waypoints(self):
        original = isaac_ga_robot.OBSTACLES
        try:
            isaac_ga_robot.OBSTACLES = [(0.0, 0.0, 1.0)]
            length, collisions = isaac_ga_robot.evaluate_trajectory(
                [np.array([-2.0, 0.0]), np.array([2.0, 0.0])]
            )
        finally:
            isaac_ga_robot.OBSTACLES = original
        self.assertAlmostEqual(length, 4.0)
        self.assertEqual(collisions, 1)


class ParticleSwarmTests(unittest.TestCase):
    def test_seeded_run_finds_global_basin(self):
        random.seed(1)
        _, best, history, _ = lab4_pso.run_gbest_pso(
            alpha=[0.5, 0.5], n_particle=20, inertia_weight=0.7
        )
        self.assertAlmostEqual(best, -84.1585, places=2)
        self.assertLessEqual(lab4_pso.fit_fcn(history[-1]), lab4_pso.fit_fcn(history[0]))


class AntColonyTests(unittest.TestCase):
    def setUp(self):
        self.cities, self.roads = lab5_aco.build_romania_network()

    def test_constructed_paths_do_not_revisit_cities(self):
        np.random.seed(5)
        ant = lab5_aco.Ant()
        ant.get_path(self.cities["Arad"], self.cities["Bucharest"])
        names = [city.name for city in ant.cities]
        self.assertEqual(len(names), len(set(names)))

    def test_known_shortest_route_cost(self):
        route = ["Arad", "Sibiu", "Rimnicu Vilcea", "Pitesti", "Bucharest"]
        cost_by_edge = {
            frozenset((road.connected_cities[0].name, road.connected_cities[1].name)): road.cost
            for road in self.roads
        }
        cost = sum(cost_by_edge[frozenset(pair)] for pair in zip(route, route[1:]))
        self.assertEqual(cost, 418.0)

    def test_failed_ants_do_not_inflate_dominance(self):
        destination = self.cities["Bucharest"]
        ants = [lab5_aco.Ant() for _ in range(10)]
        ants[0].cities = [self.cities["Arad"], destination]
        self.assertEqual(lab5_aco.get_percentage_of_dominant_path(ants, destination), 0.1)


if __name__ == "__main__":
    unittest.main()
