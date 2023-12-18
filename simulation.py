import random
import string
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from functools import partial
from typing import Optional, Callable, Iterable
import time

import mmh3
from matplotlib import pyplot as plt

from cardinality_estimation import HLLSketch, PCSASketch, SketchFlipMerge, CardinalityEstimator
from utils import generate_random_car_plate


@dataclass
class Path:
    car_plate: str
    path: list[tuple[str, str]]


@dataclass
class Camera:
    hll_sketch: HLLSketch
    pcsa_sketch: PCSASketch
    sketch_flip_merge: SketchFlipMerge

    _seen: set[str] = field(default_factory=set)

    @property
    def hll_sketch_count(self):
        return self.hll_sketch.estimate()

    @property
    def pcsa_sketch_count(self):
        return self.pcsa_sketch.estimate()

    @property
    def sfm_count(self):
        return self.sketch_flip_merge.estimate()

    @property
    def real_count(self):
        return len(self._seen)

    def increment_count(self, car_plate: str):
        self._seen.add(car_plate)
        self.hll_sketch.add(car_plate)
        self.pcsa_sketch.add(car_plate)
        self.sketch_flip_merge.add(car_plate)


@dataclass(frozen=True)
class Intersection:
    street1: str
    street2: str

    def has_street(self, street):
        return street in (self.street1, self.street2)

    def __hash__(self):
        return hash(frozenset((self.street1, self.street2)))

    def __eq__(self, other):
        return ((self.street1 == other.street1 and self.street2 == other.street2)
                or (self.street2 == other.street1 and self.street1 == other.street2))


def generate_intersection_graph() -> dict[Intersection]:
    horizontal_streets = list(reversed(['anibal_pinto', 'caupolican', 'rengo', 'lincoyan', 'angol', 'salas', 'serrano', 'prat']))
    horizontal_streets_direction = list(reversed(['right', 'left', 'right', 'left', 'right', 'left', 'right', 'left']))
    vertical_streets = ['rozas', 'heras', 'carrera', 'maipu', 'freire', 'barros', 'ohiggins']
    vertical_streets_direction = ['down', 'up', 'up_down', 'up', 'down', 'up', 'up_down']

    intersection_graph = dict()

    for i, vertical_street in enumerate(vertical_streets):
        for j, horizontal_street in enumerate(horizontal_streets):
            new_intersection = Intersection(vertical_street, horizontal_street)
            intersection_graph[new_intersection] = []

            if i > 0 and horizontal_streets_direction[j] == 'left':
                lef_intersection = Intersection(vertical_streets[i - 1], horizontal_street)
                intersection_graph[new_intersection].append(lef_intersection)

            if i < len(vertical_streets) - 1 and horizontal_streets_direction[j] == 'right':
                rig_intersection = Intersection(vertical_streets[i + 1], horizontal_street)
                intersection_graph[new_intersection].append(rig_intersection)

            if j > 0 and vertical_streets_direction[i] in ('down', 'up_down'):
                lower_intersection = Intersection(vertical_street, horizontal_streets[j - 1])
                intersection_graph[new_intersection].append(lower_intersection)

            if j < len(horizontal_streets) - 1 and vertical_streets_direction[i] in ('up', 'up_down'):
                upper_intersection = Intersection(vertical_street, horizontal_streets[j + 1])
                intersection_graph[new_intersection].append(upper_intersection)

    return intersection_graph


class TrafficGrid:
    def __init__(
            self,
            intersection_graph: dict[Intersection, list[Intersection]],
            sketch: CardinalityEstimator,
    ):
        self.intersection_graph = intersection_graph

        self._sketch = sketch

        self._intersection_to_sketch_map: dict[Intersection, CardinalityEstimator] = {}
        self._intersection_to_count_map: dict[Intersection, set] = {}

        for intersection in self.intersection_graph.keys():
            self._intersection_to_sketch_map[intersection] = deepcopy(self._sketch)
            self._intersection_to_count_map[intersection] = set()

        self._already_added = set()

    def get_frontiers(self) -> set[Intersection]:
        frontiers = set()

        for intersection, neighbours in self.intersection_graph.items():
            if len(neighbours) < 4:
                frontiers.add(intersection)

        return frontiers

    def all_intersections(self) -> list[Intersection]:
        intersections = []

        for intersection in self.intersection_graph.keys():
            intersections.append(intersection)

        return intersections

    def all_intersections_including_street(self, street: str) -> list[Intersection]:
        intersections = []

        for intersection in self.intersection_graph.keys():
            if intersection.has_street(street):
                intersections.append(intersection)

        return intersections

    @staticmethod
    def is_dead_end(node: Intersection, graph: dict, visited):
        # If for the next node, all of its neighbours were already visited before, we'll probably be stuck at a dead-end

        next_node_adjacent_nodes = set(graph[node])
        return next_node_adjacent_nodes.intersection(visited) == next_node_adjacent_nodes

    def has_a_viable_neighbour(self, node: Intersection, visited) -> bool:
        neighbours = self.intersection_graph[node]

        for intersection in neighbours:
            if intersection not in visited and self.intersection_graph[intersection]:
                return True

        return False

    def generate_random_walk(self,
                             graph: dict,
                             starting_node: Intersection,
                             ending_condition: Callable,
                             min_path_len: int = 3) -> list[Intersection]:
        current_path_len = 0

        # We'll want to avoid having cycles...
        already_visited: set[Intersection] = set()
        traversed_intersections: list[Intersection] = []

        current_node = starting_node

        # We'll want to have at least a couple of intersections per walk
        while (current_path_len < min_path_len or not ending_condition(current_node)):
            next_node = random.choice(graph[current_node])

            # Prevent a cycles and dead-ends
            while next_node in already_visited or TrafficGrid.is_dead_end(next_node, graph, already_visited):
                neighbouring_nodes = len(graph[current_node])
                visited_neighbouring_nodes = len(set(graph[current_node]).intersection(already_visited))

                if neighbouring_nodes - visited_neighbouring_nodes < 2:
                    return traversed_intersections

                if self.has_a_viable_neighbour(current_node, already_visited):
                    return traversed_intersections

                next_node = random.choice(graph[current_node])

            already_visited.add(current_node)
            traversed_intersections.append(current_node)
            current_node = next_node

        return traversed_intersections

    def generate_random_flow(self, cardinality: int, exclude=None):
        """ Generates """

        frontiers = self.get_frontiers()

        for _ in range(cardinality):
            new_car_plate = generate_random_car_plate(exclude=exclude)
            # Start from a border vertex that has a non-zero amount of adjacent nodes
            starting_intersection = random.choice([intersection for intersection in frontiers if self.intersection_graph[intersection]])

            random_walk = self.generate_random_walk(graph=self.intersection_graph,
                                                    starting_node=starting_intersection,
                                                    ending_condition=lambda current_node: current_node in frontiers)

            for intersection in random_walk:
                self._intersection_to_count_map[intersection].add(new_car_plate)

                sketch = self._intersection_to_sketch_map[intersection]
                sketch.add(new_car_plate)

    def generate_static_flow_on_street(self, street: str, cardinality: int, exclude=None):
        intersections = self.all_intersections_including_street(street)

        for _ in range(cardinality):
            new_car_plate = generate_random_car_plate(exclude=exclude)

            for intersection in intersections:
                self._intersection_to_count_map[intersection].add(new_car_plate)

                sketch = self._intersection_to_sketch_map[intersection]
                sketch.add(new_car_plate)

    def insert_path(self, car_plate: str, intersections: list[Intersection]):
        for intersection in intersections:
            self._intersection_to_count_map[intersection].add(car_plate)

            sketch = self._intersection_to_sketch_map[intersection]
            sketch.add(car_plate)

    def real_cardinality_for_intersection(self, intersection: Intersection) -> int:
        return len(self._intersection_to_count_map.get(intersection))

    def estimate_cardinality_for_intersection(self, intersection: Intersection) -> int:
        return self._intersection_to_sketch_map.get(intersection).estimate()

    def run_simulation(self, known_paths: list[Path], random_flow_parameters, static_flow_parameters):
        pass

    def find_path_for_plate(self, car_plate: str):
        path = set()

        for intersection in self.intersection_graph.keys():
            sketch_copy = deepcopy(self._intersection_to_sketch_map.get(intersection))

            cardinality_before_add = sketch_copy.estimate()

            sketch_copy.add(car_plate)

            cardinality_after_add = sketch_copy.estimate()

            if cardinality_after_add == cardinality_before_add:
                path.add(intersection)

        return path

    def find_path_using_hll_optimized(self, car_plate: str):
        if not isinstance(self._sketch, HLLSketch):
            raise ValueError(f'Only valid for self.sketch=HLLSketch, found self._sketch={type(self._sketch)}')

        path = set()

        for intersection in self.intersection_graph.keys():
            sketch: HLLSketch = self._intersection_to_sketch_map.get(intersection)

            h = mmh3.hash64(car_plate, 1, False)[0]

            bucket = sketch.get_bucket(h)
            value = sketch.get_value(h)

            leading_zeros = sketch.leading_zeros(value)

            if sketch.M[bucket] >= leading_zeros + 1:
                path.add(intersection)

        return path

    def find_path_using_pcsa_optimized(self, car_plate: str):
        if not isinstance(self._sketch, PCSASketch):
            raise ValueError(f'Only valid for self.sketch=PCSASketch, found self._sketch={type(self._sketch)}')

        path = set()

        for intersection in self.intersection_graph.keys():
            sketch: PCSASketch = self._intersection_to_sketch_map.get(intersection)

            h = mmh3.hash(car_plate, 1, False)

            index = sketch.get_index(h)
            value = sketch.get_leading_zeroes(h)

            if sketch.M[index] & (1 << value):
                path.add(intersection)

        return path


def compare_pcsa_sfm(iters: int = 2):
    path = [
        Intersection('prat', 'heras'),
        Intersection('heras', 'serrano'),
        Intersection('serrano', 'carrera'),
        Intersection('serrano', 'maipu'),
        Intersection('serrano', 'freire'),
        Intersection('salas', 'freire'),
    ]
    
    results_pcsa = []
    results_fsm = []

    mean_spur_pcsa = []
    mean_spur_sfm = []

    mae_pcsa = []
    mae_sfm = []

    for b in range(4, 10):
        print(f'Current b: {b}')

        time_pcsa = 0
        time_sfm = 0

        pcsa_spur = 0
        sfm_spur = 0

        pcsa_error = 0
        sfm_error = 0

        for i in range(iters):
            print(f'Current iter: {i}')

            grid_pcsa = TrafficGrid(intersection_graph=generate_intersection_graph(),
                                    sketch=PCSASketch(b=b))
            grid_pcsa.generate_random_flow(3000)
            grid_pcsa.insert_path('AA-AA-AA', path)

            print(f'Finding path for pcsa')

            start = time.time()
            path_found = grid.find_path_using_pcsa_optimized('AA-AA-AA')
            time_pcsa += time.time() - start

            pcsa_spur += len(path_found.difference(path))

            print(f'Finding path for fsm')

            grid_sfm = TrafficGrid(intersection_graph=generate_intersection_graph(),
                                   sketch=SketchFlipMerge(b=b, p=.85))
            grid_sfm.generate_random_flow(3000)
            grid_sfm.insert_path('AA-AA-AA', path)

            start = time.time()
            path_found = grid_sfm.find_path_for_plate('AA-AA-AA')
            time_sfm += time.time() - start

            sfm_spur += len(path_found.difference(path))

            intersection = grid.all_intersections_including_street('salas')[0]

            real_cardinality = grid_pcsa.real_cardinality_for_intersection(intersection)
            pcsa_estimation = grid_pcsa.estimate_cardinality_for_intersection(intersection)
            sfm_estimation = grid_sfm.estimate_cardinality_for_intersection(intersection)

            pcsa_error = abs(pcsa_estimation - real_cardinality)
            sfm_error = abs(sfm_estimation - real_cardinality)

        mae_pcsa.append( (b, pcsa_error / iters) )
        mae_sfm.append( (b, sfm_error / iters) )

        mean_spur_pcsa.append( (b, pcsa_spur / iters) )
        mean_spur_sfm.append( (b, sfm_spur / iters) )

        results_pcsa.append( (b, time_pcsa / iters) )
        results_fsm.append( (b, time_sfm / iters) )

    with open('time_pcsa.dat', 'w') as f:
        for b, t in results_pcsa:
            f.write(f'{b} {t}\n')

    with open('time_sfm.dat', 'w') as f:
        for b, t in results_fsm:
            f.write(f'{b} {t}\n')

    with open('error_pcsa.dat', 'w') as f:
        for b, mae in mae_pcsa:
            f.write(f'{b} {mae}\n')

    with open('error_sfm.dat', 'w') as f:
        for b, mae in mae_sfm:
            f.write(f'{b} {mae}\n')

    with open('mean_spur_pcsa.dat', 'w') as f:
        for b, spur in mean_spur_pcsa:
            f.write(f'{b} {spur}\n')

    with open('mean_spur_sfm.dat', 'w') as f:
        for b, spur in mean_spur_sfm:
            f.write(f'{b} {spur}\n')


def intersections_as_grid(intersections: Iterable[Intersection]):
    horizontal_streets = list(reversed(['anibal_pinto', 'caupolican', 'rengo', 'lincoyan', 'angol', 'salas', 'serrano', 'prat']))
    vertical_streets = ['rozas', 'heras', 'carrera', 'maipu', 'freire', 'barros', 'ohiggins']

    points = []

    for i, vertical_street in enumerate(vertical_streets):
        for j, horizontal_street in enumerate(horizontal_streets):
            if Intersection(vertical_street, horizontal_street) in intersections:
                points.append((i, j))

    return points


def plot_pcsa(points_real, points_pcsa):
    plt.plot([x for x, _ in points_pcsa], [y for _, y in points_pcsa], marker='o', linestyle='')
    plt.plot([x for x, _ in points_real], [y for _, y in points_real], marker='o', linestyle='', alpha=.6)
    plt.legend(['PCSA b=9', 'Real'])
    plt.grid(True)
    plt.xlim([0, 7])
    plt.ylim([0, 6])
    plt.savefig('plot_pcsa.png')
    plt.close()


def plot_sfm(points_real, points_sfm):
    plt.plot([x for x, _ in points_sfm], [y for _, y in points_sfm], marker='o', linestyle='')
    plt.plot([x for x, _ in points_real], [y for _, y in points_real], marker='o', linestyle='', alpha=.6)
    plt.legend(['SFM b=9 p=0.85', 'Real'])
    plt.grid(True)
    plt.xlim([0, 7])
    plt.ylim([0, 6])
    plt.savefig('plot_sfm.png')
    plt.close()


def plot_hll(points_real, points_hll):
    plt.plot([p for p, _ in points_hll], [spur for _, spur in points_hll], marker='o', linestyle='')
    plt.plot([p for p, _ in points_real], [spur for _, spur in points_real], marker='o', linestyle='', alpha=.6)
    plt.legend(['HLL p=12', 'Real'])
    plt.grid(True)
    plt.xlim([0, 7])
    plt.ylim([0, 6])
    plt.savefig('plot_hll.png')
    plt.close()


def plot_path(path_points):
    plt.plot([x for x, _ in path_points], [y for _, y in path_points], marker='o', linestyle='')
    plt.grid(True)
    plt.xlim([0, 7])
    plt.ylim([0, 6])
    plt.savefig('path.png')
    plt.close()



if __name__ != '__main__':
    graph = generate_intersection_graph()
    print(graph[Intersection('rozas', 'angol')])

if __name__ == '__main__':
    grid_hll = TrafficGrid(generate_intersection_graph(),
                           sketch=HLLSketch(p=14))
    grid_pcsa = TrafficGrid(generate_intersection_graph(),
                            sketch=PCSASketch(b=9))
    grid_sfm = TrafficGrid(generate_intersection_graph(),
                           sketch=SketchFlipMerge(b=9, p=.85))

    exclude_plates = frozenset(['AA-AA-07', 'AA-AA-05', 'AA-AR-82', 'AB-KN-67', 'BC-HM-68'])

    for grid in (grid_hll, grid_pcsa, grid_sfm):
        grid.generate_static_flow_on_street('carrera', 8000, exclude=exclude_plates)
        grid.generate_static_flow_on_street('prat', 2500, exclude=exclude_plates)
        grid.generate_random_flow(2500, exclude=exclude_plates)

    intersection = grid_hll.all_intersections_including_street('salas')[0]

    path = [
        Intersection('prat', 'heras'),
        Intersection('heras', 'serrano'),
        Intersection('serrano', 'carrera'),
        Intersection('serrano', 'maipu'),
        Intersection('serrano', 'freire'),
        Intersection('salas', 'freire'),
    ]

    plate_to_find = 'BC-HM-68'

    grid_sfm.insert_path(plate_to_find, path)
    grid_hll.insert_path(plate_to_find, path)
    grid_pcsa.insert_path(plate_to_find, path)

    path_found = grid_sfm.find_path_for_plate(plate_to_find)
    spurious_intersections = path_found.difference(path)
    print('path found (SFM)', path_found)
    print(f'spurious intersections ({len(spurious_intersections)}, SFM) {spurious_intersections}')

    optimized_hll_path_found = grid_hll.find_path_using_hll_optimized(plate_to_find)
    spurious_intersections = optimized_hll_path_found.difference(path)
    print('path found (hll, optimized)', optimized_hll_path_found)
    print(f'spurious intersections ({len(spurious_intersections)}, HLL, optimized) {spurious_intersections}')

    optimized_pcsa_path_found = grid_pcsa.find_path_using_pcsa_optimized(plate_to_find)
    spurious_intersections = optimized_pcsa_path_found.difference(path)
    print('path found (PCSA, optimized): ', optimized_pcsa_path_found)
    print(f'spurious intersections ({len(spurious_intersections)}, PCSA, optimized) {spurious_intersections}')

    plot_pcsa(points_real=intersections_as_grid(path),
              points_pcsa=intersections_as_grid(optimized_pcsa_path_found))

    plot_hll(points_real=intersections_as_grid(path),
             points_hll=intersections_as_grid(optimized_hll_path_found))

    plot_sfm(points_real=intersections_as_grid(path),
             points_sfm=intersections_as_grid(path_found))

    print('real cardinality', grid_hll.real_cardinality_for_intersection(intersection))
    print('hll estimation', grid_hll.estimate_cardinality_for_intersection(intersection))
    print('pcsa estimation', grid_pcsa.estimate_cardinality_for_intersection(intersection))
    print('sfm estimation', grid_sfm.estimate_cardinality_for_intersection(intersection))
    print('carrera-serrano cardinality', grid_hll.real_cardinality_for_intersection(Intersection('carrera', 'serrano')))

