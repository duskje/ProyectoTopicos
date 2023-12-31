import random
import string
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from functools import partial
from typing import Optional, Callable, Iterable, Tuple, TypeAlias, Union
import time

import mmh3
from matplotlib import pyplot as plt

from analysis import find_plates_until_n_leading_zeros_pcsa
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


def generate_intersection_graph() -> Tuple[dict[Intersection], set[Intersection]]:
    horizontal_streets = list(reversed(['anibal_pinto', 'caupolican', 'rengo', 'lincoyan', 'angol', 'salas', 'serrano', 'prat']))
    horizontal_streets_direction = list(reversed(['right', 'left', 'right', 'left', 'right', 'left', 'right', 'left']))
    vertical_streets = ['rozas', 'heras', 'carrera', 'maipu', 'freire', 'barros', 'ohiggins']
    vertical_streets_direction = ['down', 'up', 'up_down', 'up', 'down', 'up', 'up_down']

    intersection_graph = dict()
    borders = set()

    for i, vertical_street in enumerate(vertical_streets):
        for j, horizontal_street in enumerate(horizontal_streets):
            new_intersection = Intersection(vertical_street, horizontal_street)
            intersection_graph[new_intersection] = []

            if i == 0 or j == 0:
                borders.add(new_intersection)

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

    return intersection_graph, borders


def has_a_viable_neighbour(graph, vertex: Intersection, visited: set[Intersection]) -> bool:
    neighbours = graph[vertex]

    for intersection in neighbours:
        if intersection not in visited and graph[intersection]:
            return True

    return False


def generate_random_walk(graph: dict,
                         starting_node: Intersection,
                         ending_condition: Callable,
                         min_path_len: int = 3) -> list[Intersection]:
    current_path_len = 0

    # We'll want to avoid having cycles...
    already_visited: set[Intersection] = set()
    traversed_intersections: list[Intersection] = []

    current_node = starting_node

    # We'll want to have at least a couple of intersections per walk
    while current_path_len < min_path_len or not ending_condition(current_node):
        next_node = random.choice(graph[current_node])

        # Prevent a cycles and dead-ends
        while next_node in already_visited or TrafficGrid.is_dead_end(next_node, graph, already_visited):
            neighbouring_nodes = len(graph[current_node])
            visited_neighbouring_nodes = len(set(graph[current_node]).intersection(already_visited))

            if neighbouring_nodes - visited_neighbouring_nodes < 2:
                return traversed_intersections

            if has_a_viable_neighbour(graph, current_node, already_visited):
                return traversed_intersections

            next_node = random.choice(graph[current_node])

        already_visited.add(current_node)
        traversed_intersections.append(current_node)
        current_node = next_node

    return traversed_intersections


def generate_random_paths(graph, borders, cardinality: int, exclude=None) -> list[Tuple[str, list[Intersection]]]:
    """ Generates """
    paths = []

    for _ in range(cardinality):
        new_car_plate = generate_random_car_plate(exclude=exclude)
        # Start from a border vertex that has a non-zero amount of adjacent nodes
        starting_intersection = random.choice([intersection for intersection in graph.keys() if graph[intersection]])

        random_walk = generate_random_walk(graph=graph,
                                           starting_node=starting_intersection,
                                           ending_condition=lambda current_node: current_node in borders)

        paths.append((new_car_plate, random_walk))

    return paths


class TrafficGrid:
    def __init__(self,
                 intersection_graph: dict[Intersection, list[Intersection]],
                 borders: set[Intersection],
                 sketch: Callable[[...], CardinalityEstimator],
                 sketch_kwargs: dict[str, Union[int, float]]):
        self.intersection_graph = intersection_graph
        self.borders = borders

        self._sketch = sketch

        self._intersection_to_sketch_map: dict[Intersection, CardinalityEstimator] = {}
        self._intersection_to_count_map: dict[Intersection, set] = {}

        for intersection in self.intersection_graph.keys():
            self._intersection_to_sketch_map[intersection] = sketch(**sketch_kwargs)
            self._intersection_to_count_map[intersection] = set()

        self._already_added = set()

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
        if self._sketch.__name__ != 'HLLSketch':
            raise ValueError(f'Only valid for self.sketch=HLLSketch, found self._sketch={self._sketch.__name__}')

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
        if self._sketch.__name__ != 'PCSASketch':
            raise ValueError(f'Only valid for self.sketch=HLLSketch, found self._sketch={self._sketch.__name__}')

        path = set()

        for intersection in self.intersection_graph.keys():
            sketch: PCSASketch = self._intersection_to_sketch_map.get(intersection)

            h = mmh3.hash(car_plate, 1, False)

            index = sketch.get_index(h)
            value = sketch.get_leading_zeroes(h)

            if sketch.M[index] & (1 << value):
                path.add(intersection)

        return path


def compare_pcsa_sfm(iters: int = 5):
    intersection_graph, borders = generate_intersection_graph()

    exclude_plates = frozenset(['AA-AA-07', 'AA-AA-05', 'AA-AR-82', 'AB-KN-67', 'BC-HM-68', 'CC-HY-80'])

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

        plates = find_plates_until_n_leading_zeros_pcsa(n=8, b=b)

        time_pcsa = 0
        time_sfm = 0

        pcsa_spur = 0
        sfm_spur = 0

        pcsa_error = 0
        sfm_error = 0

        for i in range(iters):
            print(f'Current iter: {i}')
            plate_to_find = plates[8]

            grid_pcsa = TrafficGrid(intersection_graph,
                                    borders=borders,
                                    sketch=PCSASketch,
                                    sketch_kwargs={'b': b})

            grid_pcsa.insert_path(plate_to_find, path)

            grid_sfm = TrafficGrid(intersection_graph,
                                   borders=borders,
                                   sketch=SketchFlipMerge,
                                   sketch_kwargs={'b': b, 'p': .85})

            grid_sfm.insert_path(plate_to_find, path)

            for plate, path in generate_random_paths(intersection_graph, borders, 3000, exclude_plates):
                # print('Adding path', path)

                grid_pcsa.insert_path(plate, path)
                grid_sfm.insert_path(plate, path)

            print(f'Finding path for pcsa')
            start = time.time()
            path_found = grid_pcsa.find_path_using_pcsa_optimized(plate_to_find)
            time_pcsa += time.time() - start

            pcsa_spur += len(path_found.difference(path))

            print(f'Finding path for fsm')
            start = time.time()
            path_found = grid_sfm.find_path_for_plate(plate_to_find)
            time_sfm += time.time() - start

            sfm_spur += len(path_found.difference(path))

            for intersection in grid_pcsa.all_intersections():
                real_cardinality = grid_pcsa.real_cardinality_for_intersection(intersection)
                pcsa_estimation = grid_pcsa.estimate_cardinality_for_intersection(intersection)
                sfm_estimation = grid_sfm.estimate_cardinality_for_intersection(intersection)

                pcsa_error += abs(pcsa_estimation - real_cardinality)
                sfm_error += abs(sfm_estimation - real_cardinality)

        mae_pcsa.append( (b, pcsa_error / iters) )
        mae_sfm.append( (b, sfm_error / iters) )

        mean_spur_pcsa.append( (b, pcsa_spur / iters) )
        mean_spur_sfm.append( (b, sfm_spur / iters) )

        results_pcsa.append( (b, time_pcsa / iters) )
        results_fsm.append( (b, time_sfm / iters) )

    with open('plot/time_pcsa.dat', 'w') as f:
        for b, t in results_pcsa:
            f.write(f'{b} {t}\n')

    with open('plot/time_sfm.dat', 'w') as f:
        for b, t in results_fsm:
            f.write(f'{b} {t}\n')

    with open('plot/error_pcsa.dat', 'w') as f:
        for b, mae in mae_pcsa:
            f.write(f'{b} {mae}\n')

    with open('plot/error_sfm.dat', 'w') as f:
        for b, mae in mae_sfm:
            f.write(f'{b} {mae}\n')

    with open('plot/mean_spur_pcsa.dat', 'w') as f:
        for b, spur in mean_spur_pcsa:
            f.write(f'{b} {spur}\n')

    with open('plot/mean_spur_sfm.dat', 'w') as f:
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


if __name__ == '__main__':
    intersection_graph, borders = generate_intersection_graph()

    grid_hll = TrafficGrid(intersection_graph,
                           borders=borders,
                           sketch=HLLSketch,
                           sketch_kwargs={'p': 14})

    grid_pcsa = TrafficGrid(intersection_graph,
                            borders=borders,
                            sketch=PCSASketch,
                            sketch_kwargs={'b': 9})

    grid_sfm = TrafficGrid(intersection_graph,
                           borders=borders,
                           sketch=SketchFlipMerge,
                           sketch_kwargs={'b': 9, 'p': .9})

    exclude_plates = frozenset(['AA-AA-07', 'AA-AA-05', 'AA-AR-82', 'AB-KN-67', 'BC-HM-68', 'CC-HY-80'])

    # Cardinalidad de recorridos aleatorios
    random_cardinality = 4000

    for plate, path in generate_random_paths(intersection_graph, borders, random_cardinality, exclude_plates):
        print('Adding path', path)

        grid_hll.insert_path(plate, path)
        grid_pcsa.insert_path(plate, path)
        grid_sfm.insert_path(plate, path)

    for grid in (grid_hll, grid_pcsa, grid_sfm):
        grid.generate_static_flow_on_street('carrera', 3000, exclude=exclude_plates)
        grid.generate_static_flow_on_street('prat', 2500, exclude=exclude_plates)

    intersection = Intersection('serrano', 'maipu')

    path = [
        Intersection('prat', 'heras'),
        Intersection('heras', 'serrano'),
        Intersection('serrano', 'carrera'),
        Intersection('serrano', 'maipu'),
        Intersection('serrano', 'freire'),
        Intersection('salas', 'freire'),
    ]

    # 'CC-HY-80' 20 ldz pcsa
    # 'BC-HM-68' 20 ldz hll
    plate_to_find = 'CC-HY-80'

    grid_sfm.insert_path(plate_to_find, path)
    grid_hll.insert_path(plate_to_find, path)
    grid_pcsa.insert_path(plate_to_find, path)

    sfm_path_found = grid_sfm.find_path_for_plate(plate_to_find)
    spurious_intersections = sfm_path_found.difference(path)
    print('path found (SFM)', sfm_path_found)
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
             points_sfm=intersections_as_grid(sfm_path_found))

    print('real cardinality', grid_hll.real_cardinality_for_intersection(intersection))
    print('hll estimation', grid_hll.estimate_cardinality_for_intersection(intersection))
    print('pcsa estimation', grid_pcsa.estimate_cardinality_for_intersection(intersection))
    print('sfm estimation', grid_sfm.estimate_cardinality_for_intersection(intersection))
    print('carrera-serrano cardinality', grid_hll.real_cardinality_for_intersection(Intersection('carrera', 'serrano')))

