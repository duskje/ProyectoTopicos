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

from cardinality_estimation import HLLSketch, PCSASketch, SketchFlipMerge


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
    vertical_streets = ['anibal_pinto', 'capuolican', 'rengo', 'lincoyan', 'angol', 'salas', 'serrano', 'prat']
    horizontal_streets = ['rozas', 'heras', 'carrera', 'maipu', 'freire', 'barros', 'ohiggins']

    intersection_graph = dict()

    for i, vertical_street in enumerate(vertical_streets):
        for j, horizontal_street in enumerate(horizontal_streets):
            new_intersection = Intersection(vertical_street, horizontal_street)
            intersection_graph[new_intersection] = []

            if i > 0:
                upper_intersection = Intersection(vertical_streets[i - 1], horizontal_street)
                intersection_graph[new_intersection].append(upper_intersection)

            if i < len(vertical_streets) - 1:
                lower_intersection = Intersection(vertical_streets[i + 1], horizontal_street)
                intersection_graph[new_intersection].append(lower_intersection)

            if j > 0:
                left_intersection = Intersection(vertical_street, horizontal_streets[j - 1])
                intersection_graph[new_intersection].append(left_intersection)

            if j < len(horizontal_streets) - 1:
                right_intersection = Intersection(vertical_street, horizontal_streets[j + 1])
                intersection_graph[new_intersection].append(right_intersection)

    return intersection_graph


def generate_random_car_plate():
    result = ''
    result += str(random.choice(string.ascii_uppercase))
    result += str(random.choice(string.ascii_uppercase))
    result += '-'
    result += str(random.choice(string.ascii_uppercase))
    result += str(random.choice(string.ascii_uppercase))
    result += '-'
    result += str(random.choice(list(range(10))))
    result += str(random.choice(list(range(10))))
    return result


@dataclass(frozen=True)
class StaticFlow:
    cardinality: int
    street: str


@dataclass(frozen=True)
class RandomFlow:
    cardinality: int


class TrafficGrid:
    def __init__(
            self,
            intersection_graph: dict[Intersection, list[Intersection]],
            init_p: int = 14,
            init_b: int = 8,
            probability: float = 1
    ):
        self.intersection_graph = intersection_graph

        self._intersection_to_camera_map: dict[Intersection, Camera] = {}

        for intersection in self.intersection_graph.keys():
            self._intersection_to_camera_map[intersection] = Camera(HLLSketch(p=init_p),
                                                                    PCSASketch(b=init_b),
                                                                    SketchFlipMerge(b=init_b, p=probability))

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

    @staticmethod
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

                next_node = random.choice(graph[current_node])

            already_visited.add(current_node)
            traversed_intersections.append(current_node)
            current_node = next_node

        return traversed_intersections

    def generate_random_flow(self, cardinality: int):
        """ Generates """

        frontiers = self.get_frontiers()

        for _ in range(cardinality):
            new_car_plate = generate_random_car_plate()
            starting_intersection = random.choice(list(frontiers))  # Start at a random frontier intersection

            random_walk = self.generate_random_walk(graph=self.intersection_graph,
                                                    starting_node=starting_intersection,
                                                    ending_condition=lambda current_node: current_node in frontiers)

            for intersection in random_walk:
                camera = self._intersection_to_camera_map[intersection]
                camera.increment_count(new_car_plate)

    def generate_static_flow_on_street(self, street: str, cardinality: int):
        intersections = self.all_intersections_including_street(street)

        for _ in range(cardinality):
            new_car_plate = generate_random_car_plate()

            for intersection in intersections:
                camera = self._intersection_to_camera_map[intersection]
                camera.increment_count(new_car_plate)

    def insert_path(self, car_plate: str, intersections: list[Intersection]):
        for intersection in intersections:
            camera = self._intersection_to_camera_map[intersection]
            camera.increment_count(car_plate)

    def real_cardinality_for_intersection(self, intersection: Intersection) -> int:
        return self._intersection_to_camera_map.get(intersection).real_count

    def hll_cardinality_for_intersection(self, intersection: Intersection) -> int:
        return self._intersection_to_camera_map.get(intersection).hll_sketch_count

    def pcsa_cardinality_for_intersection(self, intersection: Intersection) -> int:
        return self._intersection_to_camera_map.get(intersection).pcsa_sketch_count

    def sfm_cardinality_for_intersection(self, intersection: Intersection) -> int:
        return self._intersection_to_camera_map.get(intersection).sfm_count

    def run_simulation(self, known_paths: list[Path], random_flow_parameters, static_flow_parameters):
        pass

    def find_path_for_plate(self, car_plate: str, sketch: str):
        path = set()

        for intersection in self.intersection_graph.keys():
            if sketch == 'HLL':
                sketch_copy = deepcopy(self._intersection_to_camera_map.get(intersection).hll_sketch)
            elif sketch == 'PCSA':
                sketch_copy = deepcopy(self._intersection_to_camera_map.get(intersection).pcsa_sketch)
            elif sketch == 'SFM':
                sketch_copy = deepcopy(self._intersection_to_camera_map.get(intersection).sketch_flip_merge)
            else:
                raise TypeError

            cardinality_before_add = sketch_copy.estimate()

            sketch_copy.add(car_plate)

            cardinality_after_add = sketch_copy.estimate()

            if cardinality_after_add == cardinality_before_add:
                path.add(intersection)

        return path

    def find_path_using_hll_optimized(self, car_plate: str):
        path = set()

        for intersection in self.intersection_graph.keys():
            sketch = self._intersection_to_camera_map.get(intersection).hll_sketch

            h = mmh3.hash64(car_plate, 1, False)[0]

            bucket = sketch.get_bucket(h)
            value = sketch.get_value(h)

            leading_zeros = sketch.leading_zeros(value)

            if sketch.M[bucket] >= leading_zeros + 1:
                path.add(intersection)

        return path

    def find_path_using_pcsa_optimized(self, car_plate: str):
        path = set()

        for intersection in self.intersection_graph.keys():
            sketch = self._intersection_to_camera_map.get(intersection).pcsa_sketch

            h = mmh3.hash(car_plate, 1, False)

            index = sketch.get_index(h)
            value = sketch.get_leading_zeroes(h)

            if sketch.M[index] & (1 << value):
                path.add(intersection)

        return path


def compare_pcsa_sfm(iters: int = 10):
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

            grid = TrafficGrid(generate_intersection_graph(), init_p=5, init_b=b, probability=0.85)
            grid.generate_random_flow(3000)
            grid.insert_path('AA-AA-AA', path)

            print(f'Finding path for pcsa')

            start = time.time()
            path_found = grid.find_path_using_pcsa_optimized('AA-AA-AA')
            time_pcsa += time.time() - start

            pcsa_spur += len(path_found.difference(path))

            print(f'Finding path for fsm')

            start = time.time()
            path_found = grid.find_path_for_plate('AA-AA-AA', 'SFM')
            time_sfm += time.time() - start

            sfm_spur += len(path_found.difference(path))

            intersection = grid.all_intersections_including_street('salas')[0]

            real_cardinality = grid.real_cardinality_for_intersection(intersection)
            pcsa_estimation = grid.pcsa_cardinality_for_intersection(intersection)
            sfm_estimation = grid.sfm_cardinality_for_intersection(intersection)

            pcsa_error = abs(pcsa_estimation - real_cardinality)
            sfm_error = abs(sfm_estimation - real_cardinality)

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


#if __name__ == '__main__':
#    compare_pcsa_sfm()


def intersections_as_grid(intersections: Iterable[Intersection]):
    vertical_streets = ['anibal_pinto', 'capuolican', 'rengo', 'lincoyan', 'angol', 'salas', 'serrano', 'prat']
    horizontal_streets = ['rozas', 'heras', 'carrera', 'maipu', 'freire', 'barros', 'ohiggins']

    points = []

    for i, vertical_street in enumerate(vertical_streets):
        for j, horizontal_street in enumerate(horizontal_streets):
            if Intersection(vertical_street, horizontal_street) in intersections:
                points.append((i, j))

    return points


def plot_pcsa(points_real, points_pcsa):
    plt.plot([p for p, _ in points_pcsa], [spur for _, spur in points_pcsa], marker='o', linestyle='')
    plt.plot([p for p, _ in points_real], [spur for _, spur in points_real], marker='o', linestyle='', alpha=.6)
    plt.legend(['PCSA b=9', 'Real'])
    plt.grid(True)
    plt.xlim([0, 7])
    plt.ylim([0, 6])
    plt.savefig('plot_pcsa.png')
    plt.close()

def plot_sfm(points_real, points_sfm):
    plt.plot([p for p, _ in points_sfm], [spur for _, spur in points_sfm], marker='o', linestyle='')
    plt.plot([p for p, _ in points_real], [spur for _, spur in points_real], marker='o', linestyle='', alpha=.6)
    plt.legend(['SFM b=9 p=0.8', 'Real'])
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
    plt.savefig('plot.hll.png')
    plt.close()


if __name__ == '__main__':
    grid = TrafficGrid(generate_intersection_graph(), init_p=12, init_b=9, probability=0.8)

    # print(grid.intersection_graph[Intersection('rozas', 'serrano')])
    # print(grid.all_intersections_including_street('rozas'))
    # print(generate_random_car_plate())
    # print(grid.get_frontiers())

    grid.generate_static_flow_on_street('carrera', 15000)
    grid.generate_static_flow_on_street('prat', 5000)
    intersection = grid.all_intersections_including_street('salas')[0]
    grid.generate_random_flow(3000)

    path = [
        Intersection('prat', 'heras'),
        Intersection('heras', 'serrano'),
        Intersection('serrano', 'carrera'),
        Intersection('serrano', 'maipu'),
        Intersection('serrano', 'freire'),
        Intersection('salas', 'freire'),
    ]

#    import json
#    json_path = {'path': []}
#    for json_intersection in (asdict(intersection) for intersection in path):
#        json_path['path'].append(json_intersection)
#
#    print(json.dumps(json_path, indent=2))

    grid.insert_path('AA-AA-AA', path)

    #path_found = grid.find_path_for_plate('AA-AA-AA', 'HLL')
    #spurious_intersections = path_found.difference(path)
    #print('path found (hll): ', path_found)
    #print(f'spurious intersections ({len(spurious_intersections)}, HLL) {spurious_intersections}')

    #path_found = grid.find_path_for_plate('AA-AA-AA', 'PCSA')
    #spurious_intersections = path_found.difference(path)
    #print('path found (PCSA): ', path_found)
    #print(f'spurious intersections ({len(spurious_intersections)}, PCSA) {spurious_intersections}')

    path_found = grid.find_path_for_plate('AA-AA-AA', 'SFM')
    spurious_intersections = path_found.difference(path)
    print('path found (SFM)', path_found)
    print(f'spurious intersections ({len(spurious_intersections)}, SFM) {spurious_intersections}')

    optimized_hll_path_found = grid.find_path_for_plate('AA-AA-AA', 'HLL')
    spurious_intersections = optimized_hll_path_found.difference(path)
    print('path found (hll, optimized)', optimized_hll_path_found)
    print(f'spurious intersections ({len(spurious_intersections)}, HLL, optimized) {spurious_intersections}')

    optimized_pcsa_path_found = grid.find_path_using_pcsa_optimized('AA-AA-AA')
    spurious_intersections = optimized_pcsa_path_found.difference(path)
    print('path found (PCSA, optimized): ', optimized_pcsa_path_found)
    print(f'spurious intersections ({len(spurious_intersections)}, PCSA, optimized) {spurious_intersections}')

    plot_pcsa(points_real=intersections_as_grid(path),
              points_pcsa=intersections_as_grid(optimized_pcsa_path_found))

    plot_hll(points_real=intersections_as_grid(path),
             points_hll=intersections_as_grid(optimized_hll_path_found))

    plot_sfm(points_real=intersections_as_grid(path),
             points_sfm=intersections_as_grid(path_found))

    print('hll estimation', grid.hll_cardinality_for_intersection(intersection))
    print('real cardinality', grid.real_cardinality_for_intersection(intersection))
    print('pcsa estimation', grid.pcsa_cardinality_for_intersection(intersection))
    print('sfm estimation', grid.sfm_cardinality_for_intersection(intersection))
    print('carrera-serrano cardinality', grid.real_cardinality_for_intersection(Intersection('carrera', 'serrano')))

