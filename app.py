import json
from dataclasses import asdict

from flask import Flask, request

from simulation import generate_intersection_graph, TrafficGrid, Intersection

app = Flask(__name__)

intersection_graph = generate_intersection_graph()


@app.route('/fetch_simulation', methods=['POST'])
def fetch_simulation():
    p: int = request.json.get('p')  # 'p' parameter for HLLSketch
    b: int = request.json.get('b')  # 'b' parameter for PCSA

    grid = TrafficGrid(intersection_graph, init_p=p, init_b=b)

    path_json = request.json.get('path')

    intersection_path = []
    for intersection_json in path_json:
        intersection_path.append(Intersection(street1=intersection_json.get('street1'),
                                              street2=intersection_json.get('street2')))

    grid.insert_path('AA-AA-AA', intersection_path)

    hll_path_found = grid.find_path_for_plate('AA-AA-AA', 'HLL')
    hll_spurious_intersections = len(hll_path_found.difference(intersection_path))

    hll_path_found_result = {
        'spurious_intersections': hll_spurious_intersections,
        'path_found': [],
    }

    for intersection in hll_path_found:
        hll_path_found_result['path_found'].append(asdict(intersection))

    pcsa_path_found = grid.find_path_for_plate('AA-AA-AA', 'PCSA')

    pcsa_spurious_intersections = len(pcsa_path_found.difference(intersection_path))

    pcsa_path_found_result = {
        'spurious_intersections': pcsa_spurious_intersections,
        'path_found': [],
    }

    for intersection in pcsa_path_found:
        pcsa_path_found_result['path_found'].append(asdict(intersection))

    flow_carrera: int = request.json.get('flow_carrera')
    flow_prat: int = request.json.get('flow_prat')
    flow_random_walk: int = request.json.get('flow_random_walk')

    grid.generate_static_flow_on_street('carrera', flow_carrera)
    grid.generate_static_flow_on_street('prat', flow_prat)
    grid.generate_random_flow(flow_random_walk)

    all_intersections = grid.all_intersections()

    intersections_cardinality = []

    for intersection in all_intersections:
        new_intersection_cardinality = asdict(intersection)
        new_intersection_cardinality['real_cardinality'] = grid.real_cardinality_for_intersection(intersection)
        new_intersection_cardinality['hll_cardinality'] = grid.hll_cardinality_for_intersection(intersection)
        new_intersection_cardinality['pcsa_cardinality'] = grid.pcsa_cardinality_for_intersection(intersection)
        intersections_cardinality.append(new_intersection_cardinality)

    result = {
        'cardinality': intersections_cardinality,
        'pcsa_path_found': pcsa_path_found_result,
        'hll_path_found': hll_path_found_result,
    }

    return result


@app.route('/')
def index():
    pass


if __name__ == '__main__':
    app.run()