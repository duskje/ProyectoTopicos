from cardinality_estimation import PCSASketch


def test_fsm_sketch():
    fsm_sketch = PCSASketch(32)
    fsm_sketch.add('hola')
    print(fsm_sketch.estimate())
