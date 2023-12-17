from cardinality_estimation import PCSASketch, SketchFlipMerge, HLLSketch


def test_pcsa_estimate():
    pcsa = PCSASketch(b=8)

    for i in range(1000):
        pcsa.add(str(i))

    assert pcsa.estimate() == 1012


def test_hyperloglog_estimate():
    hll = HLLSketch(p=14)

    for i in range(10000):
        hll.add(str(i))

    assert hll.estimate() == 9976


def test_sfm_estimate():
    sketch_flip_merge = SketchFlipMerge(b=9, p=.85)

    for i in range(10000):
        sketch_flip_merge.add(str(i))

    assert 7500 < sketch_flip_merge.estimate() < 10000

    sketch_flip_merge = SketchFlipMerge(b=8, p=.9)

    for i in range(10000):
        sketch_flip_merge.add(str(i))

    assert 7500 < sketch_flip_merge.estimate() < 10000

    sketch_flip_merge = SketchFlipMerge(b=10, p=1)

    for i in range(10000):
        sketch_flip_merge.add(str(i))

    assert 7500 < sketch_flip_merge.estimate() < 10000
