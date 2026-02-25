from pixsorter.vision.grouping import connected_components

def test_connected_components_basic():
    edges = {
        0: [1],
        1: [0, 2],
        2: [1],
        3: [4],
        4: [3],
    }
    comps = connected_components(edges, 6)
    comps = [sorted(c) for c in comps]
    comps = sorted(comps, key=lambda x: (len(x), x))

    assert comps == [[5], [3, 4], [0, 1, 2]]