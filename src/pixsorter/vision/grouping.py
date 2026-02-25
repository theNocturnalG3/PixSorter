from collections import deque


def connected_components(edges: dict[int, list[int]], n: int) -> list[list[int]]:
    seen = [False] * n
    groups: list[list[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        q = deque([i])
        seen[i] = True
        comp = [i]
        while q:
            cur = q.popleft()
            for nb in edges.get(cur, []):
                if not seen[nb]:
                    seen[nb] = True
                    q.append(nb)
                    comp.append(nb)
        groups.append(comp)
    return groups