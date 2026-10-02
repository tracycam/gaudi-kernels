"""Independent staged integer specification for the metadata producer."""
def reference(ids, experts, rows, capacity):
    tokens, routes = len(ids), len(ids[0])
    flat = [x for row in ids for x in row]
    counts = [flat.count(e) for e in range(experts)]
    row_status = [int(any(x < 0 or x >= experts for x in row)) |
                  (2 if len(set(row)) != routes else 0) for row in ids]
    flags = 0
    for f in row_status:
        flags |= f
    if sum(counts) != tokens * routes or any(n < 0 or n > tokens for n in counts):
        flags |= 8
    tiles = sum((n + rows - 1) // rows for n in counts)
    if tiles > capacity:
        flags |= 4
    prefix = [0] * (experts + 1)
    expert = [-1] * capacity
    base = [0] * capacity
    valid = [0] * capacity
    row_map = [-1] * (rows * capacity)
    inverse = [-1] * len(flat)
    cursor = 0
    if not flags:
        for e, n in enumerate(counts):
            prefix[e] = cursor
            for start in range(0, n, rows):
                expert[cursor], base[cursor], valid[cursor] = e, start, min(rows, n-start)
                cursor += 1
        prefix[-1] = cursor
        for e in range(experts):
            ordinal = 0
            for q, value in enumerate(flat):
                if value == e:
                    target = prefix[e] * rows + ordinal
                    ordinal += 1
                    row_map[target], inverse[q] = q, target
    return dict(counts=counts, row_status=row_status, prefix=prefix,
                tile_expert=expert, tile_base=base, valid_rows=valid,
                status=[flags], row_map=row_map, inverse=inverse)
