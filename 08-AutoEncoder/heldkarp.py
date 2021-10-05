import itertools
import random
import sys
import math
import numpy as np
import cv2


def held_karp(dists):
    """
    Implementation of Held-Karp, an algorithm that solves the Traveling
    Salesman Problem using dynamic programming with memoization.

    Parameters:
        dists: distance matrix

    Returns:
        A tuple, (cost, path).
    """
    n = len(dists)

    # Maps each subset of the nodes to the cost to reach that subset, as well
    # as what node it passed before reaching this subset.
    # Node subsets are represented as set bits.
    C = {}

    # Set transition cost from initial state
    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)

    # Iterate subsets of increasing length and store intermediate results
    # in classic dynamic programming manner
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            # Set bits for all nodes in this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find the lowest cost to get to this subset
            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + dists[m][k], m))
                C[(bits, k)] = min(res)

    # We're interested in all bits but the least significant (the start state)
    bits = (2**n - 1) - 1

    # Calculate optimal cost
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dists[k][0], k))
    opt, parent = min(res)

    # Backtrack to find full path
    path = []
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    # Add implicit start state
    path.append(0)

    return opt, list(reversed(path))


def generate_distances(coords):
    n = len(coords)
    dists = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            pointA = coords[i]
            pointB = coords[j]
            x_dist = (pointA[0] - pointB[0])**2
            y_dist = (pointA[1] - pointB[1])**2
            dist = math.sqrt(x_dist + y_dist)
            dists[i][j] = dists[j][i] = dist

    return dists

def generate_coordinates(n):
    coords = []
    for _ in range(n):
        while True:
            x = random.randint(1, 99)
            y = random.randint(1, 99)
            coord = (x, y)
            if coord not in coords:
                coords.append(coord)
                break
    return coords


def read_distances(filename):
    dists = []
    with open(filename, 'rb') as f:
        for line in f:
            # Skip comments
            if line[0] == '#':
                continue

            dists.append(map(int, map(str.strip, line.split(','))))

    return dists

def draw_dots(img, coords):
    for coord in coords:
        y, x = coord
        # OpenCV has BGR instead of RGB
        img[x, y] = (1)
    return img


def draw_path(img, coords, path):
    for i in range(1, len(path)):
        a = coords[path[i]]
        b = coords[path[i-1]]
        img = cv2.line(img, a, b, (2), 1)
    a = coords[path[0]]
    b = coords[path[-1]]
    img = cv2.line(img, a, b, (2), 1)
    return img

def draw_sol(img, coords, path):
    img = draw_path(img, coords, path)
    img = draw_dots(img, coords)
    return img

def generate_pair(n):
    coords = generate_coordinates(n)
    dists = generate_distances(coords)
    tsp_sol = held_karp(dists)
    inp = np.zeros((100, 100), np.uint8)
    inp = draw_dots(inp, coords)
    out = np.zeros((100, 100), np.uint8)
    out = draw_sol(out, coords, tsp_sol[1])
    return inp, out

random.seed(1)
if __name__ == '__main__':
    arg = sys.argv[1]

    if arg.endswith('.csv'):
        dists = read_distances(arg)
    else:
        for i in range(100):
            coords = generate_coordinates(int(arg))
            dists = generate_distances(coords)
            tsp_sol = held_karp(dists)
            img = np.ones((100, 100, 3), np.uint8)
            img *= 255
            img = draw_dots(img, coords)
            cv2.imwrite("inputs/img%d.png" %i, img)
            img = draw_sol(img, coords, tsp_sol[1])
            cv2.imwrite("outputs/img%d.png" %i, img)


    # Pretty-print the distance matrix
    for row in dists:
        print(''.join([str(int(n)).rjust(3, ' ') for n in row]))

    print('')

    print(tsp_sol)
