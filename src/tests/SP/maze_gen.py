import os
import random
import numpy as np

from pathlib import Path

ROOT = Path(__file__).resolve().parent

SMALL_SIZE = 60
LARGE_SIZE = 150

def write_grid(grid, path):
    n, m = grid.shape
    with open(path, "w") as f:
        f.write(f"{n} {m}\n")
        for row in grid:
            f.write(" ".join(map(str, row)) + "\n")


def place_start_end(grid):
    n, m = grid.shape

    while True:
        s = (random.randint(1,n-2), random.randint(1,m-2))
        if grid[s] == 0:
            break

    while True:
        e = (random.randint(1,n-2), random.randint(1,m-2))
        if grid[e] == 0 and e != s:
            break

    grid[s] = "S"
    grid[e] = "E"


def blank_grid(size):
    grid = np.zeros((size,size), dtype=object)
    grid[0,:] = 1
    grid[-1,:] = 1
    grid[:,0] = 1
    grid[:,-1] = 1
    place_start_end(grid)
    return grid


def obstacle_grid(size, density):
    grid = np.zeros((size,size), dtype=object)

    grid[0,:] = 1
    grid[-1,:] = 1
    grid[:,0] = 1
    grid[:,-1] = 1

    for i in range(1,size-1):
        for j in range(1,size-1):
            if random.random() < density:
                grid[i,j] = 1

    grid[grid == 0] = 0
    place_start_end(grid)
    return grid


# Recursive backtracking maze generator
def perfect_maze(size):

    if size % 2 == 0:
        size += 1

    grid = np.ones((size,size), dtype=object)

    stack = [(1,1)]
    grid[1,1] = 0

    directions = [(2,0),(-2,0),(0,2),(0,-2)]

    while stack:
        x,y = stack[-1]
        neighbors = []

        for dx,dy in directions:
            nx,ny = x+dx,y+dy
            if 1 <= nx < size-1 and 1 <= ny < size-1:
                if grid[nx,ny] == 1:
                    neighbors.append((nx,ny,dx,dy))

        if neighbors:
            nx,ny,dx,dy = random.choice(neighbors)
            grid[x+dx//2,y+dy//2] = 0
            grid[nx,ny] = 0
            stack.append((nx,ny))
        else:
            stack.pop()

    place_start_end(grid)
    return grid


def maze_with_loops(size):

    grid = perfect_maze(size)

    n,m = grid.shape

    # remove some walls to create loops
    loops = int((n*m)*0.02)

    for _ in range(loops):
        x = random.randint(1,n-2)
        y = random.randint(1,m-2)

        if grid[x,y] == 1:
            grid[x,y] = 0

    return grid


def generate_case(folder, count, generator):

    os.makedirs(folder, exist_ok=True)

    for i in range(count):

        if i == 0:
            size = SMALL_SIZE
        else:
            size = LARGE_SIZE

        grid = generator(size)

        path = os.path.join(folder, f"test_{i}.txt")
        write_grid(grid, path)

        print("Generated:", path)


def main():

    generate_case(
        f"{ROOT}/blank",
        10,
        blank_grid
    )

    generate_case(
        f"{ROOT}/obstacles",
        20,
        lambda s: obstacle_grid(s,0.20)
    )

    generate_case(
        f"{ROOT}/obstacles_dense",
        20,
        lambda s: obstacle_grid(s,0.35)
    )

    generate_case(
        f"{ROOT}/perfect_maze",
        20,
        perfect_maze
    )

    generate_case(
        f"{ROOT}/maze_loops",
        20,
        maze_with_loops
    )


if __name__ == "__main__":
    main()