import math
import random
import time
import gc
import os
import asyncio
import shutil
from asyncio import to_thread
from datetime import datetime
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from asyncio import Semaphore
from functools import partial
import numpy as np
import numba as nb
from scipy.spatial import Voronoi
from skimage.morphology import skeletonize, binary_erosion
from noise import snoise2
from scipy.ndimage import label, binary_dilation
from scipy.signal import convolve2d
from skimage.morphology import thin, disk
from PIL import Image
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patheffects import withStroke
from matplotlib.patches import Patch, FancyBboxPatch, Circle, Rectangle
from heapq import heappush, heappop

# Add this line to switch to a non-interactive backend
plt.switch_backend("Agg")

# Define the URL for the Montserrat font (Regular weight)
font_url = "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Regular.ttf"
font_filename = "Montserrat-Regular.ttf"
font_path = os.path.join("fonts", font_filename)

# Create a fonts directory if it doesn't exist
os.makedirs("fonts", exist_ok=True)

os.nice(22)  # Increase the niceness value to lower the priority


# Function to download the font
def download_font(url, path):
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    with open(path, "wb") as f:
        f.write(response.content)


# Download the font if it doesn't exist locally or is corrupted
try:
    if not os.path.isfile(font_path) or os.path.getsize(font_path) == 0:
        print("Downloading Montserrat font...")
        download_font(font_url, font_path)
        print("Font downloaded.")
except requests.exceptions.RequestException as e:
    print(f"Error downloading the font: {e}")
    raise

# Verify that the font is a valid TrueType font
try:
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Montserrat"
    print("Font loaded and set.")
except RuntimeError as e:
    print(f"Error loading font: {e}")
    raise

# Constants for integer coordinate encoding
BITS_PER_COORDINATE = int(math.floor(math.log2((1 << 63) - 1) / 2))

# Constants for float coordinate encoding
BITS_PER_FLOAT_SIGNIFICAND = 24
BITS_PER_FLOAT_EXPONENT = BITS_PER_COORDINATE - 1 + BITS_PER_FLOAT_SIGNIFICAND
MOST_FLOAT_COORDINATE = (1 - (1 / (1 << BITS_PER_FLOAT_SIGNIFICAND))) * (
    1 << (BITS_PER_FLOAT_EXPONENT - 1)
)
LEAST_FLOAT_COORDINATE = -MOST_FLOAT_COORDINATE


class PriorityQueue:
    def __init__(self, items=None, priorities=None):
        if items is None:
            items = []
        if priorities is None:
            priorities = []
        self.items = items
        self.priorities = priorities
        self.size = len(items)
        self.capacity = len(items)

    def is_empty(self):
        return self.size == 0

    def pop(self):
        if self.is_empty():
            raise EmptyQueueError()
        min_item = self.items[0]
        self.size -= 1
        if self.size > 0:
            last_item = self.items[self.size]
            self.items[0] = last_item
            self.priorities[0] = self.priorities[self.size]
            self._heapify(0)
        return min_item

    def insert(self, item, priority):
        if self.size >= self.capacity:
            self._grow()
        if self.size < len(self.items):
            self.items[self.size] = item
            self.priorities[self.size] = priority
        else:
            self.items.append(item)
            self.priorities.append(priority)
        self._improve_key(self.size)
        self.size += 1

    def _grow(self):
        new_size = self.new_capacity(self.capacity)
        self.capacity = new_size
        self.items.extend([None] * (new_size - len(self.items)))
        self.priorities.extend([float("inf")] * (new_size - len(self.priorities)))

    @staticmethod
    def new_capacity(current_capacity):
        return current_capacity + (current_capacity >> 1)

    def _heapify(self, i):
        smallest = i
        l = 2 * i + 1  # noqa: E741
        r = 2 * i + 2

        if l < self.size and self.priorities[l] < self.priorities[smallest]:
            smallest = l
        if r < self.size and self.priorities[r] < self.priorities[smallest]:
            smallest = r

        if smallest != i:
            self.items[i], self.items[smallest] = self.items[smallest], self.items[i]
            self.priorities[i], self.priorities[smallest] = (
                self.priorities[smallest],
                self.priorities[i],
            )
            self._heapify(smallest)

    def _improve_key(self, i):
        while i > 0 and self.priorities[(i - 1) // 2] > self.priorities[i]:
            parent = (i - 1) // 2
            self.items[i], self.items[parent] = self.items[parent], self.items[i]
            self.priorities[i], self.priorities[parent] = (
                self.priorities[parent],
                self.priorities[i],
            )
            i = parent


class EmptyQueueError(Exception):
    """Raised when an operation depends on a non-empty queue."""

    pass


@nb.jit
def encode_integer_coordinates(x, y):
    return (x & ((1 << BITS_PER_COORDINATE) - 1)) | (y << BITS_PER_COORDINATE)


@nb.jit
def decode_integer_coordinates(value):
    mask = (1 << BITS_PER_COORDINATE) - 1
    x = value & mask
    y = value >> BITS_PER_COORDINATE
    return x, y


@nb.jit
def float_to_int(f):
    significand, exponent = math.frexp(f)
    significand = int(significand * (1 << BITS_PER_FLOAT_SIGNIFICAND))
    exponent = exponent + BITS_PER_FLOAT_SIGNIFICAND - 1
    result = (significand & ((1 << BITS_PER_FLOAT_SIGNIFICAND) - 1)) | (
        exponent << BITS_PER_FLOAT_SIGNIFICAND
    )
    return result if f >= 0 else -result


@nb.jit
def int_to_float(i):
    v = abs(i)
    significand = v & ((1 << BITS_PER_FLOAT_SIGNIFICAND) - 1)
    exponent = v >> BITS_PER_FLOAT_SIGNIFICAND
    return math.ldexp(
        significand / (1 << BITS_PER_FLOAT_SIGNIFICAND),
        exponent - BITS_PER_FLOAT_SIGNIFICAND + 1,
    ) * (1 if i >= 0 else -1)


@nb.jit
def encode_float_coordinates(x, y):
    x_int = float_to_int(x)
    y_int = float_to_int(y)
    return encode_integer_coordinates(x_int, y_int)


@nb.jit
def decode_float_coordinates(value):
    x_int, y_int = decode_integer_coordinates(value)
    x = int_to_float(x_int)
    y = int_to_float(y_int)
    return x, y


@nb.jit
def make_row_major_indexer(width, node_width=1, node_height=1):
    def indexer(x, y):
        return (y // node_height) * width + (x // node_width)

    return indexer


@nb.jit
def make_column_major_indexer(height, node_width=1, node_height=1):
    def indexer(x, y):
        return (x // node_width) * height + (y // node_height)

    return indexer


@nb.jit
def make_4_directions_enumerator(
    node_width=1, node_height=1, min_x=0, min_y=0, max_x=None, max_y=None
):
    max_x = max_x if max_x is not None else np.inf
    max_y = max_y if max_y is not None else np.inf

    def enumerator(x, y, func):
        for dx, dy in [
            (0, -node_height),
            (0, node_height),
            (-node_width, 0),
            (node_width, 0),
        ]:
            next_x = x + dx
            next_y = y + dy
            if min_x <= next_x < max_x and min_y <= next_y < max_y:
                func(next_x, next_y)

    return enumerator


@nb.jit
def make_8_directions_enumerator(
    node_width=1, node_height=1, min_x=0, min_y=0, max_x=None, max_y=None
):
    max_x = max_x if max_x is not None else np.inf
    max_y = max_y if max_y is not None else np.inf

    def enumerator(x, y, func):
        for dx, dy in [
            (node_width, 0),
            (node_width, -node_height),
            (0, -node_height),
            (-node_width, -node_height),
            (-node_width, 0),
            (-node_width, node_height),
            (0, node_height),
            (node_width, node_height),
        ]:
            next_x = x + dx
            next_y = y + dy
            if min_x <= next_x < max_x and min_y <= next_y < max_y:
                func(next_x, next_y)

    return enumerator


@nb.jit
def make_manhattan_distance_heuristic(scale_factor=1.0):
    def heuristic(x1, y1, x2, y2):
        return scale_factor * (abs(x1 - x2) + abs(y1 - y2))

    return heuristic


@nb.jit
def make_octile_distance_heuristic(scale_factor=1.0):
    sqrt2 = math.sqrt(2)

    def heuristic(x1, y1, x2, y2):
        return scale_factor * (
            min(abs(x1 - x2), abs(y1 - y2)) * sqrt2 + abs(abs(x1 - x2) - abs(y1 - y2))
        )

    return heuristic


@nb.jit
def make_euclidean_distance_heuristic(scale_factor=1.0):
    def heuristic(x1, y1, x2, y2):
        return scale_factor * math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    return heuristic


def define_path_finder(
    name,
    world_size,
    frontier_size=500,
    coordinate_type="float",
    coordinate_encoder=encode_float_coordinates,
    coordinate_decoder=decode_float_coordinates,
    indexer=None,
    goal_reached_p=None,
    neighbor_enumerator=None,
    exact_cost=None,
    heuristic_cost=None,
    max_movement_cost=float("inf"),
    path_initiator=lambda length: None,
    path_processor=lambda x, y: None,
    path_finalizer=lambda: True,
):
    @nb.jit
    def path_finder_core(
        start_x, start_y, goal_x, goal_y, cost_so_far, came_from, path
    ):
        frontier = []  # We'll use a list as a simple priority queue
        heappush(frontier, (0.0, coordinate_encoder(start_x, start_y)))
        start_index = indexer(start_x, start_y)
        goal_index = indexer(goal_x, goal_y)
        cost_so_far[start_index] = 0.0

        while frontier:
            _, current = heappop(frontier)
            current_x, current_y = coordinate_decoder(current)
            current_index = indexer(current_x, current_y)

            if goal_reached_p(current_x, current_y, goal_x, goal_y):
                break

            def process_neighbor(next_x, next_y):
                next_index = indexer(next_x, next_y)
                new_cost = cost_so_far[current_index] + exact_cost(
                    current_x, current_y, next_x, next_y
                )
                if new_cost < max_movement_cost and (
                    math.isnan(cost_so_far[next_index])
                    or new_cost < cost_so_far[next_index]
                ):
                    cost_so_far[next_index] = new_cost
                    came_from[next_index] = current
                    priority = new_cost + heuristic_cost(next_x, next_y, goal_x, goal_y)
                    heappush(frontier, (priority, coordinate_encoder(next_x, next_y)))

            neighbor_enumerator(current_x, current_y, process_neighbor)

        if math.isnan(cost_so_far[goal_index]):
            return 0  # Path not found

        length = 0
        current = goal_index
        while current != start_index:
            path[length] = current
            length += 1
            current_x, current_y = coordinate_decoder(came_from[current])
            current = indexer(current_x, current_y)
        path[length] = start_index
        length += 1

        return length

    def path_finder(start_x, start_y, goal_x, goal_y, **params):
        cost_so_far = np.full(world_size, np.nan)
        came_from = np.full(world_size, -1, dtype=np.int64)
        path = np.full(world_size, -1, dtype=np.int64)

        length = path_finder_core(
            start_x, start_y, goal_x, goal_y, cost_so_far, came_from, path
        )

        if length == 0:
            return None

        path_initiator(length)
        for i in range(length - 1, -1, -1):
            path_x, path_y = coordinate_decoder(path[i])
            path_processor(path_x, path_y)

        return path_finalizer()

    path_finder.__name__ = name
    return path_finder


@nb.jit
def is_maze_solvable(maze, start, goal, max_iterations=100000):
    queue = [(start[0], start[1])]
    visited = set([(start[0], start[1])])
    iterations = 0

    while queue and iterations < max_iterations:
        iterations += 1
        x, y = queue.pop(0)

        if (x, y) == goal:
            return True

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < maze.shape[1]
                and 0 <= ny < maze.shape[0]
                and maze[ny, nx] == 0
                and (nx, ny) not in visited
            ):
                queue.append((nx, ny))
                visited.add((nx, ny))

    return False


@nb.jit(nopython=True)
def create_dla_maze(width, height):
    maze = np.zeros((height, width), dtype=np.int32)
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1

    num_seeds = np.random.randint(
        max(3, min(width, height) // 20), max(10, min(width, height) // 5)
    )
    seed_positions = np.random.randint(
        1, min(width - 1, height - 1), size=(num_seeds, 2)
    )

    # Use a loop instead of advanced indexing
    for i in range(num_seeds):
        y, x = seed_positions[i]
        maze[y, x] = 1

    num_particles = np.random.randint(width * height // 16, width * height // 8)
    directions = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)])

    for _ in range(num_particles):
        x, y = np.random.randint(1, width - 1), np.random.randint(1, height - 1)
        steps = 0
        max_steps = np.random.randint(100, 1000)
        while maze[y, x] == 0 and steps < max_steps:
            dx, dy = directions[np.random.randint(4)]
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if maze[ny, nx] == 1:
                    maze[y, x] = 1
                    break
                x, y = nx, ny
            steps += 1

    # Ensure connectivity
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if maze[y, x] == 1 and np.random.random() < 0.1:
                maze[y, x] = 0

    return maze


def create_game_of_life_maze(width, height):
    p_alive = random.uniform(0.3, 0.7)
    maze = np.random.choice([0, 1], size=(height, width), p=[1 - p_alive, p_alive])
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1

    generations = random.randint(3, 10)
    for _ in range(generations):
        new_maze = maze.copy()
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                neighbors = maze[y - 1 : y + 2, x - 1 : x + 2].sum() - maze[y, x]
                if maze[y, x] == 1:
                    if neighbors < 2 or neighbors > 3:
                        new_maze[y, x] = 0
                else:
                    if neighbors == 3:
                        new_maze[y, x] = 1
        maze = new_maze

    return maze.astype(np.int32)


def create_one_dim_automata_maze(width, height):
    maze = np.zeros((height, width), dtype=np.int32)
    maze[0] = np.random.choice([0, 1], size=width)
    maze[0, 0] = maze[0, -1] = 1

    rule_number = random.randint(0, 255)
    rule = np.zeros((2, 2, 2), dtype=np.int32)
    for i in range(8):
        rule[i // 4, (i % 4) // 2, i % 2] = (rule_number >> i) & 1

    for y in range(1, height):
        for x in range(width):
            left = maze[y - 1, (x - 1) % width]
            center = maze[y - 1, x]
            right = maze[y - 1, (x + 1) % width]
            maze[y, x] = rule[left, center, right]

    maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
    return maze


@nb.jit
def create_langtons_ant_maze(width, height):
    maze = np.zeros((height, width), dtype=np.int32)
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1

    ant_x, ant_y = random.randint(1, width - 2), random.randint(1, height - 2)
    ant_direction = random.randint(0, 3)

    steps = random.randint(width * height // 4, width * height)
    for _ in range(steps):
        maze[ant_y, ant_x] = 1 - maze[ant_y, ant_x]
        if maze[ant_y, ant_x] == 1:
            ant_direction = (ant_direction + 1) % 4
        else:
            ant_direction = (ant_direction - 1) % 4

        if ant_direction == 0:
            ant_y = max(1, ant_y - 1)
        elif ant_direction == 1:
            ant_x = min(width - 2, ant_x + 1)
        elif ant_direction == 2:
            ant_y = min(height - 2, ant_y + 1)
        else:
            ant_x = max(1, ant_x - 1)

    return maze


def create_voronoi_maze(width, height):
    num_points = np.random.randint(max(width, height) // 3, max(width, height) // 2)
    points = np.random.rand(num_points, 2) * [width, height]
    vor = Voronoi(points)

    maze = np.ones((height, width), dtype=np.int32)

    maze = draw_lines(maze, vor.vertices, vor.ridge_vertices)

    # Apply erosion to create wider passages
    maze = binary_erosion(maze, np.ones((3, 3)))

    # Skeletonize to thin the passages
    maze = skeletonize(1 - maze).astype(np.int32)
    maze = 1 - maze

    # Ensure borders are walls
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1

    return maze


@nb.jit
def recursive_divide(maze, x, y, w, h, min_size=4):
    if w <= min_size or h <= min_size:
        return

    # Randomly decide whether to divide horizontally or vertically
    if w > h:
        divide_horizontally = np.random.random() < 0.8
    else:
        divide_horizontally = np.random.random() < 0.2

    if divide_horizontally:
        divide_at = np.random.randint(y + 1, y + h - 1)
        maze[divide_at, x : x + w] = 1
        opening = np.random.randint(x, x + w)
        maze[divide_at, opening] = 0
        recursive_divide(maze, x, y, w, divide_at - y, min_size)
        recursive_divide(maze, x, divide_at + 1, w, y + h - divide_at - 1, min_size)
    else:
        divide_at = np.random.randint(x + 1, x + w - 1)
        maze[y : y + h, divide_at] = 1
        opening = np.random.randint(y, y + h)
        maze[opening, divide_at] = 0
        recursive_divide(maze, x, y, divide_at - x, h, min_size)
        recursive_divide(maze, divide_at + 1, y, x + w - divide_at - 1, h, min_size)


def create_fractal_maze(width, height, min_size=4):
    maze = np.zeros((height, width), dtype=np.int32)
    recursive_divide(maze, 0, 0, width, height, min_size)
    return maze


def create_maze_from_image(width, height):
    image_path = ""  # Enter file path here
    # Load and resize the image
    img = Image.open(image_path).convert("L")
    img = img.resize((width, height))

    # Convert to numpy array and threshold
    maze = np.array(img)
    maze = (maze > 128).astype(int)

    # Ensure borders are walls
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
    return maze


@nb.jit(nopython=True)
def get_valid_tiles(maze, x, y, width, height, tiles):
    valid = np.ones(3, dtype=np.bool_)
    for dx, dy, direction in [(0, -1, 0), (1, 0, 1), (0, 1, 2), (-1, 0, 3)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            maze_value = maze[ny, nx]
            if 0 <= maze_value < 3:
                valid &= tiles[maze_value][direction]
            else:
                valid &= False
    return np.where(valid)[0]


@nb.jit(nopython=True)
def create_wave_function_collapse_maze_core(width, height, tiles, max_iterations):
    maze = np.full((height, width), -1, dtype=np.int32)
    stack = [(np.random.randint(1, width - 2), np.random.randint(1, height - 2))]

    iterations = 0
    while stack and iterations < max_iterations:
        idx = np.random.randint(0, len(stack))
        x, y = stack[idx]
        stack.pop(idx)

        if maze[y, x] == -1:
            valid_tiles = get_valid_tiles(maze, x, y, width, height, tiles)
            if len(valid_tiles) > 0:
                maze[y, x] = np.random.choice(valid_tiles)
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (
                        0 < nx < width - 1
                        and 0 < ny < height - 1
                        and maze[ny, nx] == -1
                    ):
                        stack.append((nx, ny))

        iterations += 1

    return maze


def create_wave_function_collapse_maze(width, height, timeout=30):
    tiles = np.array(
        [
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],  # Empty
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # Wall
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # Special
        ],
        dtype=np.bool_,
    )

    start_time = time.time()
    max_iterations = width * height * 10  # Adjust this factor as needed

    maze = create_wave_function_collapse_maze_core(width, height, tiles, max_iterations)

    # Fill any remaining -1 cells with random valid tiles
    for y in range(height):
        for x in range(width):
            if maze[y, x] == -1:
                valid_tiles = get_valid_tiles(maze, x, y, width, height, tiles)
                maze[y, x] = (
                    np.random.choice(valid_tiles) if len(valid_tiles) > 0 else 0
                )

    # Ensure borders are walls
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1

    elapsed_time = time.time() - start_time
    if elapsed_time > timeout:
        print(
            f"Warning: Wave Function Collapse maze generation exceeded timeout ({elapsed_time:.2f}s > {timeout}s)"
        )

    return maze


def create_growing_tree_maze(width, height):
    maze = np.ones((height, width), dtype=np.int32)
    stack = [(1, 1)]
    maze[1, 1] = 0

    directions = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)])

    while stack:
        if np.random.random() < 0.5:
            current = stack.pop(np.random.randint(0, len(stack)))
        else:
            current = stack.pop()

        x, y = current
        np.random.shuffle(directions)  # This is now using numpy's shuffle

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < width - 1 and 0 < ny < height - 1 and maze[ny, nx] == 1:
                maze[ny, nx] = 0
                stack.append((nx, ny))

    return maze


def generate_terrain(width, height, scale, octaves, persistence, lacunarity):
    terrain = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            terrain[y, x] = snoise2(
                x * scale,
                y * scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
            )
    return terrain


def create_terrain_based_maze(width, height):
    scale = np.random.uniform(0.05, 0.1)
    octaves = np.random.randint(4, 6)
    persistence = np.random.uniform(0.5, 0.7)
    lacunarity = np.random.uniform(2.0, 2.5)

    terrain = generate_terrain(width, height, scale, octaves, persistence, lacunarity)
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    maze = (terrain > np.percentile(terrain, 60)).astype(np.int32)

    # Apply erosion to create wider passages
    maze = binary_erosion(maze, np.ones((3, 3)))

    # Skeletonize to thin the passages
    maze = skeletonize(1 - maze).astype(np.int32)
    maze = 1 - maze

    # Ensure borders are walls
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1

    return maze


def create_musicalized_maze(width, height):
    frequencies = np.linspace(1, 10, num=width)
    time = np.linspace(0, 10, num=height)
    t, f = np.meshgrid(time, frequencies)

    harmonic1 = np.sin(2 * np.pi * f * t)
    harmonic2 = np.sin(3 * np.pi * f * t)
    harmonic3 = np.sin(5 * np.pi * f * t)

    combined = (
        np.random.random() * harmonic1
        + np.random.random() * harmonic2
        + np.random.random() * harmonic3
    )

    combined = (combined - combined.min()) / (combined.max() - combined.min())
    threshold = np.random.uniform(0.3, 0.7)
    maze = (combined > threshold).astype(np.int32)

    # Apply binary dilation to create thicker walls
    structure = np.ones((3, 3), dtype=np.int32)
    maze = binary_dilation(maze, structure=structure).astype(np.int32)

    # Ensure borders are walls
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1

    return maze


def create_quantum_inspired_maze(width, height):
    x = np.linspace(-5, 5, width)
    y = np.linspace(-5, 5, height)
    xx, yy = np.meshgrid(x, y)

    psi1 = np.exp(-(xx**2 + yy**2) / 2) * np.exp(1j * (xx + yy))
    psi2 = np.exp(-((xx - 2) ** 2 + (yy - 2) ** 2) / 2) * np.exp(1j * (xx - yy))
    psi3 = np.exp(-((xx + 2) ** 2 + (yy + 2) ** 2) / 2) * np.exp(1j * (xx * yy))

    psi_combined = psi1 + psi2 + psi3
    prob_density = np.abs(psi_combined) ** 2

    prob_density = (prob_density - prob_density.min()) / (
        prob_density.max() - prob_density.min()
    )
    maze = (prob_density > np.percentile(prob_density, 70)).astype(np.int32)

    # Apply erosion to create wider passages
    maze = binary_erosion(maze, np.ones((3, 3)))

    # Skeletonize to thin the passages
    maze = skeletonize(1 - maze).astype(np.int32)
    maze = 1 - maze

    # Ensure borders are walls
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1

    return maze


def create_artistic_maze(width, height):
    canvas = np.zeros((height, width), dtype=np.int32)

    def add_brush_strokes(canvas, width, height):
        for _ in range(np.random.randint(5, 15)):
            x, y = np.random.randint(0, width - 1), np.random.randint(0, height - 1)
            length = np.random.randint(10, max(width, height) // 2)
            angle = np.random.uniform(0, 2 * np.pi)
            dx, dy = length * np.cos(angle), length * np.sin(angle)
            rr, cc = np.linspace(x, x + dx, num=100), np.linspace(y, y + dy, num=100)
            rr = np.clip(rr.astype(np.int32), 0, width - 1)
            cc = np.clip(cc.astype(np.int32), 0, height - 1)
            canvas[cc, rr] = 1
        return canvas

    canvas = add_brush_strokes(canvas, width, height)

    # Add random "splatters"
    for _ in range(np.random.randint(3, 8)):
        x, y = np.random.randint(0, width - 1), np.random.randint(0, height - 1)
        radius = np.random.randint(5, 20)
        # Fix: Pass only the radius to disk function
        splatter = disk(radius)
        x_start, y_start = max(0, x - radius), max(0, y - radius)
        x_end, y_end = min(width, x + radius + 1), min(height, y + radius + 1)
        canvas_section = canvas[y_start:y_end, x_start:x_end]
        splatter_section = splatter[: y_end - y_start, : x_end - x_start]
        canvas_section[splatter_section > 0] = 1

    # Apply binary dilation to create thicker strokes
    structure = np.ones((3, 3), dtype=np.int32)
    canvas = binary_dilation(canvas, structure=structure).astype(np.int32)

    # Thin the result to create maze-like structures
    maze = thin(canvas).astype(np.int32)

    # Ensure borders are walls
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
    return maze


def custom_rule(neighborhood):
    center = neighborhood[1, 1]
    neighbors_sum = np.sum(neighborhood) - center
    if center == 1:
        return 1 if neighbors_sum in [2, 3, 4] else 0
    else:
        return 1 if neighbors_sum in [3, 4, 5] else 0


def create_cellular_automaton_maze(width, height):
    maze = np.random.choice([0, 1], size=(height, width), p=[0.6, 0.4])

    for _ in range(np.random.randint(3, 7)):
        new_maze = np.zeros_like(maze)
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                neighborhood = maze[i - 1 : i + 2, j - 1 : j + 2]
                new_maze[i, j] = custom_rule(neighborhood)
        maze = new_maze

    # Apply convolution to smooth the maze
    kernel = np.ones((3, 3)) / 9
    maze = convolve2d(maze, kernel, mode="same", boundary="wrap")
    maze = (maze > 0.5).astype(np.int32)

    # Ensure borders are walls
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
    return maze


def create_fourier_maze_core(width, height):
    noise = np.random.rand(height, width)
    fft_noise = np.fft.fft2(noise)

    center_y, center_x = height // 2, width // 2
    y, x = np.ogrid[-center_y : height - center_y, -center_x : width - center_x]

    min_dim = min(width, height)
    low_freq = (x * x + y * y <= (min_dim // 8) ** 2).astype(np.float32)
    mid_freq = (
        (x * x + y * y <= (min_dim // 4) ** 2) & (x * x + y * y > (min_dim // 8) ** 2)
    ).astype(np.float32)
    high_freq = (
        (x * x + y * y <= (min_dim // 2) ** 2) & (x * x + y * y > (min_dim // 4) ** 2)
    ).astype(np.float32)

    mask = 0.6 * low_freq + 0.3 * mid_freq + 0.1 * high_freq

    filtered_fft = fft_noise * mask
    maze = np.real(np.fft.ifft2(filtered_fft))

    return maze


def create_fourier_maze(width, height):
    maze = create_fourier_maze_core(width, height)
    maze = (maze > np.percentile(maze, 60)).astype(np.int32)

    # Apply erosion to create wider passages
    maze = binary_erosion(maze, np.ones((3, 3)))

    # Skeletonize to thin the passages
    maze = skeletonize(1 - maze).astype(np.int32)
    maze = 1 - maze

    # Ensure borders are walls
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1

    return maze


@nb.jit(nopython=True)
def convolve2d_numba(A, kernel):
    h, w = A.shape
    kh, kw = kernel.shape
    padh, padw = kh // 2, kw // 2

    result = np.zeros_like(A)

    for i in range(h):
        for j in range(w):
            for ki in range(kh):
                for kj in range(kw):
                    ii = (i - padh + ki) % h
                    jj = (j - padw + kj) % w
                    result[i, j] += A[ii, jj] * kernel[ki, kj]

    return result


@nb.jit(nopython=True)
def reaction_diffusion_step(A, B, DA, DB, f, k, laplacian_kernel):
    A_lap = convolve2d_numba(A, laplacian_kernel)
    B_lap = convolve2d_numba(B, laplacian_kernel)
    A += DA * A_lap - A * B**2 + f * (1 - A)
    B += DB * B_lap + A * B**2 - (k + f) * B
    return np.clip(A, 0, 1), np.clip(B, 0, 1)


@nb.jit(nopython=True)
def create_reaction_diffusion_maze_core(width, height, num_iterations):
    A = np.random.rand(height, width)
    B = np.random.rand(height, width)

    laplacian_kernel = np.array([[0.05, 0.2, 0.05], [0.2, -1, 0.2], [0.05, 0.2, 0.05]])

    DA, DB = 0.16, 0.08
    f, k = 0.035, 0.065

    for _ in range(num_iterations):
        A, B = reaction_diffusion_step(A, B, DA, DB, f, k, laplacian_kernel)

    return A


def create_reaction_diffusion_maze(width, height):
    A = create_reaction_diffusion_maze_core(width, height, 20)

    maze = (A - A.min()) / (A.max() - A.min())
    maze = (maze > np.random.uniform(0.4, 0.6)).astype(np.int32)

    # Apply binary dilation to create thicker walls
    structure = np.ones((3, 3), dtype=np.int32)
    maze = binary_dilation(maze, structure=structure).astype(np.int32)

    # Ensure borders are walls
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
    return maze


def create_better_maze(width, height, maze_generation_approach):
    if maze_generation_approach == "dla":
        return create_dla_maze(width, height)
    elif maze_generation_approach == "random_game_of_life":
        return create_game_of_life_maze(width, height)
    elif maze_generation_approach == "random_one_dim_automata":
        return create_one_dim_automata_maze(width, height)
    elif maze_generation_approach == "langtons_ant":
        return create_langtons_ant_maze(width, height)
    elif maze_generation_approach == "voronoi":
        return create_voronoi_maze(width, height)
    elif maze_generation_approach == "fractal":
        return create_fractal_maze(width, height)
    elif maze_generation_approach == "wave_function_collapse":
        return create_wave_function_collapse_maze(width, height)
    elif maze_generation_approach == "growing_tree":
        return create_growing_tree_maze(width, height)
    elif maze_generation_approach == "terrain":
        return create_terrain_based_maze(width, height)
    elif maze_generation_approach == "maze_from_image":
        return create_maze_from_image(width, height)
    elif maze_generation_approach == "musicalized":
        return create_musicalized_maze(width, height)
    elif maze_generation_approach == "quantum_inspired":
        return create_quantum_inspired_maze(width, height)
    elif maze_generation_approach == "artistic":
        return create_artistic_maze(width, height)
    elif maze_generation_approach == "cellular_automaton":
        return create_cellular_automaton_maze(width, height)
    elif maze_generation_approach == "fourier":
        return create_fourier_maze(width, height)
    elif maze_generation_approach == "reaction_diffusion":
        return create_reaction_diffusion_maze(width, height)
    else:
        raise ValueError(
            f"Unknown maze generation approach: {maze_generation_approach}"
        )


@nb.jit
def get_neighbors(x, y, maze_shape):
    neighbors = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < maze_shape[1] and 0 <= ny < maze_shape[0]:
            neighbors.append((nx, ny))
    return neighbors


@nb.jit
def manhattan_distance(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


@nb.jit
def a_star(start, goal, maze):
    frontier = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    while frontier:
        current = frontier.pop(0)[1]

        if current == goal:
            break

        for next in get_neighbors(*current, maze.shape):
            new_cost = cost_so_far[current] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + manhattan_distance(goal, next)
                frontier.append((priority, next))
                frontier.sort(key=lambda x: x[0])
                came_from[next] = current

    return came_from


@nb.jit
def manhattan_distance_to_start(start_x, start_y, point):
    return abs(point[0] - start_x) + abs(point[1] - start_y)


@nb.jit
def find_min_distance_neighbor(neighbors, start_x, start_y):
    min_distance = float("inf")
    min_neighbor = None
    for neighbor in neighbors:
        distance = manhattan_distance_to_start(start_x, start_y, neighbor)
        if distance < min_distance:
            min_distance = distance
            min_neighbor = neighbor
    return min_neighbor


@nb.jit
def ensure_connectivity(maze, start, goal):
    path = a_star(start, goal, maze)
    if goal not in path:
        current = goal
        while current != start:
            x, y = current
            maze[y, x] = 0
            neighbors = get_neighbors(x, y, maze.shape)
            current = find_min_distance_neighbor(neighbors, start[0], start[1])
    return maze


@nb.jit
def smart_hole_puncher(maze, start, goal):
    path = a_star(start, goal, maze)
    if goal not in path:
        current = goal
        start_x, start_y = start
        while current != start:
            x, y = current
            if maze[y, x] == 1:
                maze[y, x] = 0
            neighbors = get_neighbors(x, y, maze.shape)
            next_step = find_min_distance_neighbor(neighbors, start_x, start_y)
            current = next_step

    return maze, True


def set_nice():
    try:
        os.nice(22)  # Increase the niceness value to lower the priority
    except AttributeError:
        pass  # os.nice() is not available on Windows


@nb.jit
def add_walls(maze, target_percentage):
    height, width = maze.shape
    total_cells = height * width
    initial_wall_count = np.sum(maze)
    target_wall_count = int(total_cells * target_percentage)

    print(
        "Adding walls. Initial count: "
        + str(initial_wall_count)
        + ", Target count: "
        + str(target_wall_count)
    )

    while np.sum(maze) < target_wall_count:
        y, x = np.random.randint(1, height - 1), np.random.randint(1, width - 1)
        if maze[y, x] == 0:
            maze[y, x] = 1

    final_wall_count = np.sum(maze)
    print("Walls added. Final count: " + str(final_wall_count))
    return maze


@nb.jit
def remove_walls(maze, target_percentage):
    height, width = maze.shape
    total_cells = height * width
    initial_wall_count = np.sum(maze)
    target_wall_count = int(total_cells * target_percentage)

    print(
        "Removing walls. Initial count: "
        + str(initial_wall_count)
        + ", Target count: "
        + str(target_wall_count)
    )

    while np.sum(maze) > target_wall_count:
        y, x = np.random.randint(1, height - 1), np.random.randint(1, width - 1)
        if maze[y, x] == 1:
            maze[y, x] = 0

    final_wall_count = np.sum(maze)
    print("Walls removed. Final count: " + str(final_wall_count))
    return maze


@nb.jit
def add_room_separators(maze):
    height, width = maze.shape
    for _ in range(3):  # Add a few separators
        x = np.random.randint(1, width - 1)
        maze[:, x] = 1
        y = np.random.randint(1, height - 1)
        maze[y, :] = 1
    return maze


def break_up_large_room(maze, max_percentage, max_iterations=1000):
    height, width = maze.shape
    total_cells = height * width

    @nb.jit
    def process_room(
        maze, labeled_maze, num_rooms, room_sizes, largest_room, iterations
    ):
        y, x = (
            np.random.choice(np.where(labeled_maze == largest_room)[0]),
            np.random.choice(np.where(labeled_maze == largest_room)[1]),
        )

        temp_maze = maze.copy()
        temp_maze[y, x] = 1
        temp_labeled, temp_num_rooms = label(1 - temp_maze)

        if temp_num_rooms <= num_rooms:
            maze[y, x] = 1
            labeled_maze, num_rooms = label(1 - maze)
            room_sizes = np.bincount(labeled_maze.flatten())[1:]
            largest_room = np.argmax(room_sizes) + 1

        iterations += 1

        if iterations % 100 == 0:
            for _ in range(5):
                ry, rx = (
                    np.random.randint(1, height - 1),
                    np.random.randint(1, width - 1),
                )
                maze[ry, rx] = 0

        return maze, labeled_maze, num_rooms, room_sizes, largest_room, iterations

    labeled_maze, num_rooms = label(1 - maze)
    room_sizes = np.bincount(labeled_maze.flatten())[1:]
    largest_room = np.argmax(room_sizes) + 1

    iterations = 0
    while (
        np.max(room_sizes) / (total_cells - np.sum(maze)) > max_percentage
        and iterations < max_iterations
    ):
        maze, labeled_maze, num_rooms, room_sizes, largest_room, iterations = (
            process_room(
                maze, labeled_maze, num_rooms, room_sizes, largest_room, iterations
            )
        )

    if iterations >= max_iterations:
        print(
            f"Warning: Maximum iterations ({max_iterations}) reached in break_up_large_room"
        )

    return maze


def break_up_large_areas(maze, max_area_percentage=0.2):
    labeled_areas, num_areas = label(1 - maze)
    area_sizes = np.bincount(labeled_areas.ravel())[1:]
    total_cells = maze.size

    print(f"Initial number of areas: {num_areas}")
    print(f"Initial area sizes: {area_sizes}")

    @nb.jit
    def process_area(maze, labeled_areas, i, size, total_cells, max_area_percentage):
        if size / total_cells > max_area_percentage:
            print(
                f"Breaking up area {i} with size {size} ({size/total_cells:.2f} of total)"
            )
            area_coords = np.argwhere(labeled_areas == i)
            sub_maze_size = int(np.sqrt(size))
            sub_maze = np.ones((sub_maze_size, sub_maze_size))
            sub_maze = create_simple_maze(sub_maze)
            x_offset, y_offset = area_coords.min(axis=0)
            maze[
                x_offset : x_offset + sub_maze.shape[0],
                y_offset : y_offset + sub_maze.shape[1],
            ] = sub_maze
        return maze

    for i, size in enumerate(area_sizes, 1):
        maze = process_area(
            maze, labeled_areas, i, size, total_cells, max_area_percentage
        )

    final_labeled_areas, final_num_areas = label(1 - maze)
    final_area_sizes = np.bincount(final_labeled_areas.ravel())[1:]
    print(f"Final number of areas: {final_num_areas}")
    print(f"Final area sizes: {final_area_sizes}")

    return maze


@nb.jit
def create_simple_maze(sub_maze):
    def carve_passages(x, y):
        directions = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)])
        np.random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx * 2, y + dy * 2
            if (
                0 <= nx < sub_maze.shape[1]
                and 0 <= ny < sub_maze.shape[0]
                and sub_maze[ny, nx] == 1
            ):
                sub_maze[ny, nx] = 0
                sub_maze[y + dy, x + dx] = 0
                carve_passages(nx, ny)

    sub_maze[1::2, 1::2] = 0
    start_x, start_y = 1, 1
    carve_passages(start_x, start_y)
    return sub_maze


@nb.njit
def connect_areas(maze, labeled_areas, num_areas):
    if num_areas > 1:
        for i in range(1, num_areas):
            area1 = np.argwhere(labeled_areas == i)
            area2 = np.argwhere(labeled_areas == i + 1)
            if len(area1) > 0 and len(area2) > 0:
                point1 = area1[np.random.randint(len(area1))]
                point2 = area2[np.random.randint(len(area2))]
                x1, y1 = point1
                x2, y2 = point2
                path = bresenham_line(x1, y1, x2, y2)
                for x, y in path:
                    if 0 <= x < maze.shape[1] and 0 <= y < maze.shape[0]:
                        maze[y, x] = 0
    return maze


def connect_disconnected_areas(maze):
    labeled_areas, num_areas = label(1 - maze)
    return connect_areas(maze, labeled_areas, num_areas)


@nb.njit
def bresenham_line(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    line_points = []
    while True:
        line_points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return line_points


def line(y0, x0, y1, x1):
    """Generate line pixels using Bresenham's line algorithm"""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    points_x, points_y = [], []

    while True:
        points_x.append(x)
        points_y.append(y)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return np.array(points_y), np.array(points_x)


def draw_lines(maze, vertices, ridge_vertices):
    for simplex in ridge_vertices:
        if simplex[0] != -1 and simplex[1] != -1:
            p1, p2 = vertices[simplex[0]], vertices[simplex[1]]
            x1, y1 = int(p1[1]), int(p1[0])
            x2, y2 = int(p2[1]), int(p2[0])
            rr, cc = line(y1, x1, y2, x2)
            valid = (rr >= 0) & (rr < maze.shape[0]) & (cc >= 0) & (cc < maze.shape[1])
            maze[rr[valid], cc[valid]] = 0
    return maze


def validate_and_adjust_maze(maze, maze_generation_approach):
    height, width = maze.shape
    total_cells = height * width

    print(f"\nValidating and adjusting maze for {maze_generation_approach}")
    print(f"Initial maze shape: {height}x{width}")

    target_wall_percentage = np.random.uniform(0.2, 0.3)
    print(f"Target wall percentage: {target_wall_percentage:.2f}")

    wall_count = np.sum(maze)
    wall_percentage = wall_count / total_cells

    print(f"Initial wall count: {wall_count:,}")
    print(f"Initial wall percentage: {wall_percentage:.2f}")

    labeled_areas, num_areas = label(1 - maze)
    if num_areas > 1:
        print(f"Connecting {num_areas} disconnected areas...")
        maze = connect_disconnected_areas(maze)

    adjustment_factor = 0.05
    if wall_percentage < target_wall_percentage - adjustment_factor:
        print("Wall percentage too low. Adding walls...")
        maze = add_walls(maze, target_wall_percentage)
    elif wall_percentage > target_wall_percentage + adjustment_factor:
        print("Wall percentage too high. Removing walls...")
        maze = remove_walls(maze, target_wall_percentage)

    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
    print("Ensured border walls")

    final_wall_count = np.sum(maze)
    final_wall_percentage = final_wall_count / total_cells
    print(f"Final wall count: {final_wall_count}")
    print(f"Final wall percentage: {final_wall_percentage:.2f}")

    labeled_areas, num_areas = label(1 - maze)
    print(f"Number of disconnected areas: {num_areas}")

    return maze


def generate_and_validate_maze(width, height, maze_generation_approach):
    maze = create_better_maze(width, height, maze_generation_approach)
    maze = validate_and_adjust_maze(maze, maze_generation_approach)
    start = (1, 1)
    goal = (width - 2, height - 2)
    maze[start[1], start[0]] = maze[goal[1], goal[0]] = 0
    is_solvable = is_maze_solvable(maze, start, goal)
    if not is_solvable:
        print(f"Maze is not solvable. Wall percentage: {np.sum(maze) / maze.size:.2f}")
    return maze, start, goal, is_solvable


def generate_solvable_maze(
    width, height, maze_generation_approach, max_attempts=20, max_workers=None
):
    print(
        f"Attempting to generate a solvable maze using {maze_generation_approach} approach..."
    )

    generate_func = partial(
        generate_and_validate_maze, width, height, maze_generation_approach
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_func) for _ in range(max_attempts)]

        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            maze, start, goal, is_solvable = future.result()
            print(f"Checked maze {idx + 1}")

            if is_solvable:
                print(f"Solvable maze found after checking {idx + 1} mazes")
                for f in futures:
                    f.cancel()
                return maze, start, goal

            # Apply smart hole puncher even if not initially solvable
            maze, success = smart_hole_puncher(maze, start, goal)
            maze = ensure_connectivity(maze, start, goal)
            if success:
                is_solvable = is_maze_solvable(maze, start, goal)
                if is_solvable:
                    print(
                        f"Solvable maze created using smart hole puncher after {idx + 1} attempts"
                    )
                    return maze, start, goal

    print(f"Failed to generate a solvable maze after {max_attempts} attempts.")
    return None, None, None


@nb.jit(nopython=True)
def prepare_exploration_map(exploration_order, frame, GRID_SIZE):
    exploration_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for idx, (x, y) in enumerate(exploration_order[:frame]):
        exploration_map[y, x] = idx + 1
    if exploration_map.max() > 0:
        exploration_map /= exploration_map.max()
    return exploration_map


@nb.jit(nopython=True)
def prepare_maze_rgba(maze, wall_color_rgba, floor_color_rgba):
    height, width = maze.shape
    maze_rgba = np.zeros((height, width, 4), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            if maze[y, x] == 1:
                maze_rgba[y, x] = wall_color_rgba
            else:
                maze_rgba[y, x] = floor_color_rgba
    return maze_rgba


def delete_small_files(folder, size_limit_kb=20):
    one_hour_ago = time.time() - 3600
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            file_size = os.path.getsize(filepath)
            last_modified_time = os.path.getmtime(filepath)
            if file_size < size_limit_kb * 1024 and last_modified_time < one_hour_ago:
                os.remove(filepath)
                print(
                    f"Deleted {filename} because it is smaller than {size_limit_kb}KB and was last modified over an hour ago."
                )


def remove_old_empty_directories(base_folder="maze_animations", age_limit_hours=1):
    if not os.path.exists(base_folder):
        return

    current_time = time.time()
    age_limit_seconds = age_limit_hours * 3600

    for dir_name in os.listdir(base_folder):
        dir_path = os.path.join(base_folder, dir_name)
        if os.path.isdir(dir_path):
            if not os.listdir(dir_path):
                last_modified_time = os.path.getmtime(dir_path)
                if (current_time - last_modified_time) > age_limit_seconds:
                    try:
                        shutil.rmtree(dir_path)
                        print(f"Removed empty and old directory: {dir_path}")
                    except OSError as e:
                        print(f"Error removing directory {dir_path}: {e}")


def create_output_folder(base_folder="maze_animations"):
    os.makedirs(base_folder, exist_ok=True)
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    animation_folder_name = f"animation_{date_time}"
    output_folder = os.path.join(base_folder, animation_folder_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def generate_and_save_frame(
    frame,
    all_mazes,
    all_exploration_orders,
    all_paths,
    all_starts,
    all_goals,
    all_maze_approaches,
    GRID_SIZE,
    wall_color,
    floor_color,
    start_color,
    goal_color,
    path_color,
    exploration_cmap,
    DPI,
    output_folder,
    frame_format,
):
    fig = plt.figure(figsize=(24, 13), dpi=DPI)  # Slightly increased height
    fig.patch.set_facecolor("#1E1E1E")  # Dark background for contrast

    # Main title
    title_color = plt.cm.viridis(0.5 + 0.2 * np.sin(frame * 0.1))
    title = fig.suptitle(
        "2D Maze Pathfinding Visualization",
        fontsize=20,
        fontweight="bold",
        color=title_color,
        y=0.98,
    )
    title.set_path_effects([withStroke(linewidth=3, foreground="black")])

    # Create layout for mazes
    gs_mazes = fig.add_gridspec(
        1, len(all_mazes), left=0.05, right=0.95, top=0.9, bottom=0.2
    )

    wall_color_rgba = (
        np.array(
            [int(wall_color[i : i + 2], 16) for i in (1, 3, 5)] + [255],
            dtype=np.float32,
        )
        / 255
    )
    floor_color_rgba = (
        np.array(
            [int(floor_color[i : i + 2], 16) for i in (1, 3, 5)] + [255],
            dtype=np.float32,
        )
        / 255
    )

    for i in range(len(all_mazes)):
        ax = fig.add_subplot(gs_mazes[i])

        maze = all_mazes[i]
        maze_rgba = prepare_maze_rgba(maze, wall_color_rgba, floor_color_rgba)

        ax.imshow(maze_rgba)

        exploration_length = len(all_exploration_orders[i])
        path_length = len(all_paths[i])
        total_steps = exploration_length + path_length

        if frame < exploration_length:
            exploration_map = prepare_exploration_map(
                all_exploration_orders[i], frame, GRID_SIZE
            )
            ax.imshow(exploration_map, cmap=exploration_cmap, alpha=0.7)
            current_step = frame
        else:
            path_frame = frame - exploration_length
            if path_frame < path_length:
                path_segment = all_paths[i][: path_frame + 1]
                points = np.array(path_segment).reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(
                    segments, colors=path_color, linewidths=3, alpha=0.8
                )
                ax.add_collection(lc)
            current_step = min(frame, total_steps)

        start_x, start_y = all_starts[i]
        goal_x, goal_y = all_goals[i]

        # Larger and more prominent start and goal markers
        start_circle = Circle(
            (start_x, start_y),
            0.8,
            color=start_color,
            alpha=0.9 + 0.1 * np.sin(frame * 0.2),
            zorder=10,
        )
        goal_star = Line2D(
            [goal_x],
            [goal_y],
            marker="*",
            color=goal_color,
            markersize=20,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=10,
        )

        ax.add_patch(start_circle)
        ax.add_line(goal_star)

        ax.set_title(
            f"{all_maze_approaches[i].replace('_', ' ').title()}",
            color="white",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        ax.set_axis_off()

        # Improved progress bar using FancyBboxPatch
        progress = current_step / total_steps
        ax.add_patch(
            FancyBboxPatch(
                (0.05, -0.15),
                width=0.9,
                height=0.04,
                boxstyle="round,pad=0.02",
                facecolor="#3E3E3E",
                edgecolor="none",
                transform=ax.transAxes,
                zorder=20,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (0.05, -0.15),
                width=0.9 * progress,
                height=0.04,
                boxstyle="round,pad=0.02",
                facecolor="#4CAF50",
                edgecolor="none",
                transform=ax.transAxes,
                zorder=21,
            )
        )

        # Progress text
        ax.text(
            0.5,
            -0.2,
            f"Progress: {current_step}/{total_steps} ({progress:.1%})",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            transform=ax.transAxes,
            zorder=22,
        )

    # Create a layout for the legend and info text
    gs_info = fig.add_gridspec(1, 2, left=0.05, right=0.95, top=0.15, bottom=0.02)

    # Add general information with a modern look
    info_ax = fig.add_subplot(gs_info[0])
    info_ax.axis("off")
    info_text = f"Frame: {frame} | Grid Size: {GRID_SIZE}x{GRID_SIZE}"
    info_ax.text(
        0.5,
        0.5,
        info_text,
        ha="center",
        va="center",
        fontsize=10,
        color="white",
        bbox=dict(
            facecolor="#3E3E3E", edgecolor="none", alpha=0.7, pad=3, boxstyle="round"
        ),
    )

    # Add a legend
    legend_ax = fig.add_subplot(gs_info[1])
    legend_ax.axis("off")
    legend_elements = [
        Line2D([0], [0], color=path_color, lw=2, label="Path"),
        Patch(facecolor=start_color, edgecolor="none", label="Start"),
        Line2D(
            [0],
            [0],
            marker="*",
            color=goal_color,
            label="Goal",
            linestyle="None",
            markersize=15,
        ),
        Patch(
            facecolor=exploration_cmap(0.5),
            edgecolor="none",
            label="Exploration",
            alpha=0.5,
        ),
    ]
    legend_ax.legend(
        handles=legend_elements,
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=10,
        labelcolor="white",
    )

    # Adjust the layout
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    frame_filename = os.path.join(output_folder, f"frame_{frame:04d}.{frame_format}")
    plt.savefig(
        frame_filename,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches="tight",
    )
    plt.close(fig)
    return frame_filename


async def save_animation_async(anim, filepath, writer, DPI):
    await to_thread(anim.save, filepath, writer=writer, dpi=DPI)


async def run_complex_examples(
    num_animations=1,
    GRID_SIZE=31,
    num_problems=1,
    DPI=50,
    FPS=60,
    save_as_frames_only=False,
    frame_format="png",
    dark_mode=False,
    override_maze_approach=None,
):
    if dark_mode:
        wall_color = "#ECF0F1"  # Light pastel for walls
        floor_color = "#2C3E50"  # Dark gray for background
        start_color = "#1ABC9C"  # Cool pastel cyan for start
        goal_color = "#E74C3C"  # Pastel red for goal
        path_color = "#9B59B6"  # Cool pastel purple for path
    else:
        wall_color = "#2C3E50"  # Default dark wall color
        floor_color = "#ECF0F1"  # Default light floor color
        start_color = "#27AE60"  # Default green for start
        goal_color = "#E74C3C"  # Default red for goal
        path_color = "#3498DB"  # Default blue for path
    exploration_colors = [
        "#FFF9C4",
        "#FFE082",
        "#FFB74D",
        "#FF8A65",
        "#E57373",
    ]  # Warm colors
    exploration_cmap = LinearSegmentedColormap.from_list(
        "exploration", exploration_colors, N=100
    )

    maze_generation_approaches = [
        "dla",
        "random_game_of_life",
        "random_one_dim_automata",
        "voronoi",
        "fractal",
        "wave_function_collapse",
        "growing_tree",
        "terrain",
        "musicalized",
        "quantum_inspired",
        "artistic",
        "cellular_automaton",
        "fourier",
        "reaction_diffusion",
    ]

    for animation_index in range(num_animations):
        print(f"\nGenerating animation {animation_index + 1} of {num_animations}")
        output_folder = create_output_folder()

        all_exploration_orders = []
        all_paths = []
        all_mazes = []
        all_starts = []
        all_goals = []
        all_maze_approaches = []

        for i in range(num_problems):
            attempts = 0
            max_attempts = 5  # Allow more attempts per problem

            while attempts < max_attempts:
                maze_generation_approach = override_maze_approach or random.choice(
                    maze_generation_approaches
                )

                print(
                    f"Starting maze generation for problem {i+1} using {maze_generation_approach} approach (attempt {attempts+1})..."
                )
                maze, start, goal = generate_solvable_maze(
                    GRID_SIZE, GRID_SIZE, maze_generation_approach
                )

                if maze is not None:
                    break

                attempts += 1
                print("Failed to generate a solvable maze. Trying again.")

            if maze is None:
                print(
                    f"Failed to generate a solvable maze for problem {i+1} after {max_attempts} attempts. Skipping this problem."
                )
                continue

            start_x, start_y = start
            goal_x, goal_y = goal
            print("Maze generation complete.")

            exploration_order = []
            path = []

            def optimized_a_star(start, goal, maze):
                nonlocal exploration_order, path
                start_x, start_y = start
                goal_x, goal_y = goal

                frontier = PriorityQueue()
                frontier.insert(encode_integer_coordinates(start_x, start_y), 0)
                came_from = {}
                cost_so_far = {encode_integer_coordinates(start_x, start_y): 0}

                while not frontier.is_empty():
                    current = frontier.pop()
                    current_x, current_y = decode_integer_coordinates(current)

                    if (current_x, current_y) == goal:
                        break

                    for dx, dy in [
                        (0, -1),
                        (0, 1),
                        (-1, 0),
                        (1, 0),
                        (-1, -1),
                        (-1, 1),
                        (1, -1),
                        (1, 1),
                    ]:
                        next_x, next_y = current_x + dx, current_y + dy
                        if (
                            0 <= next_x < GRID_SIZE
                            and 0 <= next_y < GRID_SIZE
                            and maze[next_y, next_x] == 0
                        ):
                            if abs(dx) == 1 and abs(dy) == 1:
                                if (
                                    maze[current_y + dy, current_x] == 1
                                    or maze[current_y, current_x + dx] == 1
                                ):
                                    continue

                            new_cost = cost_so_far[current] + (
                                1 if abs(dx) + abs(dy) == 1 else 1.414
                            )
                            next_node = encode_integer_coordinates(next_x, next_y)

                            if (
                                next_node not in cost_so_far
                                or new_cost < cost_so_far[next_node]
                            ):
                                cost_so_far[next_node] = new_cost
                                priority = new_cost + math.sqrt(
                                    (goal_x - next_x) ** 2 + (goal_y - next_y) ** 2
                                )
                                frontier.insert(next_node, priority)
                                came_from[next_node] = current
                                exploration_order.append((next_x, next_y))

                if encode_integer_coordinates(goal_x, goal_y) not in came_from:
                    print(
                        f"No path found from ({start_x}, {start_y}) to ({goal_x}, {goal_y})"
                    )
                    return None

                path.clear()
                current = encode_integer_coordinates(goal_x, goal_y)
                while current != encode_integer_coordinates(start_x, start_y):
                    x, y = decode_integer_coordinates(current)
                    path.append((x, y))
                    current = came_from[current]
                path.append((start_x, start_y))
                path.reverse()

                print(f"Path found with length {len(path)}")
                return path

            print("Starting pathfinding...")
            result = optimized_a_star((start_x, start_y), (goal_x, goal_y), maze)
            if result is None:
                print(
                    f"Failed to find a path for problem {i+1}. Skipping visualization for this problem."
                )
                continue
            print("Pathfinding complete.")

            all_exploration_orders.append(exploration_order)
            all_paths.append(path)
            all_mazes.append(maze)
            all_starts.append((start_x, start_y))
            all_goals.append((goal_x, goal_y))
            all_maze_approaches.append(maze_generation_approach)

        if not all_paths:
            print("No valid paths found. Skipping this animation.")
            continue
        max_frames = max(
            len(exploration_order) + len(path)
            for exploration_order, path in zip(all_exploration_orders, all_paths)
        )
        print(f"Max frames: {max_frames}")
        max_frames = max(1, max_frames)
        num_cores = max(1, os.cpu_count() - 8)  # Leave 8 cores for system processes
        max_concurrent_tasks = min(
            num_cores, 8
        )  # Limit to 8 concurrent tasks or number of cores, whichever is smaller
        semaphore = Semaphore(max_concurrent_tasks)
        print(
            f"Using {num_cores} cores for frame generation and a semaphore with {max_concurrent_tasks} permits."
        )

        frame_generator = partial(
            generate_and_save_frame,
            all_mazes=all_mazes,
            all_exploration_orders=all_exploration_orders,
            all_paths=all_paths,
            all_starts=all_starts,
            all_goals=all_goals,
            all_maze_approaches=all_maze_approaches,
            GRID_SIZE=GRID_SIZE,
            wall_color=wall_color,
            floor_color=floor_color,
            start_color=start_color,
            goal_color=goal_color,
            path_color=path_color,
            exploration_cmap=exploration_cmap,
            DPI=DPI,
            output_folder=output_folder,
            frame_format=frame_format,
        )

        async def process_frame(executor, frame):
            async with semaphore:
                return await asyncio.get_event_loop().run_in_executor(
                    executor, frame_generator, frame
                )

        # Generate and save frames concurrently using ProcessPoolExecutor and Semaphore
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            tasks = [process_frame(executor, frame) for frame in range(max_frames)]

            pbar = tqdm(
                asyncio.as_completed(tasks), total=max_frames, desc="Generating frames"
            )
            for i, task in enumerate(pbar, start=1):
                try:
                    frame_filename = await task
                    pbar.set_description(f"Generated frame {i}/{max_frames}")
                    pbar.set_postfix(
                        {"Current file": frame_filename.split("/")[-1]}, refresh=True
                    )
                except Exception as e:
                    pbar.set_description(f"Error in frame {i}/{max_frames}")
                    pbar.set_postfix({"Error": str(e)}, refresh=True)

                if i % 100 == 0:  # Every 100 frames
                    gc.collect()  # Force garbage collection

        print("\nFrame generation complete.")

        # If saving as a video, compile the saved frames
        if not save_as_frames_only:
            fig, axs = plt.subplots(1, num_problems, figsize=(20, 8), dpi=DPI)
            if num_problems == 1:
                axs = [axs]

            def update_frame(frame):
                for i, ax in enumerate(axs):
                    frame_filename = os.path.join(
                        output_folder, f"frame_{frame:04d}.{frame_format}"
                    )
                    img = Image.open(frame_filename)
                    ax.clear()
                    ax.imshow(img)
                    ax.axis("off")

            anim = FuncAnimation(
                fig, update_frame, frames=max_frames, interval=100, blit=False
            )

            ffmpeg_params = [
                "-threads",
                str(num_cores),
                "-c:v",
                "libx265",
                "-preset",
                "medium",
                "-crf",
                "25",
                "-x265-params",
                "frame-threads=5:numa-pools=36:wpp=1:pmode=1:pme=1:bframes=8:b-adapt=2:rc-lookahead=60",
                "-movflags",
                "+faststart",
            ]
            print(f"Selected FFmpeg parameters: {ffmpeg_params}")

            writer = FFMpegWriter(fps=FPS, codec="libx265", extra_args=ffmpeg_params)

            now = datetime.now()
            date_time = now.strftime("%Y%m%d_%H%M%S")
            maze_approach = all_maze_approaches[0] if all_maze_approaches else "unknown"
            filename = f"{maze_approach}_{date_time}.mp4"
            filepath = os.path.join(output_folder, filename)

            print("Saving MP4 for encoding with optimized settings...")
            await save_animation_async(anim, filepath, writer, DPI)
            print(f"Animation saved as '{filepath}'")
            delete_small_files(output_folder)
            plt.close(fig)
        remove_old_empty_directories()


def test_a_star_implementations(num_tests=100, grid_size=31):
    def optimized_a_star(start, goal, maze):
        start_x, start_y = start
        goal_x, goal_y = goal

        frontier = PriorityQueue()
        frontier.insert(encode_integer_coordinates(start_x, start_y), 0)
        came_from = {}
        cost_so_far = {encode_integer_coordinates(start_x, start_y): 0}

        while not frontier.is_empty():
            current = frontier.pop()
            current_x, current_y = decode_integer_coordinates(current)

            if (current_x, current_y) == goal:
                break

            for dx, dy in [
                (0, -1),
                (0, 1),
                (-1, 0),
                (1, 0),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]:
                next_x, next_y = current_x + dx, current_y + dy
                if (
                    0 <= next_x < grid_size
                    and 0 <= next_y < grid_size
                    and maze[next_y, next_x] == 0
                ):
                    if abs(dx) == 1 and abs(dy) == 1:
                        if (
                            maze[current_y + dy, current_x] == 1
                            or maze[current_y, current_x + dx] == 1
                        ):
                            continue

                    new_cost = cost_so_far[current] + (
                        1 if abs(dx) + abs(dy) == 1 else 1.414
                    )
                    next_node = encode_integer_coordinates(next_x, next_y)

                    if (
                        next_node not in cost_so_far
                        or new_cost < cost_so_far[next_node]
                    ):
                        cost_so_far[next_node] = new_cost
                        priority = new_cost + math.sqrt(
                            (goal_x - next_x) ** 2 + (goal_y - next_y) ** 2
                        )
                        frontier.insert(next_node, priority)
                        came_from[next_node] = current

        if encode_integer_coordinates(goal_x, goal_y) not in came_from:
            return None

        path = []
        current = encode_integer_coordinates(goal_x, goal_y)
        while current != encode_integer_coordinates(start_x, start_y):
            x, y = decode_integer_coordinates(current)
            path.append((x, y))
            current = came_from[current]
        path.append((start_x, start_y))
        path.reverse()

        return path

    def simple_a_star(start, goal, maze):
        start_x, start_y = start
        goal_x, goal_y = goal

        exploration_order = []
        path = []
        obstacles = set(
            (x, y) for y, row in enumerate(maze) for x, cell in enumerate(row) if cell
        )

        frontier = PriorityQueue()
        frontier.insert(encode_integer_coordinates(start_x, start_y), 0)
        came_from = {}
        cost_so_far = {encode_integer_coordinates(start_x, start_y): 0}

        while not frontier.is_empty():
            current = frontier.pop()
            current_x, current_y = decode_integer_coordinates(current)

            if (current_x, current_y) == (goal_x, goal_y):
                break

            for dx, dy in [
                (0, -1),
                (0, 1),
                (-1, 0),
                (1, 0),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]:
                next_x, next_y = current_x + dx, current_y + dy
                if (
                    0 <= next_x < grid_size
                    and 0 <= next_y < grid_size
                    and (next_x, next_y) not in obstacles
                ):
                    if abs(dx) == 1 and abs(dy) == 1:
                        if (current_x + dx, current_y) in obstacles or (
                            current_x,
                            current_y + dy,
                        ) in obstacles:
                            continue

                    new_cost = cost_so_far[current] + (
                        1 if abs(dx) + abs(dy) == 1 else 1.414
                    )
                    next_node = encode_integer_coordinates(next_x, next_y)

                    if (
                        next_node not in cost_so_far
                        or new_cost < cost_so_far[next_node]
                    ):
                        cost_so_far[next_node] = new_cost
                        priority = new_cost + math.sqrt(
                            (goal_x - next_x) ** 2 + (goal_y - next_y) ** 2
                        )
                        frontier.insert(next_node, priority)
                        came_from[next_node] = current
                        exploration_order.append((next_x, next_y))

        if encode_integer_coordinates(goal_x, goal_y) not in came_from:
            return None

        current = encode_integer_coordinates(goal_x, goal_y)
        while current != encode_integer_coordinates(start_x, start_y):
            x, y = decode_integer_coordinates(current)
            path.append((x, y))
            current = came_from[current]
        path.append((start_x, start_y))
        path.reverse()

        return path

    def generate_random_maze(size):
        maze = np.random.choice([0, 1], size=(size, size), p=[0.7, 0.3])
        maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
        return maze

    def is_valid_path(path, maze):
        if not path:
            return False
        for x, y in path:
            if maze[y, x] == 1:
                return False
        return True

    print(f"Running {num_tests} tests...")
    optimized_times = []
    simple_times = []
    for i in range(num_tests):
        maze = generate_random_maze(grid_size)
        start = (random.randint(1, grid_size - 2), random.randint(1, grid_size - 2))
        goal = (random.randint(1, grid_size - 2), random.randint(1, grid_size - 2))

        while maze[start[1], start[0]] == 1 or maze[goal[1], goal[0]] == 1:
            start = (random.randint(1, grid_size - 2), random.randint(1, grid_size - 2))
            goal = (random.randint(1, grid_size - 2), random.randint(1, grid_size - 2))

        # Time the optimized implementation
        start_time = time.time()
        optimized_path = optimized_a_star(start, goal, maze)
        optimized_time = time.time() - start_time
        optimized_times.append(optimized_time)

        # Time the current implementation
        start_time = time.time()
        current_path = simple_a_star(start, goal, maze)
        simple_time = time.time() - start_time
        simple_times.append(simple_time)

        if optimized_path is None and current_path is None:
            print(f"Test {i+1}: Both implementations correctly found no path.")
        elif optimized_path is None or current_path is None:
            print(
                f"Test {i+1}: Mismatch - One implementation found a path, the other didn't."
            )
            return False
        elif len(optimized_path) != len(current_path):
            print(f"Test {i+1}: Mismatch - Different path lengths.")
            return False
        elif not is_valid_path(optimized_path, maze) or not is_valid_path(
            current_path, maze
        ):
            print(f"Test {i+1}: Invalid path detected.")
            return False
        else:
            print(
                f"Test {i+1}: Both implementations found valid paths of the same length."
            )
            print(f"  Optimized time: {optimized_time:.6f} seconds")
            print(f"  Simple time: {simple_time:.6f} seconds")

    print("\nAll tests passed successfully!")

    # Calculate and print benchmark results
    avg_optimized_time = sum(optimized_times) / len(optimized_times)
    avg_simple_time = sum(simple_times) / len(simple_times)
    speedup = avg_simple_time / avg_optimized_time

    print("\nBenchmark Results:")
    print(f"Average Optimized A* Time: {avg_optimized_time:.6f} seconds")
    print(f"Average Simple A* Time: {avg_simple_time:.6f} seconds")
    print(f"Speedup: {speedup:.2f}x")

    return True


if __name__ == "__main__":
    use_test = 0
    if use_test:
        test_result = test_a_star_implementations(num_tests=20, grid_size=231)
        print(f"Overall test result: {'Passed' if test_result else 'Failed'}")

    num_animations = 1  # Set this to the desired number of animations to generate
    GRID_SIZE = 91  # Resolution of the maze grid
    num_problems = 3  # Number of mazes to show side by side in each animation
    DPI = 400  # DPI for the animation
    FPS = 4  # FPS for the animation
    save_as_frames_only = 1  # Set this to 1 to save frames as individual images in a generated sub-folder; 0 to save as a single video as well
    dark_mode = 0  # Change the theme of the maze visualization
    asyncio.run(
        run_complex_examples(
            num_animations,
            GRID_SIZE,
            num_problems,
            DPI,
            FPS,
            save_as_frames_only,
            frame_format="png",
            dark_mode=dark_mode,
        )
    )
