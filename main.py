import math
import random
import time
import os
import asyncio
import shutil
from asyncio import to_thread
from datetime import datetime
from collections import deque
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.spatial import Voronoi
from skimage.draw import line
from skimage.morphology import skeletonize, binary_erosion
from noise import snoise2
from scipy.ndimage import label, binary_dilation, generate_binary_structure
from scipy.signal import convolve2d
from skimage.morphology import thin, disk
from tqdm import tqdm
from PIL import Image
import requests
from matplotlib import font_manager

# Add this line to switch to a non-interactive backend
plt.switch_backend("Agg")

# Define the URL for the Montserrat font (Regular weight)
font_url = "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Regular.ttf"
font_filename = "Montserrat-Regular.ttf"
font_path = os.path.join("fonts", font_filename)

# Create a fonts directory if it doesn't exist
os.makedirs("fonts", exist_ok=True)

os.nice(10)  # Increase the niceness value to lower the priority


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


def encode_integer_coordinates(x, y):
    return (x & ((1 << BITS_PER_COORDINATE) - 1)) | (y << BITS_PER_COORDINATE)


def decode_integer_coordinates(value):
    mask = (1 << BITS_PER_COORDINATE) - 1
    x = value & mask
    y = value >> BITS_PER_COORDINATE
    return x, y


def float_to_int(f):
    significand, exponent = math.frexp(f)
    significand = int(significand * (1 << BITS_PER_FLOAT_SIGNIFICAND))
    exponent = exponent + BITS_PER_FLOAT_SIGNIFICAND - 1
    result = (significand & ((1 << BITS_PER_FLOAT_SIGNIFICAND) - 1)) | (
        exponent << BITS_PER_FLOAT_SIGNIFICAND
    )
    return result if f >= 0 else -result


def int_to_float(i):
    v = abs(i)
    significand = v & ((1 << BITS_PER_FLOAT_SIGNIFICAND) - 1)
    exponent = v >> BITS_PER_FLOAT_SIGNIFICAND
    return math.ldexp(
        significand / (1 << BITS_PER_FLOAT_SIGNIFICAND),
        exponent - BITS_PER_FLOAT_SIGNIFICAND + 1,
    ) * (1 if i >= 0 else -1)


def encode_float_coordinates(x, y):
    x_int = float_to_int(x)
    y_int = float_to_int(y)
    return encode_integer_coordinates(x_int, y_int)


def decode_float_coordinates(value):
    x_int, y_int = decode_integer_coordinates(value)
    x = int_to_float(x_int)
    y = int_to_float(y_int)
    return x, y


def make_row_major_indexer(width, node_width=1, node_height=1):
    return lambda x, y: (y // node_height) * width + (x // node_width)


def make_column_major_indexer(height, node_width=1, node_height=1):
    return lambda x, y: (x // node_width) * height + (y // node_height)


def goal_reached_exact_p(x, y, goal_x, goal_y):
    return x == goal_x and y == goal_y


def make_4_directions_enumerator(
    node_width=1, node_height=1, min_x=0, min_y=0, max_x=None, max_y=None
):
    max_x = max_x if max_x is not None else float("inf")
    max_y = max_y if max_y is not None else float("inf")

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


def make_8_directions_enumerator(
    node_width=1, node_height=1, min_x=0, min_y=0, max_x=None, max_y=None
):
    max_x = max_x if max_x is not None else float("inf")
    max_y = max_y if max_y is not None else float("inf")

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


def make_manhattan_distance_heuristic(scale_factor=1.0):
    return lambda x1, y1, x2, y2: scale_factor * (abs(x1 - x2) + abs(y1 - y2))


def make_octile_distance_heuristic(scale_factor=1.0):
    sqrt2 = math.sqrt(2)
    return lambda x1, y1, x2, y2: scale_factor * (
        min(abs(x1 - x2), abs(y1 - y2)) * sqrt2 + abs(abs(x1 - x2) - abs(y1 - y2))
    )


def make_euclidean_distance_heuristic(scale_factor=1.0):
    return lambda x1, y1, x2, y2: scale_factor * math.sqrt(
        (x1 - x2) ** 2 + (y1 - y2) ** 2
    )


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
    def path_finder(start_x, start_y, goal_x, goal_y, **params):
        cost_so_far = [float("nan")] * world_size
        came_from = [-1] * world_size
        path = [-1] * world_size

        frontier = PriorityQueue()

        start_index = indexer(start_x, start_y)
        goal_index = indexer(goal_x, goal_y)
        frontier.insert(coordinate_encoder(start_x, start_y), 0.0)
        cost_so_far[start_index] = 0.0

        while not frontier.is_empty():
            current = frontier.pop()
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
                    frontier.insert(
                        coordinate_encoder(next_x, next_y),
                        new_cost + heuristic_cost(next_x, next_y, goal_x, goal_y),
                    )

            neighbor_enumerator(current_x, current_y, process_neighbor)

        if came_from[goal_index] == -1:
            return None

        length = 0
        current = goal_index
        while current != start_index:
            path[length] = current
            length += 1
            current_x, current_y = coordinate_decoder(came_from[current])
            current = indexer(current_x, current_y)
        path[length] = start_index
        length += 1

        path_initiator(length)
        for i in range(length - 1, -1, -1):
            path_x, path_y = coordinate_decoder(path[i])
            path_processor(path_x, path_y)

        return path_finalizer()

    path_finder.__name__ = name
    return path_finder


def is_maze_solvable(maze, start, goal, max_iterations=100000):
    queue = deque([start])
    visited = set([start])
    iterations = 0

    while queue and iterations < max_iterations:
        iterations += 1
        x, y = queue.popleft()

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


def create_dla_maze(width, height):
    maze = np.zeros((height, width), dtype=int)
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1

    num_seeds = random.randint(
        max(3, min(width, height) // 20), max(10, min(width, height) // 5)
    )
    for _ in range(num_seeds):
        x, y = random.randint(1, width - 2), random.randint(1, height - 2)
        maze[y, x] = 1

    num_particles = random.randint(width * height // 16, width * height // 8)
    for _ in range(num_particles):
        x, y = random.randint(1, width - 2), random.randint(1, height - 2)
        steps = 0
        max_steps = random.randint(100, 1000)
        while maze[y, x] == 0 and steps < max_steps:
            dx, dy = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
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
            if maze[y, x] == 1 and random.random() < 0.1:
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

        if not np.all(np.logical_or(maze == 0, maze == 1)):
            print("Warning: Maze contains values other than 0 and 1")
            print("Unique values in maze:", np.unique(maze))
            maze = np.round(maze).astype(int)  # Round to nearest integer as a fallback
    return maze


def create_one_dim_automata_maze(width, height):
    maze = np.zeros((height, width), dtype=int)
    maze[0] = np.random.choice([0, 1], size=width)
    maze[0, 0] = maze[0, -1] = 1

    rule_number = random.randint(0, 255)
    rule = {
        (a, b, c): (rule_number >> (a * 4 + b * 2 + c)) & 1
        for a in (0, 1)
        for b in (0, 1)
        for c in (0, 1)
    }

    for y in range(1, height):
        for x in range(width):
            left = maze[y - 1, (x - 1) % width]
            center = maze[y - 1, x]
            right = maze[y - 1, (x + 1) % width]
            maze[y, x] = rule[(left, center, right)]

    maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
    return maze


def create_langtons_ant_maze(width, height):
    maze = np.zeros((height, width), dtype=int)
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
            ant_y = max(1, ant_y - 1)  # noqa: E701
        elif ant_direction == 1:
            ant_x = min(width - 2, ant_x + 1)  # noqa: E701
        elif ant_direction == 2:
            ant_y = min(height - 2, ant_y + 1)  # noqa: E701
        else:
            ant_x = max(1, ant_x - 1)  # noqa: E701

    return maze


def create_voronoi_maze(width, height):
    num_points = random.randint(max(width, height) // 3, max(width, height) // 2)
    points = np.random.rand(num_points, 2) * [width, height]
    vor = Voronoi(points)

    maze = np.ones((height, width), dtype=int)
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:
            p1, p2 = vor.vertices[simplex]
            rr, cc = line(int(p1[1]), int(p1[0]), int(p2[1]), int(p2[0]))
            rr = np.clip(rr, 0, height - 1)
            cc = np.clip(cc, 0, width - 1)
            maze[rr, cc] = 0

    # Apply erosion to create wider passages
    maze = binary_erosion(maze, np.ones((3, 3)))

    # Skeletonize to thin the passages
    maze = skeletonize(1 - maze).astype(int)
    maze = 1 - maze

    # Ensure borders are walls
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1

    return maze


def create_fractal_maze(width, height, min_size=4):
    maze = np.zeros((height, width), dtype=int)

    def recursive_divide(x, y, w, h):
        if w <= min_size or h <= min_size:
            return

        # Randomly decide whether to divide horizontally or vertically
        if w > h:
            divide_horizontally = np.random.choice([True, False], p=[0.8, 0.2])
        else:
            divide_horizontally = np.random.choice([True, False], p=[0.2, 0.8])

        if divide_horizontally:
            divide_at = np.random.randint(y + 1, y + h - 1)
            maze[divide_at, x : x + w] = 1
            opening = np.random.randint(x, x + w)
            maze[divide_at, opening] = 0
            recursive_divide(x, y, w, divide_at - y)
            recursive_divide(x, divide_at + 1, w, y + h - divide_at - 1)
        else:
            divide_at = np.random.randint(x + 1, x + w - 1)
            maze[y : y + h, divide_at] = 1
            opening = np.random.randint(y, y + h)
            maze[opening, divide_at] = 0
            recursive_divide(x, y, divide_at - x, h)
            recursive_divide(divide_at + 1, y, x + w - divide_at - 1, h)

    recursive_divide(0, 0, width, height)
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


def create_wave_function_collapse_maze(width, height, timeout=30):
    tiles = {
        0: {"up": [0, 2], "right": [0, 1], "down": [0, 2], "left": [0, 1]},  # Empty
        1: {"up": [1], "right": [1], "down": [1], "left": [1]},  # Wall
        2: {"up": [2], "right": [2], "down": [2], "left": [2]},  # Special
    }

    def get_valid_tiles(x, y):
        valid = set(tiles.keys())
        for dx, dy, direction in [
            (0, -1, "down"),
            (1, 0, "left"),
            (0, 1, "up"),
            (-1, 0, "right"),
        ]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and maze[ny, nx] != -1:
                valid &= set(tiles[maze[ny, nx]][direction])
        return list(valid)

    maze = np.full((height, width), -1, dtype=int)
    stack = [(random.randint(1, width - 2), random.randint(1, height - 2))]
    start_time = time.time()

    while stack and time.time() - start_time < timeout:
        x, y = stack.pop(random.randint(0, len(stack) - 1))
        if maze[y, x] == -1:
            valid_tiles = get_valid_tiles(x, y)
            if valid_tiles:
                maze[y, x] = random.choice(valid_tiles)
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (
                        0 < nx < width - 1
                        and 0 < ny < height - 1
                        and maze[ny, nx] == -1
                    ):
                        stack.append((nx, ny))

    # Fill any remaining -1 cells with random valid tiles
    for y in range(height):
        for x in range(width):
            if maze[y, x] == -1:
                valid_tiles = get_valid_tiles(x, y)
                maze[y, x] = random.choice(valid_tiles) if valid_tiles else 0

    # Ensure borders are walls
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1

    print(
        f"Wave function collapse maze generation took {time.time() - start_time:.2f} seconds"
    )
    return maze


def create_growing_tree_maze(width, height):
    maze = np.ones((height, width), dtype=int)
    stack = [(1, 1)]
    maze[1, 1] = 0

    while stack:
        if random.random() < 0.5:
            current = stack.pop(random.randint(0, len(stack) - 1))
        else:
            current = stack.pop()

        x, y = current
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < width - 1 and 0 < ny < height - 1 and maze[ny, nx] == 1:
                maze[ny, nx] = 0
                stack.append((nx, ny))

    return maze


def create_terrain_based_maze(width, height):
    scale = random.uniform(0.05, 0.1)
    octaves = random.randint(4, 6)
    persistence = random.uniform(0.5, 0.7)
    lacunarity = random.uniform(2.0, 2.5)

    terrain = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            terrain[y, x] = snoise2(
                x * scale,
                y * scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
            )

    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    maze = (terrain > np.percentile(terrain, 60)).astype(int)

    # Apply erosion to create wider passages
    maze = binary_erosion(maze, np.ones((3, 3)))

    # Skeletonize to thin the passages
    maze = skeletonize(1 - maze).astype(int)
    maze = 1 - maze

    # Ensure borders are walls
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1

    return maze


def create_musicalized_maze(width, height):
    # Generate a "musical" maze based on harmonic patterns
    frequencies = np.linspace(1, 10, num=width)
    time = np.linspace(0, 10, num=height)
    t, f = np.meshgrid(time, frequencies)

    # Create harmonic patterns
    harmonic1 = np.sin(2 * np.pi * f * t)
    harmonic2 = np.sin(3 * np.pi * f * t)
    harmonic3 = np.sin(5 * np.pi * f * t)

    # Combine harmonics with random weights
    combined = (
        np.random.random() * harmonic1
        + np.random.random() * harmonic2
        + np.random.random() * harmonic3
    )

    # Normalize and threshold
    combined = (combined - combined.min()) / (combined.max() - combined.min())
    threshold = np.random.uniform(0.3, 0.7)  # Increased range for more variety
    maze = (combined > threshold).astype(int)

    # Apply binary dilation to create thicker walls
    structure = generate_binary_structure(2, 2)
    maze = binary_dilation(maze, structure=structure).astype(int)

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
    maze = (prob_density > np.percentile(prob_density, 70)).astype(int)

    # Apply erosion to create wider passages
    maze = binary_erosion(maze, np.ones((3, 3)))

    # Skeletonize to thin the passages
    maze = skeletonize(1 - maze).astype(int)
    maze = 1 - maze

    # Ensure borders are walls
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1

    return maze


def create_artistic_maze(width, height):
    # Generate a maze inspired by artistic techniques
    canvas = np.zeros((height, width))

    # Add random "brush strokes"
    for _ in range(random.randint(5, 15)):
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        length = random.randint(10, max(width, height) // 2)
        angle = random.uniform(0, 2 * np.pi)
        dx, dy = length * np.cos(angle), length * np.sin(angle)
        rr, cc = np.linspace(x, x + dx, num=100), np.linspace(y, y + dy, num=100)
        rr = np.clip(rr.astype(int), 0, width - 1)
        cc = np.clip(cc.astype(int), 0, height - 1)
        canvas[cc, rr] = 1

    # Add random "splatters"
    for _ in range(random.randint(3, 8)):
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        radius = random.randint(5, 20)
        splatter = disk(radius)
        x_start, y_start = max(0, x - radius), max(0, y - radius)
        x_end, y_end = min(width, x + radius + 1), min(height, y + radius + 1)
        canvas_section = canvas[y_start:y_end, x_start:x_end]
        splatter_section = splatter[: y_end - y_start, : x_end - x_start]
        canvas_section[splatter_section > 0] = 1

    # Apply binary dilation to create thicker strokes
    structure = np.ones((3, 3))
    canvas = binary_dilation(canvas, structure=structure).astype(int)

    # Thin the result to create maze-like structures
    maze = thin(canvas).astype(int)

    # Ensure borders are walls
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
    return maze


def create_cellular_automaton_maze(width, height):
    # Generate a maze using a custom cellular automaton rule
    maze = np.random.choice([0, 1], size=(height, width), p=[0.6, 0.4])

    def custom_rule(neighborhood):
        center = neighborhood[1, 1]
        neighbors_sum = np.sum(neighborhood) - center
        if center == 1:
            return 1 if neighbors_sum in [2, 3, 4] else 0
        else:
            return 1 if neighbors_sum in [3, 4, 5] else 0

    for _ in range(random.randint(3, 7)):
        new_maze = np.zeros_like(maze)
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                neighborhood = maze[i - 1 : i + 2, j - 1 : j + 2]
                new_maze[i, j] = custom_rule(neighborhood)
        maze = new_maze

    # Apply convolution to smooth the maze
    kernel = np.ones((3, 3)) / 9
    maze = convolve2d(maze, kernel, mode="same", boundary="wrap")
    maze = (maze > 0.5).astype(int)

    # Ensure borders are walls
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
    return maze


def create_fourier_maze(width, height):
    noise = np.random.rand(height, width)
    fft_noise = np.fft.fft2(noise)

    center_y, center_x = height // 2, width // 2
    y, x = np.ogrid[-center_y : height - center_y, -center_x : width - center_x]

    # Create a more complex filter
    low_freq = (x * x + y * y <= (min(width, height) // 8) ** 2).astype(float)
    mid_freq = (
        (x * x + y * y <= (min(width, height) // 4) ** 2)
        & (x * x + y * y > (min(width, height) // 8) ** 2)
    ).astype(float)
    high_freq = (
        (x * x + y * y <= (min(width, height) // 2) ** 2)
        & (x * x + y * y > (min(width, height) // 4) ** 2)
    ).astype(float)

    mask = 0.6 * low_freq + 0.3 * mid_freq + 0.1 * high_freq

    filtered_fft = fft_noise * mask
    maze = np.real(np.fft.ifft2(filtered_fft))

    maze = (maze > np.percentile(maze, 60)).astype(int)

    # Apply erosion to create wider passages
    maze = binary_erosion(maze, np.ones((3, 3)))

    # Skeletonize to thin the passages
    maze = skeletonize(1 - maze).astype(int)
    maze = 1 - maze

    # Ensure borders are walls
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1

    return maze


def create_reaction_diffusion_maze(width, height):
    # Generate a maze using a simplified reaction-diffusion system
    A = np.random.rand(height, width)
    B = np.random.rand(height, width)

    # Define the Laplacian kernel
    laplacian_kernel = np.array([[0.05, 0.2, 0.05], [0.2, -1, 0.2], [0.05, 0.2, 0.05]])

    DA, DB = 0.16, 0.08
    f, k = 0.035, 0.065

    for _ in range(20):  # Adjust number of iterations for different patterns
        A_lap = convolve2d(A, laplacian_kernel, mode="same", boundary="wrap")
        B_lap = convolve2d(B, laplacian_kernel, mode="same", boundary="wrap")
        A += DA * A_lap - A * B**2 + f * (1 - A)
        B += DB * B_lap + A * B**2 - (k + f) * B

        # Clip values to prevent instability
        A = np.clip(A, 0, 1)
        B = np.clip(B, 0, 1)

    # Normalize and threshold
    maze = (A - A.min()) / (A.max() - A.min())
    maze = (maze > random.uniform(0.4, 0.6)).astype(int)

    # Apply binary dilation to create thicker walls
    structure = np.ones((3, 3))
    maze = binary_dilation(maze, structure=structure).astype(int)

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


def smart_hole_puncher(maze, start, goal):
    def get_neighbors(x, y):
        return [
            (x + dx, y + dy)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
            if 0 <= x + dx < maze.shape[1] and 0 <= y + dy < maze.shape[0]
        ]

    def manhattan_distance(a, b):
        return abs(b[0] - a[0]) + abs(b[1] - a[1])

    def a_star(start, goal):
        frontier = PriorityQueue()
        frontier.insert(start, 0)
        came_from = {}
        cost_so_far = {start: 0}

        while not frontier.is_empty():
            current = frontier.pop()

            if current == goal:
                break

            for next in get_neighbors(*current):
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + manhattan_distance(goal, next)
                    frontier.insert(next, priority)
                    came_from[next] = current

        return came_from

    path = a_star(start, goal)
    if goal not in path:
        current = goal
        while current != start:
            x, y = current
            if maze[y, x] == 1:
                maze[y, x] = 0
            neighbors = get_neighbors(x, y)
            next_step = min(neighbors, key=lambda n: manhattan_distance(start, n))
            current = next_step

    return maze, True


def set_nice():
    try:
        os.nice(10)  # Increase the niceness value to lower the priority
    except AttributeError:
        pass  # os.nice() is not available on Windows


def add_walls(maze, target_percentage):
    """Add walls to the maze until reaching the target wall percentage."""
    height, width = maze.shape
    total_cells = height * width
    initial_wall_count = np.sum(maze)
    target_wall_count = int(total_cells * target_percentage)

    print(
        f"Adding walls. Initial count: {initial_wall_count}, Target count: {target_wall_count}"
    )

    while np.sum(maze) < target_wall_count:
        y, x = np.random.randint(1, height - 1), np.random.randint(1, width - 1)
        if maze[y, x] == 0:
            maze[y, x] = 1

    final_wall_count = np.sum(maze)
    print(f"Walls added. Final count: {final_wall_count}")
    return maze


def remove_walls(maze, target_percentage):
    """Remove walls from the maze until reaching the target wall percentage."""
    height, width = maze.shape
    total_cells = height * width
    initial_wall_count = np.sum(maze)
    target_wall_count = int(total_cells * target_percentage)

    print(
        f"Removing walls. Initial count: {initial_wall_count}, Target count: {target_wall_count}"
    )

    while np.sum(maze) > target_wall_count:
        y, x = np.random.randint(1, height - 1), np.random.randint(1, width - 1)
        if maze[y, x] == 1:
            maze[y, x] = 0

    final_wall_count = np.sum(maze)
    print(f"Walls removed. Final count: {final_wall_count}")
    return maze


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
    labeled_maze, num_rooms = label(1 - maze, structure=np.ones((3, 3)))
    room_sizes = np.bincount(labeled_maze.flatten())[1:]
    largest_room = np.argmax(room_sizes) + 1

    iterations = 0
    while (
        np.max(room_sizes) / (total_cells - np.sum(maze)) > max_percentage
        and iterations < max_iterations
    ):
        y, x = (
            np.random.choice(np.where(labeled_maze == largest_room)[0]),
            np.random.choice(np.where(labeled_maze == largest_room)[1]),
        )

        # Check if adding a wall here would not disconnect the room
        temp_maze = maze.copy()
        temp_maze[y, x] = 1
        temp_labeled, temp_num_rooms = label(1 - temp_maze, structure=np.ones((3, 3)))

        if (
            temp_num_rooms <= num_rooms
        ):  # If it doesn't increase the number of rooms, it's safe to add a wall
            maze[y, x] = 1
            labeled_maze, num_rooms = label(1 - maze, structure=np.ones((3, 3)))
            room_sizes = np.bincount(labeled_maze.flatten())[1:]
            largest_room = np.argmax(room_sizes) + 1

        iterations += 1

        # Add random openings periodically to prevent complete blockage
        if iterations % 100 == 0:
            for _ in range(5):
                ry, rx = (
                    np.random.randint(1, height - 1),
                    np.random.randint(1, width - 1),
                )
                maze[ry, rx] = 0

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

    for i, size in enumerate(area_sizes, 1):
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

    final_labeled_areas, final_num_areas = label(1 - maze)
    final_area_sizes = np.bincount(final_labeled_areas.ravel())[1:]
    print(f"Final number of areas: {final_num_areas}")
    print(f"Final area sizes: {final_area_sizes}")

    return maze


def create_simple_maze(sub_maze):
    """Simple recursive backtracker maze generation"""

    def carve_passages(x, y):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
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


def connect_disconnected_areas(maze):
    labeled_areas, num_areas = label(1 - maze)
    if num_areas > 1:
        for i in range(1, num_areas):
            area1 = np.argwhere(labeled_areas == i)
            area2 = np.argwhere(labeled_areas == i + 1)
            point1 = area1[np.random.randint(len(area1))]
            point2 = area2[np.random.randint(len(area2))]
            path = bresenham_line(point1, point2)
            for x, y in path:
                maze[y, x] = 0
    return maze


def bresenham_line(start, end):
    """Bresenham's Line Algorithm"""
    x1, y1 = start
    x2, y2 = end
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    path = []
    while True:
        path.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return path


def validate_and_adjust_maze(maze, maze_generation_approach):
    height, width = maze.shape
    total_cells = height * width

    print(f"\nValidating and adjusting maze for {maze_generation_approach}")
    print(f"Initial maze shape: {height}x{width}")

    target_wall_percentage = np.random.uniform(0.2, 0.3)
    print(f"Target wall percentage: {target_wall_percentage:.2f}")

    wall_count = np.sum(maze)
    wall_percentage = wall_count / total_cells

    print(f"Initial wall count: {wall_count}")
    print(f"Initial wall percentage: {wall_percentage:.2f}")

    # Ensure connectivity
    labeled_areas, num_areas = label(1 - maze)
    if num_areas > 1:
        print(f"Connecting {num_areas} disconnected areas...")
        maze = connect_disconnected_areas(maze)

    # Adjust wall percentage if needed
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
            if success:
                is_solvable = is_maze_solvable(maze, start, goal)
                if is_solvable:
                    print(
                        f"Solvable maze created using smart hole puncher after {idx + 1} attempts"
                    )
                    return maze, start, goal

    print(f"Failed to generate a solvable maze after {max_attempts} attempts.")
    return None, None, None


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
    # Generate the frame as before
    fig, axs = plt.subplots(1, len(all_mazes), figsize=(20, 8), dpi=DPI)
    if len(all_mazes) == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        ax.clear()
        maze = all_mazes[i]
        colored_maze = np.where(maze == 1, 1, 0)
        wall_color_rgba = [
            int(wall_color[1:3], 16) / 255,
            int(wall_color[3:5], 16) / 255,
            int(wall_color[5:7], 16) / 255,
            1,
        ]
        floor_color_rgba = [
            int(floor_color[1:3], 16) / 255,
            int(floor_color[3:5], 16) / 255,
            int(floor_color[5:7], 16) / 255,
            1,
        ]
        maze_rgba = np.zeros((*colored_maze.shape, 4))
        maze_rgba[colored_maze == 1] = wall_color_rgba
        maze_rgba[colored_maze == 0] = floor_color_rgba

        ax.imshow(maze_rgba)

        exploration_map = np.zeros((GRID_SIZE, GRID_SIZE))
        exploration_length = len(all_exploration_orders[i])
        path_length = len(all_paths[i])

        if frame < exploration_length:
            for idx, (x, y) in enumerate(all_exploration_orders[i][:frame]):
                exploration_map[y, x] = idx + 1
            if exploration_map.max() > 0:
                exploration_map = exploration_map / exploration_map.max()
            ax.imshow(exploration_map, cmap=exploration_cmap, alpha=0.5)
        else:
            path_frame = frame - exploration_length
            if path_frame < path_length:
                path_segment = all_paths[i][: path_frame + 1]
                path_x, path_y = zip(*path_segment)
                ax.plot(path_x, path_y, color=path_color, linewidth=2, alpha=0.8)

        start_x, start_y = all_starts[i]
        goal_x, goal_y = all_goals[i]
        ax.plot(start_x, start_y, "o", color=start_color, markersize=10, label="Start")
        ax.plot(goal_x, goal_y, "o", color=goal_color, markersize=10, label="Goal")
        ax.set_title(f"Example {i+1}: {all_maze_approaches[i]}")
        ax.axis("off")

    plt.tight_layout()
    fig.canvas.draw()

    # Save the frame directly to disk
    frame_filename = os.path.join(output_folder, f"frame_{frame:04d}.{frame_format}")
    plt.savefig(frame_filename)
    plt.close(fig)

    return frame_filename


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


async def save_animation_async(anim, filepath, writer, DPI):
    await to_thread(anim.save, filepath, writer=writer, dpi=DPI)


def create_output_folder(base_folder="maze_animations"):
    """Create a unique sub-folder in the specified base folder, named with the current datetime."""
    # Ensure the base output folder exists
    os.makedirs(base_folder, exist_ok=True)

    # Create a unique sub-folder name based on the current datetime
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    animation_folder_name = f"animation_{date_time}"

    # Create the full path for the new sub-folder
    output_folder = os.path.join(base_folder, animation_folder_name)

    # Create the sub-folder
    os.makedirs(output_folder, exist_ok=True)

    return output_folder


async def run_complex_examples(
    num_animations=1,
    GRID_SIZE=31,
    num_problems=1,
    DPI=50,
    FPS=60,
    save_as_frames=False,
    frame_format="png",
    override_maze_approach=None,
):
    wall_color = "#2C3E50"
    floor_color = "#ECF0F1"
    start_color = "#27AE60"
    goal_color = "#E74C3C"
    path_color = "#3498DB"
    exploration_colors = ["#FFF9C4", "#FFE082", "#FFB74D", "#FF8A65", "#E57373"]
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
                if override_maze_approach:
                    maze_generation_approach = override_maze_approach
                else:
                    maze_generation_approach = random.choice(maze_generation_approaches)

                print(
                    f"Starting maze generation for problem {i+1} using {maze_generation_approach} approach (attempt {attempts+1})..."
                )
                maze, start, goal = generate_solvable_maze(
                    GRID_SIZE, GRID_SIZE, maze_generation_approach
                )

                if maze is not None:
                    break

                attempts += 1
                print(f"Failed to generate a solvable maze. Trying again.")

            if maze is None:
                print(
                    f"Failed to generate a solvable maze for problem {i+1} after {max_attempts} attempts. Skipping this problem."
                )
                continue

            start_x, start_y = start
            goal_x, goal_y = goal
            print("Maze generation complete.")

            obstacles = set(
                (x, y)
                for y, row in enumerate(maze)
                for x, cell in enumerate(row)
                if cell
            )

            exploration_order = []
            path = []

            def custom_path_finder(start_x, start_y, goal_x, goal_y):
                nonlocal exploration_order, path
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
                            0 <= next_x < GRID_SIZE
                            and 0 <= next_y < GRID_SIZE
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
                    print(
                        f"No path found from ({start_x}, {start_y}) to ({goal_x}, {goal_y})"
                    )
                    return False

                path.clear()
                current = encode_integer_coordinates(goal_x, goal_y)
                while current != encode_integer_coordinates(start_x, start_y):
                    x, y = decode_integer_coordinates(current)
                    path.append((x, y))
                    if current not in came_from:
                        print(f"Path reconstruction failed at ({x}, {y})")
                        return False
                    current = came_from[current]
                path.append((start_x, start_y))
                path.reverse()

                print(f"Path found with length {len(path)}")
                return True

            print("Starting pathfinding...")
            if not custom_path_finder(start_x, start_y, goal_x, goal_y):
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
        num_cores = os.cpu_count()
        print(f"Using {num_cores} cores for frame generation")

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

        # Generate and save frames concurrently
        with ProcessPoolExecutor(
            max_workers=num_cores, initializer=set_nice
        ) as executor:
            list(
                tqdm(
                    executor.map(frame_generator, range(max_frames)),
                    total=max_frames,
                    desc="Generating frames",
                )
            )

        # If saving as a video, compile the saved frames
        if not save_as_frames:
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

            print(
                f"Saving MP4 using {num_cores} cores for encoding with optimized settings..."
            )
            await save_animation_async(anim, filepath, writer, DPI)
            print(f"Animation saved as '{filepath}'")
            delete_small_files(output_folder)
            plt.close(fig)


def remove_old_empty_directories(base_folder="maze_animations", age_limit_hours=1):
    # Check if the base folder exists
    if not os.path.exists(base_folder):
        return

    current_time = time.time()
    age_limit_seconds = age_limit_hours * 3600

    # List all directories in the base folder
    for dir_name in os.listdir(base_folder):
        dir_path = os.path.join(base_folder, dir_name)
        # Check if it's a directory
        if os.path.isdir(dir_path):
            # Check if the directory is empty
            if not os.listdir(dir_path):
                # Get the last modified time
                last_modified_time = os.path.getmtime(dir_path)
                # Check if the directory hasn't been modified for over the specified limit
                if (current_time - last_modified_time) > age_limit_seconds:
                    # If empty and old, remove it
                    try:
                        shutil.rmtree(dir_path)
                        print(f"Removed empty and old directory: {dir_path}")
                    except OSError as e:
                        print(f"Error removing directory {dir_path}: {e}")


if __name__ == "__main__":
    num_animations = 1  # Set this to the desired number of animations to generate
    GRID_SIZE = 351  # Resolution of the maze grid
    num_problems = 2  # Number of mazes to show side by side in each animation
    DPI = 300  # DPI for the animation
    FPS = 60  # FPS for the animation
    save_as_frames = 1  # Set this to 1 to save frames as individual images in a generated sub-folder; 0 to save as a single video
    asyncio.run(
        run_complex_examples(
            num_animations, GRID_SIZE, num_problems, DPI, FPS, save_as_frames
        )
    )
