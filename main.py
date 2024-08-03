import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib.animation import FFMpegWriter
from scipy.spatial import Voronoi
from skimage.draw import line
from skimage.morphology import skeletonize
from tqdm import tqdm
from PIL import Image

# Constants for integer coordinate encoding
BITS_PER_COORDINATE = int(math.floor(math.log2((1 << 63) - 1) / 2))

# Constants for float coordinate encoding
BITS_PER_FLOAT_SIGNIFICAND = 24
BITS_PER_FLOAT_EXPONENT = BITS_PER_COORDINATE - 1 + BITS_PER_FLOAT_SIGNIFICAND
MOST_FLOAT_COORDINATE = (1 - (1 / (1 << BITS_PER_FLOAT_SIGNIFICAND))) * (
    1 << (BITS_PER_FLOAT_EXPONENT - 1))
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
    self.priorities.extend([float('inf')] * (new_size - len(self.priorities)))

  @staticmethod
  def new_capacity(current_capacity):
    return current_capacity + (current_capacity >> 1)

  def _heapify(self, i):
    smallest = i
    l = 2 * i + 1
    r = 2 * i + 2

    if l < self.size and self.priorities[l] < self.priorities[smallest]:
      smallest = l
    if r < self.size and self.priorities[r] < self.priorities[smallest]:
      smallest = r

    if smallest != i:
      self.items[i], self.items[smallest] = self.items[smallest], self.items[i]
      self.priorities[i], self.priorities[smallest] = self.priorities[
          smallest], self.priorities[i]
      self._heapify(smallest)

  def _improve_key(self, i):
    while i > 0 and self.priorities[(i - 1) // 2] > self.priorities[i]:
      parent = (i - 1) // 2
      self.items[i], self.items[parent] = self.items[parent], self.items[i]
      self.priorities[i], self.priorities[parent] = self.priorities[
          parent], self.priorities[i]
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
      exponent << BITS_PER_FLOAT_SIGNIFICAND)
  return result if f >= 0 else -result


def int_to_float(i):
  v = abs(i)
  significand = v & ((1 << BITS_PER_FLOAT_SIGNIFICAND) - 1)
  exponent = v >> BITS_PER_FLOAT_SIGNIFICAND
  return math.ldexp(significand / (1 << BITS_PER_FLOAT_SIGNIFICAND), exponent -
                    BITS_PER_FLOAT_SIGNIFICAND + 1) * (1 if i >= 0 else -1)


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


def make_4_directions_enumerator(node_width=1,
                                 node_height=1,
                                 min_x=0,
                                 min_y=0,
                                 max_x=None,
                                 max_y=None):
  max_x = max_x if max_x is not None else float('inf')
  max_y = max_y if max_y is not None else float('inf')

  def enumerator(x, y, func):
    for dx, dy in [(0, -node_height), (0, node_height), (-node_width, 0),
                   (node_width, 0)]:
      next_x = x + dx
      next_y = y + dy
      if min_x <= next_x < max_x and min_y <= next_y < max_y:
        func(next_x, next_y)

  return enumerator


def make_8_directions_enumerator(node_width=1,
                                 node_height=1,
                                 min_x=0,
                                 min_y=0,
                                 max_x=None,
                                 max_y=None):
  max_x = max_x if max_x is not None else float('inf')
  max_y = max_y if max_y is not None else float('inf')

  def enumerator(x, y, func):
    for dx, dy in [(node_width, 0), (node_width, -node_height),
                   (0, -node_height), (-node_width, -node_height),
                   (-node_width, 0), (-node_width, node_height),
                   (0, node_height), (node_width, node_height)]:
      next_x = x + dx
      next_y = y + dy
      if min_x <= next_x < max_x and min_y <= next_y < max_y:
        func(next_x, next_y)

  return enumerator


def make_manhattan_distance_heuristic(scale_factor=1.0):
  return lambda x1, y1, x2, y2: scale_factor * (abs(x1 - x2) + abs(y1 - y2))


def make_octile_distance_heuristic(scale_factor=1.0):
  sqrt2 = math.sqrt(2)
  return lambda x1, y1, x2, y2: scale_factor * (min(abs(x1 - x2), abs(
      y1 - y2)) * sqrt2 + abs(abs(x1 - x2) - abs(y1 - y2)))


def make_euclidean_distance_heuristic(scale_factor=1.0):
  return lambda x1, y1, x2, y2: scale_factor * math.sqrt((x1 - x2)**2 +
                                                         (y1 - y2)**2)


def define_path_finder(name,
                       world_size,
                       frontier_size=500,
                       coordinate_type='float',
                       coordinate_encoder=encode_float_coordinates,
                       coordinate_decoder=decode_float_coordinates,
                       indexer=None,
                       goal_reached_p=None,
                       neighbour_enumerator=None,
                       exact_cost=None,
                       heuristic_cost=None,
                       max_movement_cost=float('inf'),
                       path_initiator=lambda length: None,
                       path_processor=lambda x, y: None,
                       path_finalizer=lambda: True):

  def path_finder(start_x, start_y, goal_x, goal_y, **params):
    cost_so_far = [float('nan')] * world_size
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

      def process_neighbour(next_x, next_y):
        next_index = indexer(next_x, next_y)
        new_cost = cost_so_far[current_index] + exact_cost(
            current_x, current_y, next_x, next_y)
        if new_cost < max_movement_cost and (math.isnan(
            cost_so_far[next_index]) or new_cost < cost_so_far[next_index]):
          cost_so_far[next_index] = new_cost
          came_from[next_index] = current
          frontier.insert(
              coordinate_encoder(next_x, next_y),
              new_cost + heuristic_cost(next_x, next_y, goal_x, goal_y))

      neighbour_enumerator(current_x, current_y, process_neighbour)

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
    stack = [start]
    visited = set()
    iterations = 0

    while stack and iterations < max_iterations:
        iterations += 1
        x, y = stack.pop()

        if (x, y) == goal:
            return True

        if (x, y) in visited or maze[y, x] == 1:
            continue

        visited.add((x, y))

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[1] and 0 <= ny < maze.shape[0]:
                stack.append((nx, ny))

    return False


def create_dla_maze(width, height):
    maze = np.zeros((height, width), dtype=int)
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1

    num_seeds = random.randint(max(3, min(width, height) // 20), max(10, min(width, height) // 5))
    for _ in range(num_seeds):
        x, y = random.randint(1, width - 2), random.randint(1, height - 2)
        maze[y, x] = 1

    num_particles = random.randint(width * height // 8, width * height // 2)
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
                neighbors = maze[y - 1:y + 2, x - 1:x + 2].sum() - maze[y, x]
                if maze[y, x] == 1:
                    if neighbors < 2 or neighbors > 3:
                        new_maze[y, x] = 0
                else:
                    if neighbors == 3:
                        new_maze[y, x] = 1
        maze = new_maze

    return maze


def create_one_dim_automata_maze(width, height):
    maze = np.zeros((height, width), dtype=int)
    maze[0] = np.random.choice([0, 1], size=width)
    maze[0, 0] = maze[0, -1] = 1

    rule_number = random.randint(0, 255)
    rule = {(a, b, c): (rule_number >> (a * 4 + b * 2 + c)) & 1
            for a in (0, 1) for b in (0, 1) for c in (0, 1)}

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

        if ant_direction == 0: ant_y = max(1, ant_y - 1)
        elif ant_direction == 1: ant_x = min(width - 2, ant_x + 1)
        elif ant_direction == 2: ant_y = min(height - 2, ant_y + 1)
        else: ant_x = max(1, ant_x - 1)

    return maze

def create_voronoi_maze(width, height):
    num_points = random.randint(max(width, height) // 8, max(width, height) // 2)
    points = np.random.rand(num_points, 2) * [width, height]
    vor = Voronoi(points)

    maze = np.ones((height, width), dtype=int)
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:
            p1, p2 = vor.vertices[simplex]
            rr, cc = line(int(p1[1]), int(p1[0]), int(p2[1]), int(p2[0]))
            valid_indices = (rr < height) & (cc < width)
            maze[rr[valid_indices], cc[valid_indices]] = 0

    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
    return maze


def create_fractal_maze(width, height, min_size=None):
    if min_size is None:
        min_size = random.randint(4, 12)

    def recursive_divide(x, y, w, h):
        if w <= min_size or h <= min_size:
            return

        if random.random() < 0.5:
            divide_horizontally = w > h
        else:
            divide_horizontally = random.choice([True, False])

        if divide_horizontally:
            divide_at = random.randint(x + min_size, x + w - min_size)
            maze[:, divide_at] = 1
            opening = random.randint(y, y + h - 1)
            maze[opening, divide_at] = 0
            recursive_divide(x, y, divide_at - x, h)
            recursive_divide(divide_at + 1, y, x + w - divide_at - 1, h)
        else:
            divide_at = random.randint(y + min_size, y + h - min_size)
            maze[divide_at, :] = 1
            opening = random.randint(x, x + w - 1)
            maze[divide_at, opening] = 0
            recursive_divide(x, y, w, divide_at - y)
            recursive_divide(x, divide_at + 1, w, y + h - divide_at - 1)

    maze = np.zeros((height, width), dtype=int)
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
    recursive_divide(1, 1, width - 2, height - 2)
    return maze


def create_maze_from_image(width, height):
  image_path='' # Enter file path here
  # Load and resize the image
  img = Image.open(image_path).convert('L')
  img = img.resize((width, height))

  # Convert to numpy array and threshold
  maze = np.array(img)
  maze = (maze > 128).astype(int)

  # Ensure borders are walls
  maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
  return maze


def create_wave_function_collapse_maze(width, height):
    tiles = {
        0: {'up': [0, 2], 'right': [0, 1], 'down': [0, 2], 'left': [0, 1]},
        1: {'up': [1], 'right': [1], 'down': [1], 'left': [1]},
        2: {'up': [2], 'right': [2], 'down': [2], 'left': [2]},
    }

    def get_valid_tiles(x, y):
        valid = set(tiles.keys())
        for dx, dy, direction in [(0, -1, 'down'), (1, 0, 'left'), (0, 1, 'up'), (-1, 0, 'right')]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and maze[ny, nx] != -1:
                valid &= set(tiles[maze[ny, nx]][direction])
        return list(valid)

    maze = np.full((height, width), -1, dtype=int)
    stack = [(random.randint(1, width - 2), random.randint(1, height - 2))]

    while stack:
        x, y = stack.pop(random.randint(0, len(stack) - 1))
        valid_tiles = get_valid_tiles(x, y)
        if valid_tiles:
            maze[y, x] = random.choice(valid_tiles)
            random.shuffle([(0, -1), (1, 0), (0, 1), (-1, 0)])
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < width - 1 and 0 < ny < height - 1 and maze[ny, nx] == -1:
                    stack.append((nx, ny))

    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
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
        directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < width - 1 and 0 < ny < height - 1 and maze[ny, nx] == 1:
                maze[ny, nx] = maze[y + dy // 2, x + dx // 2] = 0
                stack.append((nx, ny))
                break

    return maze


def create_terrain_based_maze(width, height):
    scale = random.uniform(0.05, 0.2)
    octaves = random.randint(4, 8)
    persistence = random.uniform(0.4, 0.6)
    lacunarity = random.uniform(1.5, 2.5)

    terrain = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            terrain[y, x] = snoise2(x * scale, y * scale, octaves, persistence, lacunarity)

    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    threshold = random.uniform(0.4, 0.6)
    maze = (terrain > threshold).astype(int)

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
    else:
        raise ValueError(f"Unknown maze generation approach: {maze_generation_approach}")


def generate_solvable_maze(width, height, maze_generation_approaches, max_attempts=100, timeout=30):
    print(f"Attempting to generate a solvable maze...")
    start_time = time.time()
    attempts = 0

    while attempts < max_attempts and time.time() - start_time < timeout:
        attempts += 1
        print(f"Attempt {attempts}...")
        
        # Randomly choose a maze generation approach
        maze_generation_approach = random.choice(maze_generation_approaches)
        print(f"Using {maze_generation_approach} approach...")
        
        maze = create_better_maze(width, height, maze_generation_approach)
        
        # Ensure the start and goal are not walls
        maze[1, 1] = maze[height-2, width-2] = 0
        start = (1, 1)
        goal = (width-2, height-2)

        print(f"Checking if maze is solvable...")
        if is_maze_solvable(maze, start, goal):
            print(f"Solvable maze generated after {attempts} attempts and {time.time() - start_time:.2f} seconds")
            return maze, start, goal

    print(f"Failed to generate a solvable maze after {attempts} attempts and {time.time() - start_time:.2f} seconds")
    return None, None, None


def run_complex_examples(override_maze_approach=None):
    GRID_SIZE = 31
    world_size = GRID_SIZE * GRID_SIZE
    num_problems = 2
    DPI = 100

    wall_color = '#2C3E50'
    floor_color = '#ECF0F1'
    start_color = '#27AE60'
    goal_color = '#E74C3C'
    path_color = '#3498DB'

    exploration_colors = ['#FFF9C4', '#FFE082', '#FFB74D', '#FF8A65', '#E57373']
    exploration_cmap = LinearSegmentedColormap.from_list('exploration', exploration_colors, N=100)

    fig, axs = plt.subplots(1, num_problems, figsize=(20, 8), dpi=DPI)
    fig.suptitle('Advanced Pathfinding Visualization', fontsize=16)

    all_exploration_orders = []
    all_paths = []
    all_mazes = []
    all_starts = []
    all_goals = []
    all_maze_approaches = []

    maze_generation_approaches = [
        "dla", "random_game_of_life", "random_one_dim_automata",
        "langtons_ant", "voronoi", "fractal",
        "wave_function_collapse", "growing_tree", "terrain"
    ]

    for i in range(num_problems):
        if override_maze_approach:
            maze_generation_approach = override_maze_approach
        else:
            maze_generation_approach = random.choice(maze_generation_approaches)

        all_maze_approaches.append(maze_generation_approach)

        print(f"Starting maze generation for problem {i+1} using approach {maze_generation_approach}...")
        maze, start, goal = generate_solvable_maze(GRID_SIZE, GRID_SIZE, maze_generation_approach)
        
        if maze is None:
            print(f"Failed to generate a solvable maze for problem {i+1}. Skipping this problem.")
            continue
        
        start_x, start_y = start
        goal_x, goal_y = goal
        print("Maze generation complete.")

        obstacles = set((x, y) for y, row in enumerate(maze) for x, cell in enumerate(row) if cell)

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

                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    next_x, next_y = current_x + dx, current_y + dy
                    if 0 <= next_x < GRID_SIZE and 0 <= next_y < GRID_SIZE and (next_x, next_y) not in obstacles:
                        new_cost = cost_so_far[current] + (1 if abs(dx) + abs(dy) == 1 else 1.414)
                        next_node = encode_integer_coordinates(next_x, next_y)

                        if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                            cost_so_far[next_node] = new_cost
                            priority = new_cost + math.sqrt((goal_x - next_x)**2 + (goal_y - next_y)**2)
                            frontier.insert(next_node, priority)
                            came_from[next_node] = current
                            exploration_order.append((next_x, next_y))

            # Check if a path was found
            if encode_integer_coordinates(goal_x, goal_y) not in came_from:
                print(f"No path found from ({start_x}, {start_y}) to ({goal_x}, {goal_y})")
                return False

            # Reconstruct path
            path.clear()  # Clear existing path
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
            print(f"Failed to find a path for problem {i+1}. Skipping visualization for this problem.")
            continue
        print("Pathfinding complete.")

        print(f"Problem {i+1}:")
        print(f"  Maze generation approach: {maze_generation_approach}")
        print(f"  Exploration order length: {len(exploration_order)}")
        print(f"  Path length: {len(path)}")

        all_exploration_orders.append(exploration_order)
        all_paths.append(path)
        all_mazes.append(maze)
        all_starts.append((start_x, start_y))
        all_goals.append((goal_x, goal_y))

    def make_frame(frame):
        artists = []
        for i in range(len(all_mazes)):  # Changed from num_problems to len(all_mazes)
            ax = axs[i] if len(all_mazes) > 1 else axs
            ax.clear()

            colored_maze = np.where(all_mazes[i] == 1, 1, 0)
            wall_color_rgba = [44/255, 62/255, 80/255, 1]
            floor_color_rgba = [236/255, 240/255, 241/255, 1]
            maze_rgba = np.zeros((*colored_maze.shape, 4))
            maze_rgba[colored_maze == 1] = wall_color_rgba
            maze_rgba[colored_maze == 0] = floor_color_rgba

            im = ax.imshow(maze_rgba)
            artists.append(im)

            exploration_map = np.zeros((GRID_SIZE, GRID_SIZE))
            exploration_length = len(all_exploration_orders[i])
            path_length = len(all_paths[i])
            total_frames = exploration_length + path_length

            if frame < exploration_length:
                for idx, (x, y) in enumerate(all_exploration_orders[i][:frame]):
                    exploration_map[y, x] = idx + 1

                if exploration_map.max() > 0:
                    exploration_map = exploration_map / exploration_map.max()

                im2 = ax.imshow(exploration_map, cmap=exploration_cmap, alpha=0.5)
                artists.append(im2)
            else:
                path_frame = frame - exploration_length
                if path_frame < path_length:
                    path_segment = all_paths[i][:path_frame + 1]
                    path_x, path_y = zip(*path_segment)
                    path_plot, = ax.plot(path_x, path_y, color=path_color, linewidth=2, alpha=0.8)
                    artists.append(path_plot)

            start_x, start_y = all_starts[i]
            goal_x, goal_y = all_goals[i]
            start_marker = ax.plot(start_x, start_y, 'o', color=start_color, markersize=10, label='Start')
            goal_marker = ax.plot(goal_x, goal_y, 'o', color=goal_color, markersize=10, label='Goal')
            artists.extend(start_marker + goal_marker)

            ax.set_title(f'Example {i+1}: {all_maze_approaches[i]}')
            ax.axis('off')
            if i == 0:
                legend = ax.legend(loc='upper left', bbox_to_anchor=(0, -0.1))
                artists.append(legend)

            progress = min(frame / total_frames, 1.0)
            progress_text = ax.text(0.5, -0.1, f'Progress: {progress:.1%}', transform=ax.transAxes, ha='center')
            artists.append(progress_text)

        plt.tight_layout()
        return artists

    if not all_paths:
        print("No valid paths found. Exiting without creating animation.")
        return

    max_frames = max(len(exploration_order) + len(path) for exploration_order, path in zip(all_exploration_orders, all_paths))
    print(f"Max frames: {max_frames}")

    # Ensure we have at least one frame
    max_frames = max(1, max_frames)

    # Generate frames with a progress bar
    frames = list(tqdm(range(max_frames), desc="Generating frames"))

    anim = FuncAnimation(fig, make_frame, frames=frames, interval=100, blit=True)

    # Save as MP4
    writer = FFMpegWriter(fps=20, bitrate=5000, codec='libx264')
    print("Saving MP4...")
    anim.save("pathfinding_visualization.mp4", writer=writer, dpi=DPI)

    # Save as GIF
    writer2 = PillowWriter(fps=20)
    print("Saving GIF...")
    anim.save("pathfinding_visualization.gif", writer=writer2, dpi=DPI)
    print("Animation saved as 'pathfinding_visualization.mp4' and 'pathfinding_visualization.gif'")

    plt.close(fig)

if __name__ == "__main__":
  # Example usage:
  # To use a specific maze generation approach for all problems:
  # run_complex_examples(override_maze_approach="fractal")

  # To use random selection for each problem:
  run_complex_examples()
