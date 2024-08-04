# Advanced Pathfinding and Maze Generation

This project provides a high-performance implementation of the A\* ("A-Star") pathfinding algorithm along with various maze generation techniques to showcase how this algorithm works, as well as an advanced animated visualization of pathfinding in these mazes. The mazes are generated using different approaches, each providing unique characteristics and challenges. The A\* algorithm is designed to efficiently find the shortest path in these mazes, taking into consideration various heuristic functions and neighbor enumerators.

## Features

- **Optimized A\* Pathfinding**: Includes custom priority queue and efficient state handling for both integer and float coordinates.
- **Diverse Maze Generation**: Multiple algorithms for creating complex and varied mazes, including cellular automata, fractal generation, Voronoi diagrams, and more.
- **Advanced Visualization**: Detailed visual representation of maze generation and pathfinding, including animation of exploration and path discovery.

![A\* Visualized in Python!](https://raw.githubusercontent.com/Dicklesworthstone/visual_astar_python/main/illustration.webp)

## Pathfinding Implementation

### Design Philosophy and Performance

The A\* algorithm implementation focuses on efficiency and scalability. Key aspects include:

1. **Custom Priority Queue**: The priority queue is a fundamental component of the A\* algorithm, used to manage the open set (frontier) of nodes to be explored. In this implementation, the priority queue is optimized for fast insertion and extraction of elements based on their priority values, which represent the estimated total cost (distance traveled + heuristic) to reach the goal. This allows the algorithm to quickly focus on the most promising nodes.

2. **Coordinate Encoding**: The system supports both integer and float coordinates, which are encoded efficiently to optimize memory usage and computation. This encoding process involves converting floating-point coordinates into a unique integer representation, ensuring precise and quick decoding. The encoding scheme supports a wide range of values, accommodating both fine-grained precision and large-scale maps.

3. **Heuristic Functions**: A variety of heuristic functions are available, including Manhattan, Octile, and Euclidean distance heuristics. Each heuristic offers a different way to estimate the cost to reach the goal from a given node, balancing accuracy with computational efficiency. The choice of heuristic can significantly affect the performance of the A\* algorithm, with more accurate heuristics generally leading to faster pathfinding at the cost of additional computation.

4. **Neighbor Enumeration**: The algorithm provides customizable neighbor enumerators that define how neighboring nodes are considered during the pathfinding process. Options include 4-directional, 8-directional, and more complex movement patterns. This flexibility allows the algorithm to handle various types of terrain and movement costs, such as diagonal movement being more expensive than orthogonal movement.

### Exact and Heuristic Cost Functions

- **Exact Cost**: This function calculates the actual cost of moving from one node to another. It can account for various factors, such as the distance between nodes and any penalties associated with certain types of terrain or movement. For instance, moving diagonally may have a higher cost than moving vertically or horizontally.
- **Heuristic Cost**: The heuristic cost is an estimate of the cost to reach the goal from a given node. It serves as a guide to the A\* algorithm, helping it prioritize nodes that are likely closer to the goal. The accuracy and computational cost of the heuristic can vary; a more accurate heuristic may provide better guidance but require more computation.

## Maze Generation Methods

This project includes a rich variety of maze generation algorithms, each creating unique patterns and challenges. Below is a detailed explanation of each method:

1. **Diffusion-Limited Aggregation (DLA)**: This method simulates the random motion of particles that stick upon contact, forming complex, fractal-like structures. The algorithm starts with a few initial seed particles and progressively aggregates new particles that wander randomly until they attach to the existing structure, resulting in an intricate maze with natural, organic formations.

2. **Game of Life**: Based on Conway's Game of Life, this method uses cellular automata rules to evolve a grid of cells. Each cell can either be alive or dead, with the state of each cell in the next generation determined by the number of alive neighbors. The rules simulate life, death, and reproduction, leading to evolving and dynamic patterns that create unpredictable maze structures as the simulation progresses.

3. **One-Dimensional Automata**: This method generates mazes by applying simple rules to a one-dimensional row of cells, which then evolve over multiple generations. The state of each cell in the next row depends on the current state of itself and its two neighbors, following a specific rule set (encoded as a rule number). The result is a variety of maze patterns, from simple stripes to complex fractal-like designs.

4. **Langton's Ant**: A simulation of a simple Turing machine that moves on a grid of black and white cells. The "ant" follows a set of rules: turn left or right based on the color of the cell, flip the color of the cell, and move forward. Despite its simplicity, this algorithm creates intricate and often chaotic paths, demonstrating how simple rules can lead to complex behavior.

5. **Voronoi Diagram**: This method generates mazes by dividing the space into regions based on the distance to a set of randomly placed points. Each region corresponds to all locations closer to one specific point than any other, creating natural-looking, polygonal areas. The edges between these regions can be converted into walls, forming a maze with an organic and often irregular layout.

6. **Fractal Division**: A recursive method that divides the grid into smaller regions, introducing walls along the division lines. The process continues until the regions are sufficiently small, creating a fractal-like pattern. This method often results in highly symmetric and recursive maze structures, where paths and walls exhibit self-similar patterns at different scales.

7. **Wave Function Collapse**: Inspired by quantum mechanics, this constraint-based method starts with a grid of possible states and progressively collapses them into a single state based on neighboring constraints. This ensures that the resulting pattern is consistent and non-contradictory, producing complex and aesthetically pleasing mazes with well-defined structures.

8. **Growing Tree**: This algorithm selects a random starting point and grows the maze by iteratively adding new paths. It maintains a list of frontier cells, choosing cells to expand from based on a random or specific strategy (e.g., last-in-first-out, first-in-first-out). The result is a maze with a branching, tree-like structure, where paths extend outward from the starting point.

9. **Terrain-Based**: Utilizes Perlin noise to generate a heightmap, which is then thresholded to create walls and paths. Perlin noise provides a naturalistic texture, simulating rolling hills or rugged terrain. The resulting maze resembles natural landscapes, with smooth transitions and organic features, offering a unique challenge for pathfinding.

10. **Musicalized**: This method leverages harmonic functions to generate patterns that resemble musical rhythms. By combining multiple sine waves of different frequencies, the algorithm creates a grid where the amplitude values are thresholded to form walls and paths. The generated mazes have a rhythmic and wave-like quality, inspired by musical compositions.

11. **Quantum-Inspired**: Mimics quantum interference patterns by superimposing wave functions and calculating the resulting probability densities. The method visualizes quantum phenomena, such as wave superposition and interference, by thresholding the probability densities to form maze walls. The resulting mazes are visually stunning, with intricate and delicate patterns.

12. **Artistic**: Draws inspiration from artistic techniques, such as brush strokes and splatter effects. The algorithm randomly places strokes and splatters on a canvas, using these elements to define the maze's walls and paths. The result is an abstract, artistic maze, where the design mimics various art styles and techniques, providing a unique visual experience.

13. **Cellular Automaton**: Similar to the Game of Life, this method uses custom rules to evolve a grid of cells. However, it allows for more complex rulesets and neighbor interactions, resulting in a wider variety of patterns. The algorithm iteratively applies the rules to generate intricate structures that can vary from chaotic to highly ordered, depending on the chosen ruleset.

14. **Fourier-Based**: Applies the Fourier transform to a random noise field, selectively filtering out certain frequency components. The inverse transform then produces a pattern with a controlled frequency spectrum, which is thresholded to form walls. This method creates mazes with smooth, flowing patterns and can simulate various textures by adjusting the frequency components.

15. **Reaction-Diffusion**: Simulates a chemical reaction and diffusion process on a grid, where two substances interact to form complex patterns. The algorithm iteratively updates the concentration of substances based on local reactions and diffusion, leading to the emergence of stable patterns. The resulting maze is formed by thresholding these patterns, creating organic and biomorphic structures reminiscent of natural forms.

Each method in this collection offers a distinct visual and structural style, making it possible to explore a wide range of maze characteristics and challenges. These mazes are suitable for testing various pathfinding algorithms and for generating visually compelling visualizations.

## Visualization

The visualization component in this project is designed to provide a comprehensive and interactive display of both the maze generation process and the pathfinding algorithms at work. This component uses the `matplotlib` library to create detailed visual representations that highlight the complexities and intricacies of maze structures and pathfinding strategies. Key elements of the visualization include:

### Maze Structure

- **Walls and Floors**: The visualization distinctly represents walls and floors using a two-color scheme. Walls are typically rendered in a dark color (e.g., deep blue or gray), while floors are displayed in a contrasting light color (e.g., white or light gray). This clear differentiation helps users easily identify passable and impassable areas within the maze.

- **Color Mapping**: The code allows for the customization of wall and floor colors. This is particularly useful for creating visual themes or adjusting the visualization for different viewing conditions (e.g., color blindness). The `LinearSegmentedColormap` from `matplotlib` can be used to define custom gradients for different maze elements.

### Pathfinding Progress

- **Exploration Order**: During the pathfinding process, the visualization dynamically displays the exploration order of the algorithm. This is achieved by coloring explored cells using a gradient that represents the progression of exploration. Lighter shades indicate earlier exploration, while darker shades denote later exploration stages. The use of an exploration colormap helps visualize the pathfinding algorithm's exploration strategy and efficiency.

- **Path Discovery**: As the algorithm discovers the path from the start to the goal, the visualization highlights the path using a distinct color (e.g., blue or green). The path is typically represented as a continuous line, indicating the sequence of cells that constitute the solution. The visualization updates in real-time, allowing viewers to see how the path evolves as the algorithm progresses.

- **Markers for Start and Goal Points**: The start and goal points are clearly marked with distinct symbols (e.g., circles or stars) and colors (e.g., green for the start, red for the goal). These markers remain visible throughout the visualization, providing consistent reference points for the viewer.

### Customizable Colors

- **Customization Options**: The visualization component offers extensive customization options for colors, allowing users to adjust the appearance of walls, floors, paths, exploration stages, and start/goal markers. This customization is facilitated through parameters passed to the visualization functions, enabling users to tailor the display to their preferences or specific use cases.

- **Colormap Selection**: For the exploration and path colors, users can select from predefined colormaps or create custom ones using `LinearSegmentedColormap`. This flexibility ensures that the visualization can be adapted to various aesthetic preferences or accessibility needs.

- **Transparency and Layering**: The visualization supports transparency and layering effects, particularly for the exploration map. By adjusting the alpha value, users can overlay the exploration progress on top of the maze structure without obscuring the underlying details. This feature is useful for simultaneously visualizing the explored area and the structural layout of the maze.

### Animation and Export

- **Frame Generation**: The visualization is animated by generating frames that capture the state of the maze and pathfinding process at each time step. The code uses concurrent processing to efficiently generate these frames, leveraging multiple CPU cores for faster rendering. Each frame is created by plotting the maze, exploration progress, and current path status.

- **Animation Playback**: The frames are compiled into an animation using `FuncAnimation` from `matplotlib.animation`. The playback speed can be adjusted by setting the frames per second (FPS), allowing for slower or faster visualization of the pathfinding process. The animation provides a smooth and continuous representation of the algorithm's operation, from initial exploration to final pathfinding.

- **Output Formats**: The animations can be exported in various formats, including MP4 and GIF. For MP4 exports, the `FFMpegWriter` is used, allowing for fine-tuned control over encoding parameters, such as bitrate and codec settings. This ensures high-quality video output suitable for presentations or further analysis. The GIF export option provides a more lightweight and easily shareable format, though it may require adjustments to frame rate and resolution for optimal quality.

- **Resource Management**: To manage disk space and avoid clutter, the code includes functionality to delete small or temporary files after the animation is saved. This helps maintain a clean working directory and ensures that only the most relevant files are retained.

## Usage

### Initial Setup

Clone the repo and set up a virtual environment with the required packages using (tested on Python 3.12 and Ubuntu 22):

```bash
git clone https://github.com/Dicklesworthstone/visual_astar_python.git
cd visual_astar_python
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install wheel
python -m pip install --upgrade setuptools wheel
pip install -r requirements.txt
```

### Generating and Visualizing Mazes

To generate and visualize a maze, run the main script with the desired parameters:

```bash
python main.py
```

### Customization

- **Maze Generation Approach**: Specify the desired maze generation approach (e.g., `dla`, `wave_function_collapse`) to customize the type of maze generated.
- **Grid Size**: Set the `GRID_SIZE` parameter to adjust the size of the maze.
- **Visualization Settings**: Modify color schemes and animation settings to suit your preferences.

## Parameter Configuration

This project includes several parameters that users can configure to customize the output of maze generation and pathfinding visualization. Key parameters include:

- **num_animations**: Determines the number of separate animations to generate. Each animation can feature different mazes and pathfinding scenarios.
- **GRID_SIZE**: Specifies the resolution of the maze grid, defining the number of cells along one dimension. Higher grid sizes create more complex and detailed mazes.
- **num_problems**: Sets the number of mazes displayed side by side in each animation. This allows for simultaneous comparison of different maze generation methods or pathfinding strategies.
- **DPI**: Determines the dots per inch (DPI) for the animation, affecting the resolution and quality of the output. Higher DPI values produce sharper images but may increase rendering time.
- **FPS**: Controls the frames per second (FPS) for the animation playback. A higher FPS results in smoother animations but may require more processing power and storage.

These parameters provide users with extensive control over the behavior and appearance of the generated mazes and visualizations, allowing for fine-tuning according to specific requirements or preferences.

## A\* Algorithm: Theoretical Overview and Advanced Implementation

### The Big Ideas Behind A\*

The A\* algorithm excels at finding the shortest path in a wide range of scenarios by being an intelligent and goal-oriented navigator. Imagine you're navigating through a labyrinth or a complex game environment; A\* doesn't just wander aimlessly or follow the first path it finds. Instead, it strategically evaluates its options, much like a skilled navigator plotting the best route on a map. This approach ensures that A\* not only finds a path but finds the most efficient one, saving time and resources.

At its core, A\* makes smart, informed decisions by constantly assessing the potential of different paths. Think of it as a savvy traveler who, at each crossroads, checks both how far they've already traveled and how much further they need to go to reach their destination. This dual consideration is what sets A\* apart from simpler algorithms, which might only focus on one aspect—like just moving forward without considering the overall distance left.

A\* balances two key considerations:

1. **The Journey So Far**: This involves calculating the actual cost of traveling along the current path to reach a particular point. It's like tallying up the miles you've driven on a road trip. This helps A\* understand the exact effort taken to get to the current position, ensuring it doesn't overlook more efficient paths that might be slightly longer but quicker in the long run.

2. **The Journey Ahead**: Here, A\* makes an educated guess about the remaining distance to the goal, known as the heuristic estimate. This isn't a wild guess—it's a calculated one that uses available information to predict how close or far the goal is. For example, in a grid-based maze, the heuristic might consider the straight-line distance left to travel, giving a quick but reliable estimate of the remaining journey.

By integrating these two aspects, A\* effectively "thinks ahead." It doesn't blindly follow the nearest path or the least expensive immediate option; instead, it carefully weighs both the cost so far and the estimated cost to reach the goal. This dual focus allows A\* to efficiently prioritize paths that seem promising, avoiding those that might look attractive in the short term but lead to longer, less efficient routes.

**Why A\* Stands Out Compared to Other Algorithms**

A\* offers significant advantages over simpler pathfinding algorithms, such as Depth-First Search (DFS), Breadth-First Search (BFS), and Dijkstra's algorithm:

- **Depth-First Search (DFS)** tends to dive deep into one path before exploring others, often resulting in long, convoluted paths. It doesn't have a built-in mechanism for considering the remaining distance to the goal, which can lead to inefficient searches and, sometimes, failure to find a path at all.

- **Breadth-First Search (BFS)** explores all possible paths layer by layer. While it guarantees finding the shortest path in an unweighted graph, it does so without any sense of direction or goal proximity, making it inefficient in terms of time and memory usage in large or complex spaces.

- **Dijkstra's Algorithm** is a close relative of A\* and also finds the shortest path by considering the cost so far. However, Dijkstra's algorithm does not include a heuristic for estimating the remaining distance to the goal. This lack of foresight means that Dijkstra's explores every possible path evenly, without prioritizing those more likely to lead to the goal quickly.

**A\* vs. Simpler Alternatives**

What makes A\* particularly powerful is its heuristic, which serves as a foresight tool. This heuristic enables A\* to make decisions that bring it closer to the goal more quickly, avoiding the exhaustive and sometimes unnecessary exploration that algorithms like Dijkstra's might perform. In essence, A\* blends the thoroughness of Dijkstra's with the goal-directed approach of informed search strategies, resulting in a method that is both efficient and effective.

For instance, in a video game, A\* can guide characters through complex environments, navigating around obstacles and towards objectives with uncanny efficiency. In robotics, A\* helps autonomous robots find the fastest route through a factory floor, avoiding collisions and minimizing travel time.

In summary, A\* is not just a pathfinder—it's a path optimizer. By considering both the path traveled and the estimated journey ahead, it excels at finding the shortest and most efficient route in complex environments. This dual focus on the present situation and future possibilities makes A\* a preferred choice for many real-world applications where efficiency and precision are crucial.

### Why This Implementation Shines

The implementation of A\* in this project goes beyond the basics, making it not only effective but also exceptionally efficient. It is heavily based on the lisp implementation given [here](https://gitlab.com/lockie/cl-astar) by Andrew Kravchuk. Here are some of the features of this implementation:

1. **Smart Organization**: It uses a special system to keep track of all possible paths, making sure that the most promising ones are always at the top of the list. This system is fast and organized, so the algorithm spends less time figuring out where to go next.

2. **Precision Handling**: It can handle both simple, grid-like paths and more complex, floating-point coordinates. This flexibility means it can work accurately in detailed, large-scale environments, making it versatile for various applications.

3. **Adaptive Strategy**: The algorithm can adapt to different situations by using different methods to estimate the remaining distance to the goal. Whether the environment is straightforward or complex, it can adjust its strategy to find the best path efficiently.

4. **Navigating Complex Terrains**: It’s equipped to handle different kinds of movements, whether it's straightforward like moving up, down, left, or right, or more complex, including diagonal or custom paths. This makes it incredibly adaptable, capable of navigating through a variety of terrains and obstacles.

5. **Efficient Path Reconstruction**: Once it finds the destination, A\* can quickly retrace its steps to map out the optimal path. This backtracking is done in a streamlined way, ensuring that the final path is as direct as possible.

6. **Built-in Safeguards**: The implementation is robust, designed to handle tricky situations gracefully. Whether it's encountering a dead end or an unreachable goal, it manages these scenarios without crashing or getting stuck, providing clear feedback on what's happening.

7. **Efficient Data Representation**: It uses bit fields for encoding these coordinates, which streamlines the process of storing and retrieving node information. This bit-level manipulation not only saves memory but also enhances processing speed, as bitwise operations are computationally inexpensive.

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
- SciPy
- Scikit-Image
- Noise
- Pillow
- TQDM
- FFmpeg (for video encoding)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
