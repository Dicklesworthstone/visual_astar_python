# Visual A\* Pathfinding and Maze Generation in Python

This project provides a high-performance implementation of the A\* ("A-Star") pathfinding algorithm (based on [this](https://gitlab.com/lockie/cl-astar) Lisp implementation by Andrew Kravchuck) along with various maze generation techniques to showcase how this algorithm works, as well as an advanced animated visualization of pathfinding in these mazes. The mazes are generated using different approaches, each providing unique characteristics and challenges. The A\* algorithm is designed to efficiently find the shortest path in these mazes, taking into consideration various heuristic functions and neighbor enumerators.

## Features

- **Optimized A\* Pathfinding**: Includes custom priority queue and efficient state handling for both integer and float coordinates.
- **Diverse Maze Generation**: Multiple algorithms for creating complex and varied mazes, including Diffusion-Limited Aggregation (DLA), Game of Life, One-Dimensional Automata, Langton's Ant, Voronoi Diagrams, Fractal Division, Wave Function Collapse, Growing Tree, Terrain-Based, Musicalized, Quantum-Inspired, Artistic, Cellular Automaton, Fourier-Based, and Reaction-Diffusion.
- **Advanced Visualization**: Detailed visual representation of maze generation and pathfinding, including animation of exploration and path discovery.

<div align="center">
  <a href="https://www.youtube.com/watch?v=iA6XJRE6CTM">
    <img src="https://img.youtube.com/vi/iA6XJRE6CTM/0.jpg" alt="Demo of it in Action">
  </a>
  <br>
  <em>Demo of it in Action (click thumbnail for YouTube video!)</em>
</div>

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

### Maze Generation Methods

This project includes a rich variety of maze generation algorithms, each creating unique patterns and challenges. Below is a detailed explanation of each method:

1. **Diffusion-Limited Aggregation (DLA)**:
   - **Description**: DLA is a process that simulates the random motion of particles in a medium until they stick to a surface or to each other, forming aggregates. In this algorithm, particles start from random positions and move randomly until they either stick to an existing structure or fall off the boundary of the defined space.
   - **Mechanism**: The algorithm initializes with a few seed particles on the grid. New particles are introduced at random locations and follow a random walk. When a particle encounters an occupied cell (another particle), it sticks to it, thereby growing the aggregate structure. This process results in intricate, tree-like patterns, which can resemble natural formations like snowflakes or mineral deposits.

2. **Game of Life**:
   - **Description**: Based on Conway's Game of Life, this method uses cellular automata rules to evolve a grid of cells, where each cell can be either alive or dead. The next state of each cell is determined by its current state and the number of alive neighbors it has.
   - **Mechanism**: The grid is initialized with a random configuration of alive (1) and dead (0) cells. The state of each cell in the next generation is determined by counting its alive neighbors. Cells with exactly three alive neighbors become alive, while cells with fewer than two or more than three alive neighbors die. This evolution creates dynamic and unpredictable patterns, often resulting in maze-like structures with complex corridors and dead-ends.

3. **One-Dimensional Automata**:
   - **Description**: This method involves the use of simple rules applied to a single row of cells (1D) which then evolves over time to form a 2D maze pattern. The rule set, often represented as a binary number, dictates the state of a cell based on the states of its neighbors.
   - **Mechanism**: A row of cells is initialized randomly. Each cell's state in the next row is determined by its current state and the states of its immediate neighbors, according to a specific rule set (e.g., Rule 30, Rule 110). This process iteratively generates new rows, creating complex patterns that range from simple to highly chaotic, depending on the rule used.

4. **Langton's Ant**:
   - **Description**: A simple Turing machine that moves on a grid of black and white cells, with its movement rules determined by the color of the cell it encounters.
   - **Mechanism**: The ant follows a set of rules: if it encounters a white cell, it turns right, flips the color of the cell to black, and moves forward; if it encounters a black cell, it turns left, flips the color to white, and moves forward. Despite the simplicity, the system exhibits complex behavior, leading to the formation of highways and chaotic regions. Over time, the ant's path can generate intricate and unpredictable patterns.

5. **Voronoi Diagram**:
   - **Description**: This method divides space into regions based on the distance to a set of seed points, where each region contains all points closer to one seed point than to any other.
   - **Mechanism**: Random points are placed on the grid, and the Voronoi diagram is computed by determining the nearest seed point for each grid cell. The edges between different regions are treated as walls, resulting in a maze with polygonal cells. The boundaries between the cells are then refined to form passages, often creating a natural, organic feel to the maze structure.

6. **Fractal Division**:
   - **Description**: A recursive subdivision method that divides the grid into smaller regions by introducing walls along the division lines.
   - **Mechanism**: The algorithm begins by splitting the grid with a wall either horizontally or vertically and then adds an opening in the wall. The process repeats recursively on the resulting subregions. This method, also known as the recursive division algorithm, can produce highly symmetrical and self-similar patterns, where the layout at smaller scales resembles the overall structure.

7. **Wave Function Collapse**:
   - **Description**: Inspired by the concept of quantum mechanics, this method uses a constraint-based approach to determine the state of each cell based on its neighbors.
   - **Mechanism**: The algorithm starts with an undecided grid where each cell can potentially take on multiple states. It then collapses each cell's possibilities based on constraints from neighboring cells, ensuring that the pattern remains consistent and non-contradictory. This method produces highly detailed and aesthetically pleasing mazes, where the structure is consistent with the predefined rules and patterns.

8. **Growing Tree**:
   - **Description**: A procedural method for creating mazes by expanding paths from a starting point, using a selection strategy to decide which frontier cell to grow from.
   - **Mechanism**: The algorithm begins with a single cell and iteratively adds neighboring cells to the maze. The selection strategy can vary (e.g., random, last-in-first-out, first-in-first-out), affecting the overall structure. The growing tree method is flexible and can generate mazes with a variety of appearances, from long corridors to densely packed networks.

9. **Terrain-Based**:
   - **Description**: This approach uses Perlin noise to generate a terrain-like heightmap, which is then converted into a maze by thresholding.
   - **Mechanism**: Perlin noise, a type of gradient noise, is used to create a smooth and continuous terrain heightmap. The grid is then divided into passable and impassable regions based on a threshold value. This method produces mazes that resemble natural landscapes with hills and valleys, offering a different challenge with natural-looking obstacles.

10. **Musicalized**:
    - **Description**: Inspired by musical compositions, this method generates mazes by interpreting harmonic functions and waves.
    - **Mechanism**: The algorithm generates a grid where the value at each cell is determined by the sum of multiple sine waves with different frequencies and amplitudes. The resulting wave patterns are then thresholded to create walls and paths, resembling rhythmic and wave-like structures. This method provides a unique aesthetic, mirroring the periodic nature of music.

11. **Quantum-Inspired**:
    - **Description**: Mimics quantum interference patterns by superimposing wave functions, creating complex interference patterns.
    - **Mechanism**: The algorithm uses a combination of wave functions to create a probability density field. By thresholding this field, the maze walls are determined. The resulting patterns are intricate and delicate, often resembling the complex interference patterns seen in quantum physics experiments. This method offers visually stunning mazes with a high degree of symmetry and complexity.

12. **Artistic**:
    - **Description**: Utilizes artistic techniques such as brush strokes and splatter effects to create abstract maze patterns.
    - **Mechanism**: The algorithm randomly places brush strokes and splatters on a canvas, with each stroke affecting multiple cells on the grid. The placement and orientation of strokes are randomized, creating unique and abstract patterns. This artistic approach results in mazes that mimic various art styles, offering a visually distinct experience.

13. **Cellular Automaton**:
    - **Description**: Uses custom rules to evolve a grid of cells, with each cell's state influenced by its neighbors.
    - **Mechanism**: The grid is initialized with random states. A set of rules determines the next state of each cell based on the states of its neighbors. This process is iterated multiple times, with the specific rules and number of iterations influencing the final pattern. The method can generate a wide range of structures, from highly ordered to chaotic, depending on the chosen ruleset.

14. **Fourier-Based**:
    - **Description**: Applies the Fourier transform to a noise field, selectively filtering frequencies to create smooth patterns.
    - **Mechanism**: The algorithm begins with a random noise field and transforms it into the frequency domain using the Fourier transform. Certain frequency components are then filtered out, and the inverse transform is applied to obtain the spatial domain pattern. The result is a maze with smooth, flowing structures, influenced by the selected frequencies and their combinations.

15. **Reaction-Diffusion**:
    - **Description**: Simulates chemical reaction and diffusion processes to create organic and biomorphic patterns.
    - **Mechanism**: The algorithm models the interaction between two chemical substances that spread out and react with each other. The concentration of these substances evolves over time according to reaction-diffusion equations. The resulting patterns are thresholded to form the maze structure. This method creates mazes with natural, fluid-like structures, similar to those seen in biological organisms and chemical reactions.

Each method in this collection offers a distinct visual and structural style, making it possible to explore a wide range of maze characteristics and challenges. These mazes are suitable for testing various pathfinding algorithms and for generating visually compelling visualizations.

### Maze Validation and Adjustment Techniques

In maze generation, ensuring that the resulting structures are not only visually appealing but also functionally navigable is critical. Various techniques and methods are employed to validate the generated mazes and modify them if they don't meet specific criteria, such as solvability, complexity, or connectivity. This section details the philosophy, theory, and practical implementations behind these techniques, with a focus on ensuring high-quality maze structures.

#### Overview

The approach to maze validation and adjustment involves a multi-step process:

1. **Validation**: After generating a maze, we assess it against predefined criteria such as connectivity, solvability, and structural diversity.
2. **Modification**: If the maze fails to meet these criteria, specific functions are employed to adjust the structure, such as adding or removing walls, creating pathways, or ensuring connectivity between regions.
3. **Final Verification**: The modified maze is re-evaluated to confirm that it now meets all the desired criteria.

This process ensures that each maze not only provides a challenging and engaging environment but also maintains a balance between complexity and solvability.

#### Detailed Function Explanations

1. **smart_hole_puncher**
   - **Purpose**: To ensure that a generated maze has a path from the start to the goal by strategically removing walls.
   - **Mechanism**: The function iteratively selects wall cells and removes them, prioritizing areas where the path might be blocked. It stops once a viable path is found, minimizing changes to the maze's overall structure. This method is particularly useful for complex mazes that may have isolated regions.

2. **ensure_connectivity**
   - **Purpose**: To guarantee that all open regions in a maze are connected, preventing isolated areas.
   - **Mechanism**: This function uses pathfinding algorithms to verify that a continuous path exists between important points (e.g., start and goal). If disconnected regions are found, the function identifies the shortest path between these regions and creates openings to link them, ensuring the maze is fully navigable.

3. **add_walls**
   - **Purpose**: To increase the complexity of a maze by adding walls, which can create new challenges and alter the maze's navigability.
   - **Mechanism**: Additional walls are placed in the maze in a controlled manner to achieve a target wall density. This function randomly selects open cells to convert into walls, balancing between adding challenge and maintaining solvability.

4. **remove_walls**
   - **Purpose**: To simplify a maze by removing walls, making it less dense and more navigable.
   - **Mechanism**: The function selects walls for removal based on the need to decrease wall density to a target percentage. It ensures that the removals do not oversimplify the maze, maintaining a level of challenge and complexity.

5. **add_room_separators**
   - **Purpose**: To divide large open spaces into smaller, distinct areas, thereby adding structure and complexity.
   - **Mechanism**: The function introduces separators or walls within large open areas of the maze. These separators create distinct rooms or sections, which can then be connected or further modified. This technique prevents overly large open areas that can make the maze less challenging.

6. **break_up_large_room**
   - **Purpose**: To prevent excessively large open spaces that could simplify navigation and reduce the challenge.
   - **Mechanism**: The function identifies large rooms in the maze and introduces additional walls to break them into smaller sections. This process involves a careful analysis of the room sizes and a controlled introduction of walls to maintain the balance between openness and complexity.

7. **break_up_large_areas**
   - **Purpose**: Similar to breaking up large rooms, this function targets large contiguous open areas in the maze, ensuring they are partitioned for increased complexity.
   - **Mechanism**: The function identifies large connected areas and introduces walls to create smaller, manageable sections. This helps in preventing navigational ease due to large uninterrupted spaces and ensures a more challenging experience.

8. **create_simple_maze**
   - **Purpose**: To generate a basic structure or fill in small areas within a larger maze.
   - **Mechanism**: This function uses simple algorithms, such as recursive division or random path generation, to create a basic maze structure. It is often used in conjunction with other techniques to fill specific areas or as a foundation that can be modified further.

9. **connect_areas**
   - **Purpose**: To ensure that all regions of a maze are accessible and interconnected.
   - **Mechanism**: The function uses a combination of pathfinding and wall removal to connect distinct regions or areas within a maze. This ensures that no area is isolated, facilitating complete navigation from any starting point.

10. **connect_disconnected_areas**
    - **Purpose**: Specifically focuses on connecting areas that are entirely isolated from the rest of the maze.
    - **Mechanism**: This function identifies completely disconnected regions and creates paths to integrate them into the main maze. It uses algorithms like breadth-first search (BFS) to find the shortest paths for connection, ensuring efficiency and minimal structural change.

11. **bresenham_line**
    - **Purpose**: To draw straight lines on a grid, typically used for creating direct connections or walls.
    - **Mechanism**: The Bresenham's line algorithm is employed to draw straight lines between two points on a grid, ensuring the line is as continuous and close to a true line as possible. This is useful for creating corridors or walls that follow a straight path.

12. **validate_and_adjust_maze**
    - **Purpose**: To perform a comprehensive check of the maze's structural integrity and navigability, followed by necessary adjustments.
    - **Mechanism**: This function validates the maze against criteria such as solvability, wall density, and connectivity. Based on the assessment, it applies various adjustments (like wall addition/removal, area connection) to ensure the maze meets all necessary conditions. It serves as the final quality check before the maze is considered complete.

13. **generate_and_validate_maze**
    - **Purpose**: To generate a maze using one of the specified algorithms and ensure it meets all criteria for quality and functionality.
    - **Mechanism**: This function integrates the entire process of maze generation, validation, and adjustment. It starts with generating a maze, runs validation checks, and applies modifications as needed. If the maze does not meet the criteria, the function can regenerate or further adjust it until all requirements are satisfied.

### Approach and Philosophy

The philosophy behind these validation and modification techniques is to balance between challenging maze complexity and ensuring solvability. The ultimate goal is to create mazes that are not only aesthetically diverse but also functionally engaging for pathfinding. By using a combination of algorithms and heuristic techniques, the system ensures each maze offers a unique experience, with carefully controlled difficulty levels and a guarantee of a solution.

This comprehensive approach allows for the creation of mazes that are structurally sound and enjoyable to navigate, supporting various applications from entertainment (like games) to practical scenarios (like robotics and AI pathfinding training). The emphasis is on adaptability, ensuring the framework can handle a wide range of maze styles and complexities.

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

- **Animation Playback**: The frames can be compiled into an animation using `FuncAnimation` from `matplotlib.animation`. The playback speed can be adjusted by setting the frames per second (FPS), allowing for slower or faster visualization of the pathfinding process. The animation provides a smooth and continuous representation of the algorithm's operation, from initial exploration to final pathfinding.

- **Output Formats**: The frames can be saved individually as images or compiled into a video. Each frame is saved as an individual image in the specified `frame_format` (e.g., PNG, JPG). This option is useful for creating high-quality image sequences or for detailed post-processing of individual frames. Alternatively, if `save_as_frames_only` is set to `False`, the frames are compiled into an animation in formats such as MP4. For MP4 exports, the `FFMpegWriter` is used, allowing for fine-tuned control over encoding parameters, such as bitrate and codec settings. This ensures high-quality video output suitable for presentations or further analysis.

- **Resource Management**: To manage disk space and avoid clutter, the code includes functionality to delete small or temporary files after the animation or frame sequence is saved. This helps maintain a clean working directory and ensures that only the most relevant files are retained. This feature is particularly useful when saving individual frames, as it can help prevent the accumulation of numerous image files.

### Assembling Frames into an MP4 File Using FFmpeg

If you have saved the frames as individual image files and wish to manually assemble them into an MP4 video, you can use FFmpeg, a powerful multimedia processing tool. Below are the steps and recommended settings to create a high-quality MP4 file from a sequence of frames.

#### Prerequisites

Ensure FFmpeg is installed on your system. You can download it from the [official FFmpeg website](https://ffmpeg.org/download.html) or install it via a package manager. Alternatively, you can download a pre-compiled binary from the most recent version [here](https://johnvansickle.com/ffmpeg/). Additionally, you may need the `bc` command for calculations, which can be installed using:

```bash
sudo apt install bc
```

#### Command Example

Assuming your frames are named sequentially (e.g., `frame_0001.png`, `frame_0002.png`, etc.) and stored in the current directory, you can use the following command to generate a 30-second video file using x265:

```bash
cd /home/ubuntu/visual_astar_python/maze_animations/animation_20240805_114757/ # Change to the directory containing the frames-- this is just an example
ffmpeg -framerate $(echo "($(find . -maxdepth 1 -type f -name 'frame_*.png' | wc -l) + 30 - 1) / 30" | bc) -i frame_%04d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=3840:2160" -c:v libx265 -preset slow -crf 28 -pix_fmt yuv420p -x265-params "pools=16:bframes=8:ref=4:no-open-gop=1:me=star:rd=4:aq-mode=3:aq-strength=1.0" -movflags +faststart output.mp4
```

For encoding using x264, use:

```bash
ffmpeg -framerate $(echo "($(find . -maxdepth 1 -type f -name 'frame_*.png' | wc -l) + 30 - 1) / 30" | bc) -i frame_%04d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -crf 18 -pix_fmt yuv420p -threads 16 -movflags +faststart output_x264.mp4
```

#### Explanation of Options

- **`-framerate $(...)`**: Calculates the frame rate based on the number of images and desired video duration (30 seconds in this example). This ensures that the video plays for the correct duration regardless of the number of frames.
- **`-i frame_%04d.png`**: Specifies the input file pattern. `%04d` indicates that the input files are sequentially numbered with four digits (e.g., `frame_0001.png`, `frame_0002.png`).
- **`-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=3840:2160"`**: The `pad` filter ensures the video dimensions are even, which is required for many codecs. The `scale` filter resizes the video to 4K resolution (3840x2160). These filters ensure the output video has compatible dimensions and resolution.
- **`-c:v libx265`**: Specifies the use of the x265 codec for encoding, which provides efficient compression. The x264 variant uses `-c:v libx264` for compatibility and high-quality output.
- **`-preset slow`**: Sets the encoding preset, balancing compression efficiency and encoding time. `slow` is a good compromise for higher compression at a slower speed.
- **`-crf 28`** (for x265) and **`-crf 18`** (for x264): Controls the Constant Rate Factor, affecting the quality and file size. Lower values yield higher quality at the cost of larger file sizes. `crf 28` is suitable for x265, while `crf 18` provides nearly lossless quality for x264.
- **`-pix_fmt yuv420p`**: Sets the pixel format to YUV 4:2:0, ensuring compatibility with most media players and devices.
- **`-x265-params "pools=16:bframes=8:ref=4:no-open-gop=1:me=star:rd=4:aq-mode=3:aq-strength=1.0"`**: Specifies advanced x265 settings to fine-tune the encoding process. These parameters set the number of threads (`pools`), number of B-frames, reference frames, and other encoding settings for optimal quality and compression.
- **`-threads 16`** (for x264): Limits the number of threads used for encoding to 16, balancing performance and resource usage.
- **`-movflags +faststart`**: Enables the `faststart` option, which moves the metadata to the beginning of the file, allowing the video to start playing before it is fully downloaded. This is useful for streaming scenarios.

These commands and explanations should help you efficiently create high-quality MP4 videos from a sequence of frames using FFmpeg.

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

The code is tested with Python 3.12. If you want to use that version without messing with your system Python version, then on Ubuntu you can install and use PyEnv like so:

```bash
if ! command -v pyenv &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git

    git clone https://github.com/pyenv/pyenv.git ~/.pyenv
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
    echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
    source ~/.zshrc
fi
cd ~/.pyenv && git pull && cd -
pyenv install 3.12
cd visual_astar_python
pyenv local 3.12
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

You can see what it looks like running here:

[![asciicast](https://asciinema.org/a/BwnzShGe1Py5QZJ2IXru7xGMv.svg)](https://asciinema.org/a/BwnzShGe1Py5QZJ2IXru7xGMv)

### Customization

- **Maze Generation Approach**: Specify the desired maze generation approach (e.g., `dla`, `wave_function_collapse`) to customize the type of maze generated.
- **Grid Size**: Set the `GRID_SIZE` parameter to adjust the size of the maze.
- **Visualization Settings**: Modify color schemes and animation settings to suit your preferences.

## Parameter Configuration

This project includes several parameters that users can configure to customize the output of maze generation and pathfinding visualization. Key parameters include:

- **num_animations**: The number of separate animations to generate, each featuring different mazes and pathfinding scenarios.
- **GRID_SIZE**: The size of the maze grid, determining the number of cells along one dimension. Higher values create more detailed and complex mazes.
- **num_problems**: The number of mazes to display side by side in each animation, allowing for comparison of different generation methods or pathfinding strategies.
- **DPI**: The dots per inch for the animation, affecting image resolution and quality. Higher DPI values yield sharper images.
- **FPS**: Frames per second for animation playback. Higher values create smoother animations but may require more resources.
- **save_as_frames_only**: A boolean parameter indicating whether to save each frame as an individual image. Set `True` to save frames, `False` to compile them into a video.
- **frame_format**: The format for saving frames when `save_as_frames_only` is `True`. Common formats include 'png' and 'jpg'.
- **dark_mode**: Enables a dark theme for visualizations.
- **override_maze_approach**: Forces the use of a specific maze generation approach for consistency across all animations.

These parameters provide users with extensive control over the behavior and appearance of the generated mazes and visualizations, allowing for fine-tuning according to specific requirements or preferences.

## A\* Algorithm: Theoretical Overview and Advanced Implementation

### The Big Ideas Behind A\*

The A\* algorithm is a sophisticated method for finding the shortest path in various environments, whether it's navigating a complex maze or plotting a course through a dynamically changing landscape. Unlike simpler algorithms, A\* intelligently evaluates paths by considering both the journey already taken and the estimated distance to the goal. This dual approach makes A\* not only a pathfinder but a path optimizer, ensuring the selected path is the most efficient.

Imagine you're in a vast maze; A\* doesn't just explore paths blindly. It uses a strategic approach akin to a seasoned traveler who checks their progress and considers the remaining distance to the destination at every decision point. This ability to foresee and plan makes A\* particularly adept at avoiding dead ends and minimizing travel time.

### Key Components of A\*

1. **The Journey So Far (g-cost)**: This aspect involves calculating the exact cost from the start point to the current position. It's like keeping track of the miles traveled during a road trip. By accumulating these costs, A\* can compare different paths to the same point, ensuring it chooses the most efficient one.

2. **The Journey Ahead (h-cost)**: Known as the heuristic estimate, this component predicts the cost from the current position to the goal. It's a calculated guess based on the nature of the environment. For example, in a grid, this might be the Euclidean distance (straight-line distance), which provides a quick and reasonably accurate estimate of the remaining journey.

The sum of these two values (g-cost + h-cost) forms the **f-cost**, which A\* uses to prioritize paths. This combination ensures that A\* not only seeks to minimize the total travel cost but also maintains a focus on progressing towards the goal.

### Why A\* Excels Compared to Other Algorithms

A\* stands out for its efficiency and effectiveness, particularly in comparison to simpler algorithms:

- **Depth-First Search (DFS)** explores as deep as possible along one branch before backtracking. While it may find a path, it often does so inefficiently, sometimes missing the shortest path entirely due to its lack of goal-awareness and tendency to get trapped in deep branches.

- **Breadth-First Search (BFS)** methodically explores all nodes at the present depth before moving on to the next. While BFS guarantees finding the shortest path in an unweighted graph, it is computationally expensive and memory-intensive, especially in large graphs, as it explores all nodes without any consideration of the goal's location.

- **Dijkstra's Algorithm** is a precursor to A\* that calculates the shortest path by considering the total cost from the start node to each node. However, it does not incorporate a heuristic, treating all paths equally regardless of their direction relative to the goal. This can lead to unnecessary exploration and inefficiencies, particularly in large graphs with varied edge costs.

A\* merges the strengths of Dijkstra's thorough cost analysis with an informed heuristic approach, directing its search towards the goal and avoiding unnecessary paths. This combination allows it to find the shortest path efficiently, making it suitable for a wide range of practical applications.

### Applications and Advantages

A\* is widely used in various domains due to its reliability and efficiency:

- **Video Games**: A\* is the backbone of many AI pathfinding systems, guiding characters through complex virtual worlds with precision. Its ability to navigate around obstacles and efficiently reach objectives makes it ideal for real-time strategy games and role-playing games.

- **Robotics**: In robotics, A\* helps autonomous robots navigate through environments, such as factory floors or outdoor terrains. The algorithm enables robots to avoid obstacles, plan efficient routes, and respond to dynamic changes in their surroundings.

- **Navigation Systems**: A\* is used in GPS navigation systems to find the quickest route between two points, considering factors like road distances and traffic conditions.

- **AI and Machine Learning**: A\* is used in AI for problem-solving, such as solving puzzles or planning tasks. Its ability to incorporate different heuristics allows it to be adapted to various types of problems.

### Advanced Implementation Features

This project's implementation of A\* goes beyond standard features, incorporating advanced techniques to enhance performance and versatility:

1. **Smart Organization**: The algorithm uses a priority queue to manage paths, ensuring that the most promising paths are explored first. This efficient data structure reduces the time spent evaluating less optimal paths.

2. **Precision Handling**: The implementation supports both integer and floating-point coordinates, making it adaptable to different scenarios, from simple grid maps to detailed real-world environments.

3. **Adaptive Heuristics**: It allows for various heuristic functions, such as Manhattan, Euclidean, or Octile distances, which can be tailored to the specifics of the problem space, optimizing the search process.

4. **Complex Terrain Navigation**: The implementation can handle diverse movements, including diagonal and custom paths, enhancing its capability to navigate through varied terrains.

5. **Efficient Path Reconstruction**: Upon reaching the goal, the implementation efficiently reconstructs the path, ensuring minimal computational overhead in finalizing the route.

6. **Robust Error Handling**: The algorithm gracefully manages exceptional situations, such as encountering impassable regions or unsolvable configurations, providing clear feedback to users.

7. **Optimized Data Structures**: The use of bit fields for coordinate encoding enhances memory efficiency and processing speed, crucial for handling large-scale environments or high-resolution grids.

This detailed implementation not only highlights A\*'s theoretical strengths but also showcases practical enhancements that make it a powerful tool for modern pathfinding and navigation challenges. Whether in gaming, robotics, or complex AI systems, A\* remains a go-to algorithm for its unmatched balance of efficiency and accuracy.

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
- SciPy
- Scikit-Image
- Noise
- Pillow
- TQDM
- Numba (for JIT compilation)
- FFmpeg (for video encoding)
- Requests (for downloading custom fonts)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
