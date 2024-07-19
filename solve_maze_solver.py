import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tkinter import Tk, filedialog
import networkx as nx
import pydot
from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot
import matplotlib.animation as animation

# Configuration Section
# =====================

# Maze settings
start_position = (1, 1)  # Starting position in the maze

# Maze definitions
maze5x5 = np.array([
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
])
end_position_5x5 = (2, 4)  # End position for 5x5 maze

maze10x10 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 1, 0, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])
end_position_10x10 = (5, 5)  # End position for 10x10 maze

maze20x20 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ...
])
end_position_20x20 = (16, 8)  # End position for 20x20 maze

maze30x30 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ...
])
end_position_30x30 = (17, 16)  # End position for 30x30 maze

# Define possible actions: up, down, left, right
possible_actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Function to get the current state of the maze
def get_current_state(maze, current_position):
    x, y = current_position
    state = [
        x / maze.shape[0],  # Normalized x position
        y / maze.shape[1],  # Normalized y position
        maze[x-1, y] if x > 0 else 1,  # Is there a wall above?
        maze[x+1, y] if x < maze.shape[0] - 1 else 1,  # Is there a wall below?
        maze[x, y-1] if y > 0 else 1,  # Is there a wall to the left?
        maze[x, y+1] if y < maze.shape[1] - 1 else 1  # Is there a wall to the right?
    ]
    return state

# Function to solve the maze using the trained model
def solve_maze(model, maze, end_position, max_steps=2000):
    current_position = start_position
    path = [current_position]
    steps = 0
    recent_positions = []

    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(-0.5, maze.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, maze.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    ax.imshow(maze, cmap='gray')

    def update(frame):
        nonlocal current_position, steps
        if current_position == end_position or steps >= max_steps:
            ani.event_source.stop()
            return

        current_state = get_current_state(maze, current_position)
        prediction = model.predict(np.array([current_state]), verbose=0)[0]

        for _ in range(4):
            action_index = np.argmax(prediction)
            action = possible_actions[action_index]
            new_position = tuple(map(sum, zip(current_position, action)))
            if new_position not in recent_positions and maze[new_position] != 1:
                break
            prediction[action_index] = -np.inf

        if maze[new_position] != 1:
            current_position = new_position
            path.append(current_position)
            recent_positions.append(current_position)
            if len(recent_positions) > 5:
                recent_positions.pop(0)
            steps += 1

        ax.clear()
        ax.imshow(maze, cmap='gray')
        ax.set_xticks(np.arange(-0.5, maze.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, maze.shape[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        ax.tick_params(which='minor', size=0)
        for pos in path:
            ax.add_patch(plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, fill=True, color='blue', alpha=0.5))
        ax.add_patch(plt.Rectangle((start_position[1] - 0.5, start_position[0] - 0.5), 1, 1, fill=True, color='green', alpha=0.5))
        ax.add_patch(plt.Rectangle((end_position[1] - 0.5, end_position[0] - 0.5), 1, 1, fill=True, color='red', alpha=0.5))
        ax.set_title(f'Step: {steps}')

    ani = FuncAnimation(fig, update, frames=max_steps, repeat=False, interval=200)
    plt.show()

    return path, steps

# Function to print the solution path in the maze
def print_solution(maze, path, end_position):
    solution_maze = maze.copy()
    for x, y in path:
        if (x, y) != start_position and (x, y) != end_position:
            solution_maze[x, y] = 2  # Mark path
    solution_maze[start_position] = 3  # Mark start
    solution_maze[end_position] = 4  # Mark end

    print("\nSolution:")
    for row in solution_maze:
        print(' '.join(['#' if cell == 1 else 'S' if cell == 3 else 'E' if cell == 4 else '.' if cell == 2 else ' ' for cell in row]))

def visualize_model(model):
    dot = model_to_dot(model, show_shapes=True, show_layer_names=True)
    return SVG(dot.create(prog='dot', format='svg'))

def highlight_active_nodes(model, current_state):
    layers = [layer for layer in model.layers if 'dense' in layer.name]
    activations = []

    for layer in layers:
        intermediate_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
        activations.append(intermediate_model.predict(np.array([current_state]))[0])

    return activations

if __name__ == "__main__":
    # Open file dialog to select the model file
    root = Tk()
    root.withdraw()  # Hide the root window
    model_file_path = filedialog.askopenfilename(
        initialdir=os.getcwd() + "/models",
        title="Select model file",
        filetypes=(("H5 files", "*.h5"), ("All files", "*.*"))
    )
    if not model_file_path:
        raise FileNotFoundError("No model file selected.")
    print("Selected model:", model_file_path)

    # Load the selected model
    model = load_model(model_file_path)

    # Select the maze and end position based on the maze size
    maze = maze10x10  # Change this to select different mazes
    end_position = end_position_10x10  # Change this to match the selected maze

    print("Maze layout:")
    print(maze)
    print("Start:", start_position)
    print("End:", end_position)

    # Visualize the model
    print("Visualizing the model...")
    svg = visualize_model(model)
    display(svg)

    # Solve the maze using the trained model
    print("Solving the maze...")
    solution_path, steps = solve_maze(model, maze, end_position)
    print("Solution path:", solution_path)

    # Visualize the solution
    print_solution(maze, solution_path, end_position)
