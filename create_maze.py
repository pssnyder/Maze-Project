import numpy as np
import matplotlib.pyplot as plt
import random

def create_prim_maze(width, height, start, end):
    # Initialize the maze with walls
    maze = np.ones((height, width), dtype=int)
    
    # Directions: right, down, left, up (for movement in the maze)
    directions = [(2, 0), (0, 2), (-2, 0), (0, -2)]
    
    def is_valid_move(x, y):
        return 0 <= x < width and 0 <= y < height and maze[y][x] == 1

    def add_frontier(x, y, frontier):
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid_move(nx, ny):
                frontier.append((nx, ny, x, y))

    # Start Prim's algorithm from the start position
    x, y = start
    maze[y][x] = 0
    frontier = []
    add_frontier(x, y, frontier)
    
    while frontier:
        # Choose a random frontier cell
        idx = random.randint(0, len(frontier) - 1)
        fx, fy, px, py = frontier.pop(idx)
        
        if maze[fy][fx] == 1:
            # Carve the passage
            maze[fy][fx] = 0
            maze[(fy + py) // 2][(fx + px) // 2] = 0
            # Add new frontiers
            add_frontier(fx, fy, frontier)
    
    # Ensure the end point is set
    maze[end[1]][end[0]] = 0

    return maze

def main():
    # Prompt the user for maze dimensions
    width = int(input("Enter maze width (odd number): "))
    height = int(input("Enter maze height (odd number): "))

    # Set the start position to (1, 1)
    start = (1, 1)

    # Randomly choose an end position in the bottom 25% of the maze
    end_y = random.randint(height * 3 // 4, height - 2)
    end_x = random.choice([x for x in range(1, width - 1) if x % 2 == 1])
    end = (end_x, end_y)

    # Create the maze
    maze = create_prim_maze(width, height, start, end)

    # Ensure the perimeter is filled with 1s
    maze[0, :] = 1
    maze[:, 0] = 1
    maze[-1, :] = 1
    maze[:, -1] = 1

    # Print the maze array string
    maze_array_string = np.array2string(maze, separator=', ')
    print("Maze array:")
    print(maze_array_string)

    # Generate the title based on inputs
    title = f"{width}x{height} Maze from {start} to {end}"

    # Visualize the maze
    plt.imshow(maze, cmap='binary')
    plt.title(title)

    # Mark the start and end points
    plt.scatter(start[0], start[1], color='green', s=100, label='Start')
    plt.scatter(end[0], end[1], color='red', s=100, label='End')

    # Add legend
    plt.legend(loc='upper right')

    plt.show()

if __name__ == "__main__":
    main()