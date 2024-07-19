import os
import datetime
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

# Configuration Section
# =====================

# Environment settings
TF_CPP_MIN_LOG_LEVEL = '2'  # Suppress TensorFlow logging
TF_ENABLE_ONEDNN_OPTS = '0'  # Disable oneDNN optimization

# Logging settings
LOG_DIR = './logs/'
LOG_FILENAME = 'maze_training_log_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.log'

# Maze settings
start_position = (1, 1)  # Starting position in the maze
end_positions = [(2, 4), (5, 5), (16, 8), (17, 16)]

# Training parameters
NUM_EPISODES = 1000  # Number of training episodes
MAX_STEPS_PER_EPISODE = 200  # Maximum steps per episode
EPOCHS = 100  # Number of epochs for training the neural network
BATCH_SIZE = 32  # Batch size for training
VALIDATION_SPLIT = 0.2  # Fraction of data to use for validation

# Neural network architecture
INPUT_SHAPE = (6,)  # Shape of the input layer
HIDDEN_LAYERS = [
    {'units': 36, 'activation': 'relu'},  # First hidden layer
    {'units': 12, 'activation': 'relu'}   # Second hidden layer
]
OUTPUT_LAYER = {'units': 4, 'activation': 'softmax'}  # Output layer

# Optimizer settings
OPTIMIZER = 'Nadam'  # Optimizer to use
LOSS_FUNCTION = 'categorical_crossentropy'  # Loss function

## Define the mazes
# 5x5 Maze - 1 solution no dead ends - end_position(2,4)
maze5x5 = np.array([
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
])

# 10x10 Maze - 1 solution 1 dead ends - end_position(5,5)
maze10x10 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

# 20x20 Maze - double wide multiple solutions multiple dead ends - end_position(16,8)
maze20x20 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

# 30x30 Maze - triple wide multiple solutions multiple dead ends - end_position(17,16)
maze30x30 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

# Create the maze: 1 represents walls, 0 represents open paths
# Define possible actions: up, down, left, right
possible_actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CPP_MIN_LOG_LEVEL
os.environ['TF_ENABLE_ONEDNN_OPTS'] = TF_ENABLE_ONEDNN_OPTS

# Set up logging
logging.basicConfig(filename=LOG_DIR + LOG_FILENAME, level=logging.DEBUG)

def create_neural_network_model():
    """Creates and compiles the neural network model."""
    model = Sequential()
    model.add(Input(shape=INPUT_SHAPE))
    for layer in HIDDEN_LAYERS:
        model.add(Dense(layer['units'], activation=layer['activation']))
    model.add(Dense(OUTPUT_LAYER['units'], activation=OUTPUT_LAYER['activation']))
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=['accuracy'])
    return model

def one_hot_encode(action_index, num_classes=4):
    encoded = np.zeros(num_classes)
    encoded[action_index] = 1
    return encoded

def format_time(seconds):
    if seconds >= 60:
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes)} minutes, {int(seconds)} seconds"
    else:
        return f"{seconds:.2f} seconds"

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

def generate_training_data(mazes, end_positions, num_episodes=NUM_EPISODES, max_steps_per_episode=MAX_STEPS_PER_EPISODE):
    states = []
    actions = []
    episode_times = []
    start_time = time.time()
    for maze, end_position in zip(mazes, end_positions):
        for i in range(num_episodes // len(mazes)):
            episode_start = time.time()
            current_position = start_position
            steps = 0
            visited_positions = set()
            while current_position != end_position and steps < max_steps_per_episode:
                current_state = get_current_state(maze, current_position)
                action_index = np.random.randint(4)
                action = possible_actions[action_index]
                new_position = tuple(map(sum, zip(current_position, action)))
                if new_position == end_position:
                    reward = 100
                elif maze[new_position] == 1:
                    reward = -50
                else:
                    if new_position in visited_positions:
                        reward = -10  # Penalty for revisiting
                    else:
                        reward = 10  # Reward for new space
                    visited_positions.add(new_position)
                states.append(current_state)
                actions.append(one_hot_encode(action_index))
                if maze[new_position] != 1:
                    current_position = new_position
                if reward == 100:
                    break
                steps += 1
            episode_end = time.time()
            episode_times.append(episode_end - episode_start)
            if i % 100 == 0 and i != 0:
                avg_time_per_episode = np.mean(episode_times)
                elapsed_time = time.time() - start_time
                estimated_total_time = (num_episodes - i) * avg_time_per_episode
                print(f"Episode {i}/{num_episodes}")
                print(f"Average time per episode: {avg_time_per_episode:.2f} seconds")
                print(f"Estimated time remaining: {format_time(estimated_total_time)}")
                print(f"Elapsed time: {format_time(elapsed_time)}")
    plt.plot(range(len(episode_times)), episode_times)
    plt.xlabel('Episode')
    plt.ylabel('Time per Episode (seconds)')
    plt.show()
    return np.array(states), np.array(actions)

if __name__ == "__main__":
    mazes = [maze5x5, maze10x10, maze20x20, maze30x30]
        
    print("Mazes layout:")
    for maze in mazes:
        print(maze)
    print("Start:", start_position)
    print("End positions:", end_positions)
    
    # Create and compile the model
    model = create_neural_network_model()
    print(model.summary())
    
    # Generate training data
    print("Generating training data...")
    training_states, training_actions = generate_training_data(mazes, end_positions)
    print("Training data shape:", training_states.shape, training_actions.shape)
    
    # Train the model
    print("Starting training...")
    log_dir = LOG_DIR + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='10,15')
    history = model.fit(
        training_states, training_actions,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1,
        callbacks=[tensorboard_callback]
    )
    
    model_save_path = os.getcwd() + "/models/maze_solver_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
    model.save(model_save_path)
    print("Training completed.")
    print("Model saved to:" + model_save_path)