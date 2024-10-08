# Machine-Learning-Coursework-2
This code was done as part of the coursework for the Machine Learning Module for King's College London, where I achieved 100% for this project. I have only uploaded the code that I made. This script implements a Q-learning agent for the Pacman game, this assignment was taken from the Berkeley AI project. The GameStateFeatures class extracts useful information from the game state, such as Pacman's legal actions, position, food state, and ghost position. The QLearnAgent class initializes with hyperparameters like learning rate (alpha), exploration rate (epsilon), discount factor (gamma), maximum attempts per action, and number of training episodes. It maintains Q-values and visitation counts using dictionaries. The agent computes rewards based on game state transitions, updates Q-values using the Q-learning algorithm, and employs an epsilon-greedy strategy to balance exploration and exploitation. The agent's actions are determined by the best Q-value or a random choice based on the exploration rate. The script also includes methods to handle the end of episodes, update learning parameters, and track the number of training episodes.

-Q-Learning Algorithm (achieved 100% in this coursework)
- Implemented a Q-learning algorithm to optimize Pacman's decision-making process in a classic game scenario.
- Utilized object-oriented programming principles, including inheritance and method overriding, to create modular and reusable
code.
- Incorporated type hinting, decorators, and default dictionaries for enhanced code readability and maintainability.
- Achieved Pacman winning rates of 80%+ in small and medium-sized grid environments, showcasing the effectiveness of the
Q-learning approach.

