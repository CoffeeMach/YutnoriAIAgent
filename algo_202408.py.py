import yut.engine
import yut.rule
import random
import numpy as np
import scipy.stats
import os
import json
from collections import defaultdict

def analyze_game_data(log_file):
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"The file '{log_file}' does not exist.")

    with open(log_file, 'r') as file:
        data = json.load(file)
        print(data)  # Inspect the loaded data


    q_table = defaultdict(lambda: defaultdict(float))
    action_counts = defaultdict(lambda: defaultdict(int))

    for entry in data:
        print(entry)  # Check the entry structure
        state_key = tuple(entry["state_key"]) if isinstance(entry["state_key"], list) else entry["state_key"]
        action = tuple(entry["action"]) if isinstance(entry["action"], list) else entry["action"]
        reward = entry["reward"]


        q_table[state_key][action] += reward
        action_counts[state_key][action] += 1

    for state_key in q_table:
        for action in q_table[state_key]:
            if action_counts[state_key][action] > 0:  # Prevent division by zero
                q_table[state_key][action] /= action_counts[state_key][action]

    return q_table


def save_prefilled_q_table(q_table, output_file):
    data_to_save = []
    for state_key, actions in q_table.items():
        for action_key, value in actions.items():
            data_to_save.append({
                "state_key": list(state_key),  # Convert tuple to list for saving
                "action": list(action_key),     # Convert tuple to list for saving
                "reward": value
            })

    with open(output_file, 'w') as file:
        json.dump(data_to_save, file, indent=4)


def load_q_table_from_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    with open(file_path, 'r') as file:
        raw_data = json.load(file)

    print("Raw data loaded:", raw_data)  # Debug: check what was loaded

    # Ensure we load the Q-table correctly
    q_table = {}
    
    if isinstance(raw_data, list):  # Check if the loaded data is a list
        for entry in raw_data:
            state_key = tuple(entry["state_key"]) if isinstance(entry["state_key"], list) else entry["state_key"]
            action_key = tuple(entry["action"]) if isinstance(entry["action"], list) else entry["action"]
            reward = entry["reward"]

            if state_key not in q_table:
                q_table[state_key] = {}
            q_table[state_key][action_key] = reward  # Directly assign the reward for this action

    else:
        raise ValueError("Loaded data is not in the expected list format.")

    return q_table




if __name__ == "__main__":
 # Path to the logged data file (replace with your actual file path)
    log_file_path = "C:\\Users\\Monster\\Desktop\\CoE202_final (2)\\final\\game_data.json"

    if not os.path.exists(log_file_path):
        print(f"Error: The file '{log_file_path}' does not exist. Please create or specify the correct file path.")
    else:
        try:
            q_table = analyze_game_data(log_file_path)
            print("Q-table initialized with meaningful values:")
            print(q_table)
        except Exception as e:
            print(f"An error occurred: {e}")



class MyAlgo(yut.engine.Player):
    def __init__(self):
        self.q_table = {}
        self.action_counts = {}  # Dictionary to track counts of actions
        self.alpha = 0.5  # Learning rate
        self.gamma = 0.85  # Discount factor
        self.epsilon = 0.4  # Exploration rate
        self.min_epsilon = 0.1  # Minimum exploration rate
        self.epsilon_decay = 0.98  # Decay factor
        self.exploration_bonus = 0.1  # Exploration bonus
        self.log_file_path = "C:\\Users\\Monster\\Desktop\\CoE202_final (2)\\final\\game_data.json"  # Path to your JSON file


    def name(self):
        return "TeamYutopiaAlgorithm"

    def reset(self, random_state):
        self.random_state = random_state
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


    def action(self, state):
        global available_yutscores
        turn, my_positions, enemy_positions, available_yutscores = state
        scores = []
        state_key = self.state_to_key(my_positions, enemy_positions, available_yutscores)
        

        if np.random.rand() < self.epsilon:
            # Exploration
            for mi, mp in enumerate(my_positions):
                if mp == yut.rule.FINISHED:
                    continue
                for ys in available_yutscores:
                    for shortcut in [True, False]:
                        legal_move, next_my_positions, next_enemy_positions, num_mals_caught = yut.rule.make_move(my_positions, enemy_positions, mi, ys, shortcut)
                        if legal_move:
                            score = self.evaluate_move(next_my_positions, next_enemy_positions, num_mals_caught > 0)
                            scores.append((score, mi, ys, shortcut))
        else:
            # Exploitation
            for mi, mp in enumerate(my_positions):
                if mp == yut.rule.FINISHED:
                    continue
                for ys in available_yutscores:
                    for shortcut in [True, False]:
                        legal_move, next_my_positions, next_enemy_positions, num_mals_caught = yut.rule.make_move(my_positions, enemy_positions, mi, ys, shortcut)
                        if legal_move:
                            score = self.q_table.get(state_key, {}).get((mi, ys, shortcut), 0.0)
                            scores.append((score, mi, ys, shortcut))
                            

        if scores:
            scores.sort(reverse=True)
            return scores[0][1], scores[0][2], scores[0][3], ""
        else:
            return 0, available_yutscores[0], False, ""

    def evaluate_move(self, my_positions, enemy_positions, throw_again):
        global available_yutscores
        my_duplicates = [sum(np == p for np in my_positions) for p in my_positions]
        enemy_duplicates = [sum(np == p for np in enemy_positions) for p in enemy_positions]
        multipliers = [1, 1, 0.6, 0.6, 0.7]

        score = -sum(distance_to_goal[p] * (multipliers[np] if p != 0 else 1) for p, np in zip(my_positions, my_duplicates)) \
                + sum(distance_to_goal[p] * (multipliers[np] if p != 0 else 1) for p, np in zip(enemy_positions, enemy_duplicates)) \
                + (1 if throw_again else 0)

        for ep in enemy_positions:
            if ep < yut.rule.FINISHED:  # Only consider un-finished enemies
                distance_to_enemy = distance_to_goal[ep]
                if distance_to_enemy < min(distance_to_goal[mp] for mp in my_positions):  # Enemy is ahead
                    score += 5 - distance_to_enemy  # Reward for being closer to the enemy in front

        # Penalize for enemies behind (higher distance to goal)
        for ep in enemy_positions:
            if ep < yut.rule.FINISHED:  # Only consider un-finished enemies
                distance_to_enemy = distance_to_goal[ep]
                if distance_to_enemy > min(distance_to_goal[mp] for mp in my_positions):  # Enemy is behind
                    score -= max(2, distance_to_enemy - 4)  # Penalize based on how far behind they are

        return score
    
    def log_data(self, state_key, action, reward):
        log_entry = {
            "state_key": list(state_key),  # Convert tuple to list for JSON
            "action": list(action),          # Convert tuple to list for JSON
            "reward": reward
        }

        # Append new log entry to the existing log file
        if os.path.exists(self.log_file_path):
            with open(self.log_file_path, 'r+') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:  # Handle case where the file is empty
                    data = []
                data.append(log_entry)
                file.seek(0)  # Move the cursor to the beginning of the file
                json.dump(data, file, indent=4)  # Write updated data back to file
        else:
            # Create the file if it does not exist and write the first entry
            with open(self.log_file_path, 'w') as file:
                json.dump([log_entry], file, indent=4)

    def on_my_action(self, state, my_action, result):
        turn, my_positions, enemy_positions, available_yutscores = state
        state_key = self.state_to_key(my_positions, enemy_positions, available_yutscores)
        next_state_key = self.state_to_key(result[1], result[2], available_yutscores)

        reward = self.calculate_reward(result)
        self.update_q_table(state_key, my_action, reward, next_state_key)

        # Log the data after updating Q-table
        self.log_data(state_key, my_action, reward)

        # Adjust epsilon based on recent performance
        if reward > 0:
            self.epsilon = max(self.min_epsilon, self.epsilon * 0.99)
        else:
            self.epsilon = min(1.0, self.epsilon + 0.01)

     
    def calculate_reward(self, result):
        legal_move, _, _, num_mals_caught = result
        if not legal_move:
            return -10
        reward = 10 * num_mals_caught
        return reward + (1 if num_mals_caught > 0 else 0)

    def convert_numpy_types(self, data):
        """Recursively convert NumPy types to standard Python types."""
        if isinstance(data, dict):
            return {key: self.convert_numpy_types(value) for key, value in data.items()}  # Call with self
        elif isinstance(data, list):
            return [self.convert_numpy_types(item) for item in data]  # Call with self
        elif isinstance(data, (np.integer, np.floating)):
            return data.item()  # Convert NumPy int/float to standard Python int/float
        elif isinstance(data, np.ndarray):
            return data.tolist()  # Convert NumPy array to a standard Python list
        else:
            return data  # Return the data as is if it's a standard type

    def update_q_table(self, state_key, action, reward, next_state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
            self.action_counts[state_key] = {}

        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0
            self.action_counts[state_key][action] = 0

        self.action_counts[state_key][action] += 1
        max_next_q = max(self.q_table.get(next_state_key, {}).values(), default=0.0)
        exploration_bonus = self.exploration_bonus / (1 + self.action_counts[state_key][action])

        # Update Q-value
        self.q_table[state_key][action] += self.alpha * (reward + self.gamma * max_next_q - self.q_table[state_key][action] + exploration_bonus)

        # Prepare log data
        log_data = {
            "state_key": [[int(k) for k in sub_key] for sub_key in state_key],  # Convert each element of the nested tuples to int
            "action": [int(a) if isinstance(a, (int, np.integer)) else a for a in action],  # Convert each action element to int
            "reward": float(reward)  # Ensure reward is a float
        }

        # Convert any NumPy types to standard Python types
        log_data = self.convert_numpy_types(log_data)  # Call the conversion function

        self.log_game_data(log_data)

    def log_game_data(self, log_data):
        log_file = "C:\\Users\\Monster\\Desktop\\CoE202_final (2)\\final\\game_data.json"
        
        # Load existing data
        if os.path.exists(log_file):
            with open(log_file, 'r') as file:
                try:
                    existing_data = json.load(file)
                except json.JSONDecodeError:
                    existing_data = []  # Handle empty or corrupted JSON file
        else:
            existing_data = []

        # Append new log entry
        existing_data.append(log_data)

        # Save updated data back to the file
        with open(log_file, 'w') as file:
            json.dump(existing_data, file, indent=4)



    def state_to_key(self, my_positions, enemy_positions, available_yutscores):
        return tuple(my_positions), tuple(enemy_positions), tuple(available_yutscores)


distance_to_goal = np.zeros( yut.rule.FINISHED+1 )
outcomes, probs = yut.rule.enumerate_all_cast_outcomes(depth=5)

for _ in range(10):
	for s in range( yut.rule.FINISHED-1, -1, -1):
		weighted_sum = 0.0
		for outcome, prob in zip( outcomes, probs ):
			pos = s
			for ys in outcome:
				pos = yut.rule.next_position( pos, ys, True )
			weighted_sum += ( 1 + distance_to_goal[pos] ) * prob
		distance_to_goal[s] = weighted_sum

def evaluate_score( my_positions, enemy_positions, throw_again ):
	my_duplicates = [ sum(np == p for np in my_positions) for p in my_positions ]
	enemy_duplicates = [ sum(np == p for np in enemy_positions) for p in enemy_positions ]
	multipliers = [ 1, 1, 0.7, 0.4, 0.3 ]

	return - sum( distance_to_goal[p] * (multipliers[np] if p != 0 else 1) for p,np in zip(my_positions,my_duplicates) ) \
			+ sum( distance_to_goal[p] * (multipliers[np] if p != 0 else 1) for p,np in zip(enemy_positions,enemy_duplicates) ) \
			+ ( +1 if throw_again else 0 )


class ExamplePlayer(yut.engine.Player):
	def name(self):
		return "Example"

	def action(self, state):
		turn, my_positions, enemy_positions, available_yutscores = state

		scores = []
		for mi, mp in enumerate(my_positions):
			if mp == yut.rule.FINISHED:
				continue
			for ys in available_yutscores:
				for shortcut in [True, False]:
					legal_move, next_my_positions, next_enemy_positions, num_mals_caught = yut.rule.make_move( my_positions, enemy_positions, mi, ys, shortcut )
					if legal_move:
						scores.append( (evaluate_score(next_my_positions, next_enemy_positions, num_mals_caught>0), mi, ys, shortcut ) )
		scores.sort(reverse=True)

		return scores[0][1], scores[0][2], scores[0][3], ""
     
def main(q_table_file):
    # Load the Q-table
    try:
        q_table = load_q_table_from_file(q_table_file)
        my_player = MyAlgo()
        my_player.q_table = q_table

    except Exception as e:
        print(f"Error loading Q-table: {e}")

# Main game loop
if __name__ == "__main__":
    engine = yut.engine.GameEngine()
    player1 = MyAlgo()  # Your custom algorithm
    player2 = ExamplePlayer()  # Competitor

    # Track wins for both players
    player1_wins = 0
    player2_wins = 0

    # Simulate 100 games
    for seed in range(100):
        random_state = np.random.RandomState(seed)
        player1.reset(random_state)
        player2.reset(random_state)

        winner = engine.play(player1, player2, seed=seed)

        if winner == 0:
            player1_wins += 1
            print(f"Game {seed}: {player1.name()} won!")
        else:
            player2_wins += 1
            print(f"Game {seed}: {player2.name()} won!")

    # Display final results
    print(f"Final Results: {player1.name()} wins: {player1_wins}, {player2.name()} wins: {player2_wins}")
    
