import yut.engine
import yut.rule
import numpy as np

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


    def name(self):
        return "ETyut"

    def reset(self, random_state):
        self.random_state = random_state
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Randomize initial Q-table for more variability
        for state_key in self.q_table:
            for action in self.q_table[state_key]:
                self.q_table[state_key][action] += random_state.rand() * 0.01  # Adding a small random perturbation


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

    def on_my_action(self, state, my_action, result):
        turn, my_positions, enemy_positions, available_yutscores = state
        state_key = self.state_to_key(my_positions, enemy_positions, available_yutscores)
        next_state_key = self.state_to_key(result[1], result[2], available_yutscores)

        reward = self.calculate_reward(result)
        self.update_q_table(state_key, my_action, reward, next_state_key)

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

    def update_q_table(self, state_key, action, reward, next_state_key):
        # Ensure the state key exists
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
            self.action_counts[state_key] = {}

        # Initialize the action if not present
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0
            self.action_counts[state_key][action] = 0

        self.action_counts[state_key][action] += 1

        # Compute maximum Q-value for the next state
        max_next_q = max(self.q_table.get(next_state_key, {}).values(), default=0.0)
        
        # Calculate exploration bonus
        exploration_bonus = self.exploration_bonus / (1 + self.action_counts[state_key][action])

        # Update the Q-value using a more robust formula
        self.q_table[state_key][action] += self.alpha * (reward + self.gamma * max_next_q - self.q_table[state_key][action] + exploration_bonus)


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


if __name__ == "__main__":
	p = ExamplePlayer()
	q = MyAlgo()
	engine = yut.engine.GameEngine()
	c = 0
	for s in range(100):
		winner = engine.play( p, q, seed=s )
		if winner == 0:
			print( "Player 1 won!" )
			c += 1
		else:
			print( "Player 2 won!" )
	print( "Player 1 won", c, "times" )
	print( "Player 2 won", 100-c, "times" )



