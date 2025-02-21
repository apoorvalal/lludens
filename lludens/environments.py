import numpy as np


class Game:
    def __init__(self, agent1, agent2, n_rounds):
        self.agent1 = agent1
        self.agent2 = agent2
        self.n_rounds = n_rounds
        self.history = []  # Store the history of the game
        self.total_payoffs = [0, 0]

    def get_payoff(self, move1, move2):
        raise NotImplementedError()

    def get_system_prompt(self):
        raise NotImplementedError()

    def get_valid_moves(self):
        raise NotImplementedError()

    def play_round(self, round_num):
        prompt = (
            "What is your move for this round? Respond with one of: "
            + ", ".join(self.get_valid_moves())
            + "."
        )
        move1 = self.agent1.interact(prompt)
        move2 = self.agent2.interact(prompt)

        payoff1, payoff2 = self.get_payoff(move1, move2)

        self.total_payoffs[0] += payoff1
        self.total_payoffs[1] += payoff2

        summary = (
            f"Round {round_num}: Agent 1 chose {move1}, Agent 2 chose {move2}. "
            f"Payoffs: Agent 1 = {payoff1}, Agent 2 = {payoff2}."
        )
        self.history.append(summary)

        self.agent1.update_history("You are agent 1. " + summary)
        self.agent2.update_history("You are agent 2. " + summary)

        return move1, move2, payoff1, payoff2, summary

    def run_game(self):
        for round_num in range(1, self.n_rounds + 1):
            print(f"\n--- Round {round_num} ---")
            move1, move2, payoff1, payoff2, summary = self.play_round(round_num)
            print(summary)
        print("\nFinal Total Payoffs:")
        print("Agent 1:", self.total_payoffs[0])
        print("Agent 2:", self.total_payoffs[1])

    def introspect(self, agent_num, question):
        if agent_num == 1:
            return self.agent1.interact(question)
        elif agent_num == 2:
            return self.agent2.interact(question)
        else:
            return "Invalid agent number."

    def get_history(self):  # added a history getter
        return self.history
