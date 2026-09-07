import random
from collections import defaultdict
from typing import Optional

from src.model.state import State

class QModel:

    def __init__(
        self,
        learning_rate: float = 0.12,
        discount: float = 0.90,
        epsilon: float = 0.20,
        min_epsilon: float = 0.03,
        epsilon_decay: float = 0.9995,
    ):
        self.lr = learning_rate
        self.gamma = discount

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        self.q_table = defaultdict(lambda: [0.0, 0.0])

        self.last_state_key: Optional[tuple[int, ...]] = None
        self.last_action: Optional[int] = None

    def discretize(self, state: State) -> tuple[int, ...]:
        if not state.isFishing:
            return (-999,)

        distance_y = state.fish_y - state.greenbar_center_y

        distance_bin = self._bin_value(
            distance_y,
            bins=(-120, -70, -35, -12, 12, 35, 70, 120),
        )

        fish_velocity_bin = self._bin_value(
            state.fish_velocity_y,
            bins=(-12, -5, -1, 1, 5, 12),
        )

        greenbar_velocity_bin = self._bin_value(
            state.greenbar_velocity_y,
            bins=(-12, -5, -1, 1, 5, 12),
        )

        in_bar = int(state.fish_in_greenbar)

        return (
            distance_bin,
            fish_velocity_bin,
            greenbar_velocity_bin,
            in_bar,
        )

    def _bin_value(
        self,
        value: float,
        bins: tuple[float, ...],
    ) -> int:
        for index, threshold in enumerate(bins):
            if value < threshold:
                return index

        return len(bins)

    def predict(self, state: State) -> int:
        """
        返回动作:
            0 = release
            1 = hold
        """
        if not state.isFishing:
            return 0

        state_key = self.discretize(state)

        if random.random() < self.epsilon:
            action = random.choice([0, 1])
        else:
            q_values = self.q_table[state_key]

            if q_values[1] > q_values[0]:
                action = 1
            else:
                action = 0

        self.last_state_key = state_key
        self.last_action = action

        return action

    def learn(
        self,
        next_state: State,
        reward: float,
        done: bool = False,
    ) -> None:
        if self.last_state_key is None or self.last_action is None:
            return

        old_q = self.q_table[self.last_state_key][self.last_action]

        if done or not next_state.isFishing:
            target_q = reward
        else:
            next_state_key = self.discretize(next_state)
            target_q = reward + self.gamma * max(self.q_table[next_state_key])

        new_q = old_q + self.lr * (target_q - old_q)

        self.q_table[self.last_state_key][self.last_action] = new_q

        self._decay_epsilon()

    def _decay_epsilon(self) -> None:
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay,
        )

    def reset_episode(self) -> None:
        self.last_state_key = None
        self.last_action = None