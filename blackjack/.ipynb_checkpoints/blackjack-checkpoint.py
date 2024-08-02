import numpy as np
import pandas as pd
import altair as alt
from tqdm import tqdm
import os

def create_invert_map(lst):
    return {v: i for i, v in enumerate(lst)}

class BlackJack:

    cards = ["A", 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    agent_sums = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # Referenced as 0, 1, ...
    dealers_cards = ["A", 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Referenced as 0, 1, ...
    ace_usability = [False, True]  # Referenced as 0, 1
    actions_possible = ["S", "H"]  # Referenced as 0, 1

    # Create the maps t iterate through, Dealers Cards Map: {'A': 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}

    agent_sums_map, dealers_cards_map, ace_usability_map, actions_map, cards_map = map(
        create_invert_map,
        [agent_sums, dealers_cards, ace_usability, actions_possible, cards],
    )

    def __init__(
        self, M, epsilon, alpha, seed=0,
    ):
        self.seed = seed
        self.M = M
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q_hist = None
        self.initialize()

    def initialize(self):
        # Q is a 4D array giving the state-action values.
        self.Q = np.zeros(
            (
                len(self.agent_sums),
                len(self.dealers_cards),
                len(self.ace_usability),
                len(self.actions_possible),
            )
        )
        # C is a 4D array counting the times a state-action pair was visited. It's not actually needed for the algorithm.
        self.C = np.zeros_like(self.Q, dtype=int)

        # Create the data directory if it doesn't already exist.
        if not os.path.exists('data'):
            os.mkdir('data')

    def sample_cards(self, N):
        """Draws N samples for the infinite 13 card deck."""
        sampled_cards = np.random.choice(self.cards, size=N, replace=True)
        return [c if c == "A" else int(c) for c in sampled_cards]

    def map_to_indices(self, agent_sum, dealers_card, useable_ace):
        """Map from agent_sum (12-21), dealers_card(ace-10) and useable_ace (True, False) to the indices of Q."""
        return (
            self.agent_sums_map[agent_sum],
            self.dealers_cards_map[dealers_card],
            self.ace_usability_map[useable_ace],
        )

    def behavior_policy(self, agent_sum, dealers_card, useable_ace):
        """Returns H (Hit) or S (Stick) to determine the actions to take during the game."""
        agent_sum_idx, dealers_card_idx, useable_ace_idx = self.map_to_indices(
            agent_sum, dealers_card, useable_ace
        )
        greedy_action = self.Q[
            agent_sum_idx, dealers_card_idx, useable_ace_idx, :
        ].argmax()
        do_greedy = np.random.binomial(1, 1 - self.epsilon + (self.epsilon / 2))
        if do_greedy:
            return self.actions_possible[int(greedy_action)]
        else:
            return self.actions_possible[int(not greedy_action)]

    def target_policy(self, agent_sum, dealers_card, useable_ace):
        """
        The target policy is the same as the behavior policy in on-policy learning. Note: this method is never actually
        referenced.
        """
        return self.behavior_policy(
            agent_sum=agent_sum, dealers_card=dealers_card, useable_ace=useable_ace
        )

    def is_ratio(self, states_remaining, actions_remaining):
        """The Importance Sampling ratio that can be overwritten for off policy control."""
        return 1

    @staticmethod
    def calc_sum_useable_ace(cards):
        """Returns the sum-value of cards and whether there is a useable ace"""
        cards_sum = sum([c for c in cards if c != "A"])
        ace_count = len([c for c in cards if c == "A"])

        if ace_count == 0:
            return cards_sum, False
        else:
            cards_sum_0 = cards_sum + ace_count  # 0 aces count as 11
            cards_sum_1 = cards_sum + 10 + ace_count  # 1 ace counts as 11

            if cards_sum_1 > 21:
                return cards_sum_0, False

            return cards_sum_1, True

    def play_game(self):
        """
        Returns:
            states: a list of states, one for each time state.
            actions: a list of agent actions, one for each time state.
            reward: final return of the game. Either -1, 0, 1
            how: string describing how the game ended.
        """

        agent_cards = self.sample_cards(2)  # Agent is delt 2 cards
        dealers_card = self.sample_cards(1)[0]  # Dealer is (effectively) delt 1 cards

        states = [[agent_cards, dealers_card]]
        actions = []

        # Hit until agent_sum >= 12
        while True:
            agent_cards, dealers_card = states[-1]

            agent_sum, _ = self.calc_sum_useable_ace(agent_cards)
            if agent_sum < 12:
                actions.append("H")
                agent_cards_next = agent_cards + self.sample_cards(1)
                states.append([agent_cards_next, dealers_card])
            else:
                break

        # Play game according to agents policy
        while True:
            agent_cards, dealers_card = states[-1]
            agent_sum, useable_ace = self.calc_sum_useable_ace(agent_cards)

            action = self.behavior_policy(
                agent_sum=agent_sum, dealers_card=dealers_card, useable_ace=useable_ace
            )
            actions.append(action)
            if action == "S":
                states.append([agent_cards, dealers_card])
                break
            else:
                agent_cards_next = agent_cards + self.sample_cards(1)
                states.append([agent_cards_next, dealers_card])

                agent_sum_next, useable_ace = self.calc_sum_useable_ace(
                    agent_cards_next
                )

                if agent_sum_next > 21:
                    how = "bust"
                    reward = -1
                    return states, actions, reward, how

        # Dealer plays
        dealers_cards = [states[-1][1]] + self.sample_cards(1)  # Turn over card
        while True:
            dealers_sum, _ = self.calc_sum_useable_ace(dealers_cards)
            if dealers_sum > 21:
                how = "dealer_bust:" + ",".join([str(c) for c in dealers_cards])
                reward = 1
                return states, actions, reward, how
            if dealers_sum > 16:
                break
            else:
                dealers_cards += self.sample_cards(1)

        agent_sum, useable_ace = self.calc_sum_useable_ace(states[-1][0])

        # If the game hasn't already ended, determine the winner.
        if agent_sum == dealers_sum:
            return states, actions, 0, f"{agent_sum} = {dealers_sum}"
        elif agent_sum > dealers_sum:
            return states, actions, 1, f"{agent_sum} > {dealers_sum}"
        else:
            return states, actions, -1, f"{agent_sum} < {dealers_sum}"

    def get_hyper_str(self):
        """Returns a string uniquely identifying the class arguments."""
        return f"M{self.M}__epsilon{str(self.epsilon).replace('.', '_')}__alpha{str(self.alpha).replace('.', '_')}__seed{str(self.seed)}"

    def get_file_names(self):
        hyper_str = self.get_hyper_str()
        Q_name = "data\\Q_" + hyper_str
        C_name = "data\\C_" + hyper_str
        Q_hist_name = "data\\Q_hist_" + hyper_str
        return Q_name, C_name, Q_hist_name

    def save(self):
        Q_name, C_name, Q_hist_name = self.get_file_names()
        print(f"Saving {Q_name}")
        np.save(Q_name, self.Q)
        print(f"Saving {C_name}")
        np.save(C_name, self.C)
        if self.Q_hist is not None:
            print(f"Saving {Q_hist_name}")
            self.Q_hist.to_pickle(Q_hist_name + ".pkl")

    def load(self):
        Q_name, C_name, Q_hist_name = self.get_file_names()
        print(f"Loading...")
        self.Q = np.load(Q_name + ".npy")
        self.C = np.load(C_name + ".npy")
        try:
            self.Q_hist = pd.read_pickle(Q_hist_name + ".pkl")
        except:
            print("No Q hist to load")

    def _get_ms(self):
        """Returns a list of episodes index's (sometimes called 'm's) for which we record action-value pairs."""
        return list(range(0, self.M + 1, 1000))

    def mc_control(self, track_history=True):

        """Return the algorithm to learn the Q-table."""

        self.initialize()
        np.random.seed(self.seed)

        Q_hist = []
        ms = self._get_ms()

        for m in tqdm(range(self.M + 1)):
            states, actions, reward, how = self.play_game()

            if track_history:
                if m in ms:
                    Q_hist.append(self.get_df("Q").assign(m=m))

            for i, (state, action) in enumerate(zip(states[:-1], actions)):

                agent_cards, dealers_card = state
                agent_sum, useable_ace = self.calc_sum_useable_ace(agent_cards)

                if agent_sum < 12:
                    continue

                agent_sum_idx, dealers_card_idx, useable_ace_idx = self.map_to_indices(
                    agent_sum, dealers_card, useable_ace
                )
                actions_idx = self.actions_map[action]

                q_val = self.Q[
                    agent_sum_idx, dealers_card_idx, useable_ace_idx, actions_idx
                ]

                rho = self.is_ratio(states[i + 1 :], actions[i + 1 :])

                self.Q[
                    agent_sum_idx, dealers_card_idx, useable_ace_idx, actions_idx
                ] = q_val + self.alpha * (rho * reward - q_val)

                self.C[
                    agent_sum_idx, dealers_card_idx, useable_ace_idx, actions_idx
                ] += 1

        if track_history:
            self.Q_hist = pd.concat(Q_hist, axis=0)

        self.save()

    def plot_over_m(
        self, agent_sum, dealers_card, useable_ace, width=400, height=200, **kwargs
    ):
        """Creates a line plot of the action-value of hit and stick as episodes are processed. Note: we don't record
        action-value after every episodes, so not all action-values are shown."""

        if dealers_card == "A":
            dealers_card = f"'{dealers_card}'"

        df = self.Q_hist.query(
            f"(agent_sum == {agent_sum}) & (dealers_card == {dealers_card}) & (useable_ace == {useable_ace})"
        )[["Q", "action", "m"]]

        return (
            alt.Chart(df)
            .mark_line(**kwargs)
            .encode(x="m", y="Q", color="action")
            .properties(height=height, width=width)
        )

    def make_slice_df(self, Q_or_C_slice, name):
        df = pd.DataFrame(
            Q_or_C_slice, index=self.agent_sums, columns=self.dealers_cards
        )
        df.index.name = "agent_sum"
        df.columns.name = "dealers_card"
        return df.stack().to_frame(name)

    def get_df(self, which="Q"):

        assert which in ["Q", "C"]

        array = getattr(self, which)

        df = []
        for idx1 in [0, 1]:
            for idx2 in [0, 1]:

                slc = array[:, :, idx1, idx2]
                slc_df = self.make_slice_df(slc, which)

                if which == "Q":
                    opt = array.argmax(-1)[:, :, idx1]
                    is_opt = opt == idx2

                    slc_df = pd.concat(
                        [slc_df, self.make_slice_df(is_opt.astype(int), "opt")], axis=1
                    )

                slc_df = slc_df.reset_index().assign(
                    useable_ace=self.ace_usability[idx1],
                    action=self.actions_possible[idx2],
                )
                df.append(slc_df)

        return pd.concat(df, axis=0)

    def plot(self, which="Q", height=200, width=350):
        """Returns 2-by-2 grid of heatmaps giving either the action values or the counts each state-action pair was visited."""

        assert which in ["Q", "C"]

        df = self.get_df(which)

        if "Q" in df.columns:
            col = "Q"
            color = alt.Color(col, scale=alt.Scale(domain=[-1, 1]))
        else:
            col = "C"
            color = alt.Color(col)

        heatmap = (
            alt.Chart(df)
            .mark_rect()
            .encode(x="agent_sum:O", y="dealers_card:O", color=color,)
        )
        df["rounded"] = df[col].round(2)

        args = dict(x="agent_sum:O", y="dealers_card:O", text="rounded")
        txt = alt.Chart(df).mark_text().encode(**args)

        if which == "Q":
            opt = (
                alt.Chart(df)
                .mark_rect(fill=None, strokeWidth=2, stroke="green")
                .encode(x="agent_sum:O", y="dealers_card:O")
                .transform_filter(alt.datum.opt == 1)
            )
            chart = heatmap + opt + txt
        else:
            chart = heatmap + txt

        return chart.properties(height=height, width=width).facet(
            row="action", column="useable_ace"
        )