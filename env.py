# env.py
import random
import numpy as np
from utils import A2I, I2A, word_to_indices, ALPHABET

class HangmanEnv:
    """
    Hangman environment with improved reward shaping and HMM alignment bonus.

    State layout:
      [masked_onehot (max_len * 26), guessed_vec (26),
       word_len_onehot (max_len), guessed_count (1), lives (1),
       hmm_vec (26)]
    """
    def __init__(self, word_list, hmm, max_len=15, max_wrong=10):
        self.word_list = list(word_list)
        self.hmm = hmm
        self.max_len = max_len
        self.max_wrong = max_wrong
        self.reset()

    def reset(self, word=None):
        if word is None:
            word = random.choice(self.word_list)
        self.target = word.lower()
        if len(self.target) > self.max_len:
            self.target = self.target[:self.max_len]
        self.target_indices = word_to_indices(self.target, self.max_len)
        self.word_len = 0
        for idx in self.target_indices:
            if idx >= 0:
                self.word_len += 1
            else:
                break
        self.mask = ['_'] * self.word_len
        self.guessed = set()
        self.wrong = 0
        self.repeated = 0
        self.done = False
        return self._get_state()

    def _get_masked_word(self):
        return ''.join(self.mask)

    def _get_state(self):
        masked_oh = np.zeros((self.max_len, 26), dtype=np.float32)
        for i in range(self.word_len):
            if self.mask[i] != '_':
                masked_oh[i, A2I[self.mask[i]]] = 1.0

        guessed_vec = np.zeros(26, dtype=np.float32)
        for g in self.guessed:
            guessed_vec[A2I[g]] = 1.0

        word_len_vec = np.zeros(self.max_len, dtype=np.float32)
        if 1 <= self.word_len <= self.max_len:
            word_len_vec[self.word_len - 1] = 1.0

        guessed_count = np.array([len(self.guessed) / 26.0], dtype=np.float32)
        lives = np.array([(self.max_wrong - self.wrong) / self.max_wrong], dtype=np.float32)

        masked_word = self._get_masked_word()
        hmm_vec = self.hmm.letter_probs_for_mask(masked_word)  # normalized

        state = np.concatenate([
            masked_oh.flatten(),
            guessed_vec,
            word_len_vec,
            guessed_count,
            lives,
            hmm_vec
        ]).astype(np.float32)
        return state

    def available_actions_mask(self):
        mask = np.ones(26, dtype=np.float32)
        for g in self.guessed:
            mask[A2I[g]] = 0.0
        return mask

    def step(self, letter):
        """
        Accepts letter char or int index.
        Returns: next_state, reward, done, info
        Reward logic:
          - correct guess: +1 per hit + progress bonus (quadratic)
          - HMM alignment bonus: +0.25 * prob_of_chosen_letter (encourages using HMM)
          - wrong guess: -0.7 initially, scales slightly with number of wrongs
          - loss end: -8 additional penalty
          - finish: +12 final bonus
        """
        if isinstance(letter, int):
            if not (0 <= letter < 26):
                raise ValueError("action index out of range")
            letter = I2A[letter]
        letter = letter.lower()
        info = {}

        # repeated guess -> tiny penalty
        if letter in self.guessed:
            self.repeated += 1
            reward = -0.2
            return self._get_state(), reward, False, info

        self.guessed.add(letter)

        # compute HMM alignment prob for the chosen letter in current masked pattern
        masked_word = self._get_masked_word()
        hmm_vec = self.hmm.letter_probs_for_mask(masked_word)  # normalized 26 vector
        hmm_bonus = float(hmm_vec[A2I[letter]]) if (masked_word.count('_') > 0) else 0.0

        if letter in self.target:
            hits = 0
            for i, ch in enumerate(self.target[:self.word_len]):
                if ch == letter and self.mask[i] == '_':
                    self.mask[i] = letter
                    hits += 1
            revealed = (self.word_len - self.mask.count('_'))
            progress_ratio = revealed / max(1, self.word_len)
            # base reward for correct hits + progress bonus (quadratic)
            reward = 1.0 * hits + 4.0 * (progress_ratio ** 2)
            # small HMM alignment reward to encourage following HMM signal
            reward += 0.25 * hmm_bonus
            # finishing bonus
            if '_' not in self.mask:
                reward += 12.0
                self.done = True
            return self._get_state(), reward, self.done, info
        else:
            # wrong guess penalty (mildly increasing)
            self.wrong += 1
            reward = -0.7 - 0.1 * (self.wrong ** 1.05)
            # small negative HMM penalty if chosen letter was very unlikely
            reward += -0.05 * (1.0 - hmm_bonus)
            if self.wrong >= self.max_wrong:
                reward -= 8.0  # extra penalty for losing the game
                self.done = True
            return self._get_state(), reward, self.done, info

    def render(self):
        print("Word:", ''.join(self.mask), "guessed:", ''.join(sorted(self.guessed)),
              f"wrong {self.wrong}/{self.max_wrong}")

    def get_result_metrics(self):
        return {
            "target": self.target,
            "won": ('_' not in self.mask),
            "wrong": self.wrong,
            "repeated": self.repeated
        }
