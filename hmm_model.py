# hmm_model.py
import numpy as np
from utils import ALPHABET, A2I, I2A

class LetterHMM:
    """
    Simple bigram-based Hidden Markov Model for letter sequences.
    Provides smoothed initial and transition probabilities,
    and posterior inference for partially observed words (Hangman states).
    """
    def __init__(self, smoothing=1e-2):
        self.smoothing = smoothing
        self.K = len(ALPHABET)
        self.pi = np.ones(self.K) / self.K  # start uniform
        self.trans = np.ones((self.K, self.K)) / self.K
        self.vocab = ALPHABET

    def train_from_corpus(self, words):
        """
        Train initial and transition probabilities from a list of lowercase words.
        """
        init_counts = np.zeros(self.K)
        trans_counts = np.zeros((self.K, self.K))

        for w in words:
            letters = [c for c in w if c in A2I]
            if not letters:
                continue
            init_counts[A2I[letters[0]]] += 1
            for a, b in zip(letters, letters[1:]):
                trans_counts[A2I[a], A2I[b]] += 1

        # Laplace (additive) smoothing
        init_counts += self.smoothing
        trans_counts += self.smoothing

        # Normalize
        self.pi = init_counts / init_counts.sum()
        self.trans = trans_counts / trans_counts.sum(axis=1, keepdims=True)

    def forward_backward(self, observed):
        """
        observed: list of characters or None for blank positions.
        Returns posterior probabilities over letters for each position.
        """
        N = len(observed)
        K = self.K
        if N == 0:
            return np.zeros((0, K))

        # Emission matrix
        E = np.ones((N, K), dtype=np.float64)
        for t, ch in enumerate(observed):
            if ch is not None and ch in A2I:
                v = np.zeros(K)
                v[A2I[ch]] = 1.0
                E[t] = v

        # Forward pass
        alpha = np.zeros((N, K), dtype=np.float64)
        alpha[0] = self.pi * E[0]
        alpha[0] /= alpha[0].sum() + 1e-12

        for t in range(1, N):
            alpha[t] = E[t] * (alpha[t - 1] @ self.trans)
            alpha[t] /= alpha[t].sum() + 1e-12

        # Backward pass
        beta = np.ones((N, K), dtype=np.float64)
        for t in range(N - 2, -1, -1):
            beta[t] = (self.trans @ (E[t + 1] * beta[t + 1]))
            beta[t] /= beta[t].sum() + 1e-12

        posterior = alpha * beta
        posterior /= posterior.sum(axis=1, keepdims=True) + 1e-12
        return posterior

    def letter_probs_for_mask(self, masked_word):
        """
        masked_word: string with known letters and '_' for blanks.
        Returns an averaged posterior distribution (26-dim vector)
        over letters likely to fill the blanks.
        """
        observed = [c if c != '_' else None for c in masked_word]
        posterior = self.forward_backward(observed)
        blanks = [i for i, c in enumerate(masked_word) if c == '_']

        if not blanks:
            return np.zeros(self.K, dtype=np.float32)

        avg_post = posterior[blanks].mean(axis=0)
        avg_post /= avg_post.sum() + 1e-12  # re-normalize
        return avg_post.astype(np.float32)
