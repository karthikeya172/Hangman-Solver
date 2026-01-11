# train.py
import numpy as np
import random
import torch
from utils import ALPHABET
from hmm_model import LetterHMM
from env import HangmanEnv
from dqn_agent import DQNAgent
import pickle
import matplotlib.pyplot as plt
from tqdm import trange
import os

def load_corpus(path):
    with open(path, 'r', encoding='utf-8') as f:
        words = [w.strip().lower() for w in f if w.strip()]
    words = [w for w in words if all(ch in ALPHABET for ch in w) and 1 <= len(w) <= 15]
    return words

def exp_epsilon(ep, start_eps=1.0, end_eps=0.05, decay_rate=0.995):
    # faster exponential decay
    return max(end_eps, start_eps * (decay_rate ** ep))

def train_dqn(corpus_path='corpus.txt', test_path='test.txt',
              episodes=8000, max_len=15, checkpoint_every=1000):
    corpus = load_corpus(corpus_path)
    test_words = load_corpus(test_path)

    short_words = [w for w in corpus if len(w) <= 6]
    medium_words = [w for w in corpus if 7 <= len(w) <= 10]
    long_words = [w for w in corpus if len(w) >= 11]

    hmm = LetterHMM()
    hmm.train_from_corpus(corpus)

    # use higher allowed wrong guesses
    env = HangmanEnv(corpus, hmm, max_len=max_len, max_wrong=10)
    state0 = env.reset()
    state_dim = state0.shape[0]

    agent = DQNAgent(
        state_dim,
        action_dim=26,
        lr=5e-5,
        gamma=0.99,
        buffer_size=80000,
        batch=128,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=4000
    )

    rewards_per_ep = []
    start_ep = 1
    if os.path.exists("dqn_checkpoint_latest.pt"):
        try:
            agent.net.load_state_dict(torch.load("dqn_checkpoint_latest.pt", map_location='cpu'))
            agent.target.load_state_dict(agent.net.state_dict())
            with open("rewards_latest.pkl", "rb") as f:
                rewards_per_ep = pickle.load(f)
            start_ep = len(rewards_per_ep) + 1
            print("Resumed from checkpoint at episode", start_ep)
        except Exception:
            print("Found checkpoint but failed to load; starting fresh.")

    print("Starting training for", episodes, "episodes (start_ep =", start_ep, ")")
    losses = []

    for ep in trange(start_ep, episodes + 1):
        # curriculum: short -> medium -> long
        if ep <= int(0.3 * episodes) and len(short_words) > 0:
            state = env.reset(word=random.choice(short_words))
        elif ep <= int(0.6 * episodes) and len(medium_words) > 0:
            state = env.reset(word=random.choice(medium_words))
        else:
            state = env.reset()

        total_reward = 0.0
        epsilon = exp_epsilon(ep, start_eps=1.0, end_eps=0.05, decay_rate=0.995)

        for step in range(140):  # allow slightly more guesses
            avail_mask = env.available_actions_mask()
            action = agent.select_action(state, avail_mask, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, float(done), avail_mask)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            total_reward += reward
            state = next_state
            if done:
                break

        rewards_per_ep.append(total_reward)

        if ep % 200 == 0:
            avg_r = np.mean(rewards_per_ep[-200:])
            print(f"Ep {ep:5d} | Avg Reward (last 200): {avg_r:6.3f} | eps = {epsilon:5.3f}")

        if ep % checkpoint_every == 0:
            torch.save(agent.net.state_dict(), f"dqn_checkpoint_{ep}.pt")
            torch.save(agent.net.state_dict(), "dqn_checkpoint_latest.pt")
            with open("rewards_latest.pkl", "wb") as f:
                pickle.dump(rewards_per_ep, f)
            with open("hmm_checkpoint.pkl", "wb") as f:
                pickle.dump({"pi": hmm.pi, "trans": hmm.trans}, f)
            print(f"Saved checkpoint at episode {ep}")

    torch.save(agent.net.state_dict(), "dqn_net_final.pt")
    with open("training_rewards_final.pkl", "wb") as f:
        pickle.dump(rewards_per_ep, f)
    with open("hmm_model_final.pkl", "wb") as f:
        pickle.dump({"pi": hmm.pi, "trans": hmm.trans}, f)

    # plot
    plt.figure(figsize=(9,4))
    plt.plot(rewards_per_ep)
    plt.title("Training rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reward_curve_final.png", dpi=200)
    plt.show()

    print("Training finished. Saved dqn_net_final.pt and logs.")
    return agent, hmm, corpus, test_words, rewards_per_ep

if __name__ == "__main__":
    train_dqn(episodes=8000)
