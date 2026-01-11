# evaluate.py
import torch
from train import load_corpus
from env import HangmanEnv
from hmm_model import LetterHMM
from dqn_agent import DQNAgent
import numpy as np
from tqdm import tqdm

def evaluate_model(model_path='dqn_net_final.pt', corpus_path='corpus.txt', test_path='test.txt', games=None):
    corpus = load_corpus(corpus_path)
    tests = load_corpus(test_path)
    if games is None:
        games = len(tests)
    
    hmm = LetterHMM()
    hmm.train_from_corpus(corpus)
    env = HangmanEnv(tests, hmm, max_len=15, max_wrong=10)  
    state0 = env.reset()
    state_dim = state0.shape[0]
    
    agent = DQNAgent(state_dim)
    agent.net.load_state_dict(torch.load(model_path, map_location='cpu'))
    agent.net.eval()

    wins = 0
    wrongs = 0
    repeated = 0
    by_len = {}

    for i in tqdm(range(games)):
        w = tests[i % len(tests)]
        state = env.reset(word=w)
        while True:
            avail = env.available_actions_mask()
            a = agent.select_action(state, avail, epsilon=0.0)
            next_state, r, done, _ = env.step(a)
            state = next_state
            if done:
                res = env.get_result_metrics()
                wins += 1 if res['won'] else 0
                wrongs += res['wrong']
                repeated += res['repeated']
                wl = len(w)
                by_len.setdefault(wl, []).append(res['won'])
                break

    success_rate = wins / games
    final_score = (success_rate * 2000) - (wrongs * 5) - (repeated * 2)

    print("Games:", games)
    print("Wins:", wins, f"Success rate: {success_rate:.3f}")
    print("Total wrong guesses:", wrongs)
    print("Total repeated guesses:", repeated)
    print(f"\n✅ Final Score (as per Hackathon formula): {final_score:.3f}")

    print("\nPerformance by word length:")
    for l, arr in sorted(by_len.items()):
        print(f"Length {l}: games {len(arr)} success_rate {np.mean(arr):.3f}")

    return {
        "games": games,
        "wins": wins,
        "success_rate": success_rate,
        "wrongs": wrongs,
        "repeated": repeated,
        "final_score": final_score
    }

if __name__ == "__main__":
    evaluate_model()
