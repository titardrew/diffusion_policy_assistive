from pathlib import Path
from collections import defaultdict

import numpy as np
import zarr

def calc_dataset_stats(input_path: Path):
    store = zarr.open(str(input_path))
    epi_start = 0
    stats = defaultdict(list)
    for k, v in store['data'].items():
        print(k, v.shape)
    for epi_end in store['meta']['episode_ends']:
        episode_rewards = store['data']['reward'][epi_start: epi_end]
        stats['episode_total_reward'].append(np.sum(episode_rewards))
        stats['episode_length'].append(epi_end - epi_start)
        epi_start = epi_end
    
    for k, v in stats.items():
        print(f"{k} (mean/std): {np.mean(v):.3f}/{np.std(v):.3f}")
        print(f"{k} (min/max): {np.min(v):.3f}/{np.max(v):.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="Path to .zarr")
    args = parser.parse_args()
    calc_dataset_stats(Path(args.input))