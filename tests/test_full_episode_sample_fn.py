import numpy as np
from diffusion_policy.common.sampler import get_create_indices_fn_whole_sequence

gap_horizon = 1
out_obs_horizon = 2
episode_ends = np.array([10, 18])
episode_mask = np.array([1, 1])

create_indices = get_create_indices_fn_whole_sequence(gap_horizon, out_obs_horizon)
indices = create_indices(episode_ends, sequence_length=10, episode_mask=episode_mask)

indices = np.array(indices)
expected_indices = np.array([
    [ 0,  4,  0,  4,  4],
    [ 0,  5,  0,  5,  5],
    [ 0,  6,  0,  6,  6],
    [ 0,  7,  0,  7,  7],
    [ 0,  8,  0,  8,  8],
    [ 0,  9,  0,  9,  9],
    [ 0, 10,  0, 10, 10],
    [10, 14,  0,  4,  4],
    [10, 15,  0,  5,  5],
    [10, 16,  0,  6,  6],
    [10, 17,  0,  7,  7],
    [10, 18,  0,  8,  8],
    [10, 18,  0,  8,  9],
    [10, 18,  0,  8, 10],
])
assert np.isclose(indices, expected_indices).all(), indices