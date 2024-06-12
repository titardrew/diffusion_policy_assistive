import zarr
import pandas as pd
import numpy as np


def zarr_to_voraus_pq(in_path, out_path, path_to_eval_json=None):
    store = zarr.open(in_path, mode="r")
    N, Ns = np.array(store.data.state).shape
    N0, Na = np.array(store.data.action).shape
    N_episodes = np.array(store.meta.episode_ends).shape
    assert N0 == N

    columns = [f"mdp_state_{i}" for i in range(Ns)] + [f"mdp_action_{i}" for i in range(Na)]
    df = pd.DataFrame(
        data=np.concatenate([np.array(store.data.state), np.array(store.data.action)], axis=1),
        columns=columns,
    )
    if path_to_eval_json:
        import json
        with open(path_to_eval_json, "r") as eval_file:
            json_data = json.load(eval_file)
            success_mask = np.array(json_data["test/success_episodes"]).astype(np.bool_)
            anomaly_mask = ~success_mask
    else:
        anomaly_mask = np.zeros(N_episodes).astype(np.bool_)

    samples_col = np.zeros(N).astype(np.int_)
    time_col = np.zeros(N)
    anomaly_col = np.zeros(N).astype(np.bool_)
    category_col = np.zeros(N).astype(np.int_)
    setting_col = np.zeros(N).astype(np.int_)
    prev_epi_end = 0
    for sample_id, epi_end in enumerate(np.array(store.meta.episode_ends)):
        anomaly_mask[sample_id] = (epi_end - prev_epi_end == 200) 
        samples_col[prev_epi_end: epi_end] = sample_id
        time_col[prev_epi_end: epi_end] = np.arange(epi_end - prev_epi_end) / 10  # assumes 10 hz
        anomaly_col[prev_epi_end: epi_end] = anomaly_mask[sample_id]
        # 1 - ANOMALY, 0 - OKAY
        category_col[prev_epi_end: epi_end] = int(anomaly_mask[sample_id]) * 1
        # 1 - ANOMALY, 0 - OKAY
        setting_col[prev_epi_end: epi_end] = int(anomaly_mask[sample_id]) * 1
        prev_epi_end = epi_end

    df["sample"] = samples_col.astype(np.int_)
    df["anomaly"] = anomaly_col
    df["category"] = category_col
    df["setting"] = setting_col
    df["active"] = 1  # unused AFAIK
    df["action"] = 0  # EXECUTE_POLICY
    
    df["time"] = time_col

    df.to_parquet(out_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    parser.add_argument("--eval_json")
    parser.add_argument("--concat_inputs", action="store_true")
    args = parser.parse_args()

    from diffusion_policy.scripts.cat_stores import cat_zarr_stores
    if args.concat_inputs:
        from pathlib import Path
        output_path = Path(args.input_path) / "cat.zarr"
        input_paths = [Path(p) for p in Path(args.input_path).glob("*.zarr") if "cat" not in str(p)]
        cat_zarr_stores(input_paths, output_file=output_path)
        input_path = str(output_path)
    else:
        input_path = args.input_path
    zarr_to_voraus_pq(input_path, args.output_path, path_to_eval_json=args.eval_json)
