from safety_model.experiments.utils import make_experiment_name

EXP_NAME = make_experiment_name(__file__)

def run(start_from: int = 0, exp_name=EXP_NAME):

    from safety_model.train import train
    from safety_model.experiments.utils import get_data_catalog, DataConfig
    import wandb
    from pathlib import Path

    wandb.login()

    catalog = get_data_catalog(invalid_ok=False)

    entry: DataConfig
    for i, (name, entry) in enumerate(catalog.items()):
        print(f"[{i}/{len(catalog)}] Running on {name}")
        if i < start_from: continue

        Path("trained_models").mkdir(exist_ok=True)

        config = dict(
            project=f"{name}",
            save_path=f"trained_models/{name}/{exp_name}" + "/model_{epoch}.pth",
            experiment_path=f"{exp_name}",

            zarr_path=entry.zarr_path,
            test_parquet_path=entry.test_parquet_path,
            test_zarr_path=entry.test_zarr_path,
            max_episode_length=entry.max_episode_length,

            test_freq=5,
            test_zarr_freq=0,

            # I/O DATA PARAMETERS
            env_type=entry.env_name,
            in_horizon=40,
            gap_horizon=7,
            out_obs_horizon=3,
            full_episode=False,

            # ALGORITHM PARAMETERS
            model_type="state_predictor:cnn",
            num_epochs=300,
            use_times=False,
            use_maximum=False,
            batch_size=64,
            lr=8e-4,
            #kl_weight=100,  # unused
            val_ratio=0.1,
            ensemble_size=5,
            metric="ensemble",

            # MISC
            backend="wandb",
            device="cuda",
        )

        train(**config)

if __name__ == "__main__":
    run()
