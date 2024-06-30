import collections
from pathlib import Path
import os
import time

import click
import tensorboardX as tbx
import numpy as np
import pandas as pd
import torch
import tqdm

from torch.utils.data import DataLoader

from diffusion_policy.dataset.assistive_dataset import AssistiveLowdimDataset
from safety_model.utils import multi_repeat

TENSOR_FILES = {}
def save_tensor(out_path: str, tensor, max_writes=None):
    global TENSOR_FILES

    if out_path not in TENSOR_FILES:
        TENSOR_FILES[out_path] = (open(out_path, "w"), 0)

    f, num_writes = TENSOR_FILES[out_path]
    if max_writes is None or num_writes < max_writes:
        f.write(f"\n shape={tensor.shape} mean={tensor.float().mean()} \n {tensor}")
        TENSOR_FILES[out_path] = f, num_writes + 1
        f.flush()

def _loss_str(loss_dict):
    return ", ".join([f"{loss_key}: {loss_value:.6f}" for loss_key, loss_value in loss_dict.items()])

class StatsLogger:
    BACKENDS = ['tb', 'wandb']
    def __init__(self, experiment_path: str, comment: str = None, project: str = None, backend='wandb'):
        self.wandb = None
        self.writer: tbx.SummaryWriter = None

        if experiment_path is not None and Path(experiment_path).exists():
            timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
            experiment_path = f"{experiment_path.rstrip(os.sep)}_{timestr}"
            assert not Path(experiment_path).exists(), experiment_path

        if backend == "wandb":
            import wandb
            if experiment_path is not None:
                experiment_name = Path(experiment_path).stem
            else:
                experiment_name = None
            wandb.init(project=(project if project else "anomaly"), name=experiment_name)
            self.wandb = wandb
        elif backend == "tb":
            assert experiment_path is not None, "for tb, specify experiment_path"
            self.writer = tbx.SummaryWriter(experiment_path, comment=comment)
        else:
            raise NotImplementedError(f"Unknown backend {backend}!")
    
    def finish(self):
        if self.wandb:
            self.wandb.finish()

    def _add_scalars(self, main_tag, loss_dict, iteration):
        log_dict = {f"{main_tag}/{k}": v for k, v in loss_dict.items()}
        if self.writer:
            for k, v in log_dict.items():
                self.writer.add_scalar(k, v, global_step=iteration)
        if self.wandb:
            log_dict.update({"epoch": iteration})
            self.wandb.log(log_dict)
    
    def add_config(self, cfg_dict):
        if self.wandb:
            self.wandb.config = cfg_dict
    
    def add_model(self, model):
        if self.wandb:
            self.wandb.watch(model)
    
    def add_train_metrics(self, train_loss_dict: dict, iteration: int):
        print(f"[TRAIN] Epoch: {iteration}, Metrics: [{_loss_str(train_loss_dict)}]")
        self._add_scalars("train", train_loss_dict, iteration)
    
    def add_val_metrics(self, val_loss_dict: dict, iteration: int):
        print(f"[ VAL ] Epoch: {iteration}, Metrics: [{_loss_str(val_loss_dict)}]")
        self._add_scalars("val", val_loss_dict, iteration)

    def add_test_metrics(self, test_loss_dict: dict, iteration: int):
        print(f"[TEST ] Epoch: {iteration}, Metrics: [{_loss_str(test_loss_dict)}]")
        self._add_scalars("test", test_loss_dict, iteration)



def get_registered_safety_model_file_and_params(safety_model_name, safety_model_metric=None):
    safety_model_path = {
        # feeding: Tuning the VAE ensemble size.
        "vae0_e5": "ensemble_vae_sm_100_To8_Ta8.pth",
        "vae0_e3": "ensemble_vae_sm_100_To8_Ta8_E3.pth",
        "vae0_e1": "ensemble_vae_sm_100_To8_Ta8_E1.pth",

        # feeding 250
        "feeding_250_state_predictor": "feeding_250_ensemble_state_prediction_sm_To3_Ta3_H8_O3.pth",
        "feeding_250_state_predictor_ss": "feeding_250_ensemble_state_prediction_sm_To1_Ta1_H2_O1.pth",
        "feeding_250_vae": "ensemble_vae_sm_250_To8_Ta16_E5.pth",

        # feeding 50
        "feeding_50_state_predictor": "feeding_50_ensemble_state_prediction_sm_To3_Ta3_H8_O3.pth",
        "feeding_50_state_predictor_ss": "feeding_50_ensemble_state_prediction_sm_To1_Ta1_H2_O1.pth",
        "feeding_50_vae": "ensemble_vae_sm_50_To8_Ta16_E5.pth",

        # feeding 100
        "feeding_100_state_predictor": "ensemble_state_prediction_sm_Feeding_100_To3_Ta3_H8_O3.pth",
        "feeding_100_state_predictor_ss": "feeding_100_ensemble_state_prediction_sm_To1_Ta1_H2_O1.pth",
        "feeding_100_vae": "ensemble_vae_sm_100_To8_Ta16_E5.pth",

        # drinking
        "drinking_state_predictor": "drinking_ensemble_state_prediction_sm_To3_Ta3_O3_H8_E5.pth",
        "drinking_state_predictor_ss": "drinking_ensemble_state_prediction_sm_To1_Ta1_H2_O1.pth",
        "drinking_vae": "drinking_ensemble_vae_sm_250_To8_Ta16_E5.pth",

        # bed bathing
        "bed_bathing_state_predictor": "bed_bathing_ensemble_state_prediction_sm_To3_Ta3_O3_H8_E5.pth",
        "bed_bathing_state_predictor_ss": "bed_bathing_ensemble_state_prediction_sm_To1_Ta1_H2_O1.pth",
        "bed_bathing_vae": "bed_bathing_ensemble_vae_sm_250_To8_Ta16_E5.pth",

        # arm manipulation
        "arm_manipulation_state_predictor": "arm_manipulation_ensemble_state_prediction_sm_To3_Ta3_O3_H8_E5.pth",
        "arm_manipulation_state_predictor_ss": "arm_manipulation_ensemble_state_prediction_sm_To1_Ta1_H2_O1.pth",
        "arm_manipulation_vae": "arm_manipulation_ensemble_vae_sm_250_To8_Ta16_E5.pth",

        # scratch itch
        "scratch_itch_state_predictor": "scratch_itch_ensemble_state_prediction_sm_To3_Ta3_O3_H8_E5.pth",
        "scratch_itch_state_predictor_ss": "scratch_itch_ensemble_state_prediction_sm_To1_Ta1_H2_O1.pth",
        "scratch_itch_vae": "scratch_itch_ensemble_vae_sm_250_To4_Ta8_E5.pth",
    }[safety_model_name]

    METRICS = ["mse", "ensemble", "gate", "recon", "pw_kl"]

    # defaults
    safety_model_kwargs = {
        # mse
        "mse_threshold": 0.09,

        # ensemble
        "std_threshold": 0.09,

        # recon
        "reconstruction_threshold": 4.0,

        # pw_kl
        "pw_kl_threshold": 0.0004,

        # gate
        "gate_n_sigmas": 2.0,
        "gate_min_sigma": 0.1,
    }

    if safety_model_metric:
        assert safety_model_metric in METRICS, f"Unknown metric {safety_model_metric}. Must be one of {METRICS}"
        safety_model_kwargs.update({"metric": safety_model_metric})
    else:
        if "vae" in safety_model_name:
            safety_model_metric = "pw_kl"
        elif "state_predictor":
            safety_model_metric = "ensemble"
        else:
            raise NotImplementedError()

    if "feeding_50" in safety_model_name:
        safety_model_kwargs.update({
            "std_threshold": 0.07,
            "pw_kl_threshold": 0.0004,
        })
    elif "drinking" in safety_model_name:
        safety_model_kwargs.update({
            "std_threshold": 0.09,
            "pw_kl_threshold": 0.00004,
        })
    elif "bed_bathing" in safety_model_name:
        safety_model_kwargs.update({
            "std_threshold": 0.015,
            "pw_kl_threshold": 0.000007,
        })
    elif "arm_manipulation" in safety_model_name:
        safety_model_kwargs.update({
            "std_threshold": 0.25,
            "pw_kl_threshold": 0.00005,
        })
    elif "scratch_itch" in safety_model_name:
        safety_model_kwargs.update({
            "std_threshold": 0.10,
            "pw_kl_threshold": 0.000009,
        })

    return safety_model_path, safety_model_kwargs

class MultibatchDataLoader:
    def __init__(self, ensemble_size, *dataloader_args, **dataloader_kwargs):
        self.dataloaders = [DataLoader(*dataloader_args, **dataloader_kwargs) for _ in range(ensemble_size)]
        self.iters = None

    def __iter__(self):
        self.iters = [iter(dataloader) for dataloader in self.dataloaders]
        return self
    
    def __next__(self):
        if self.iters is None:
            raise StopIteration
        vals = []
        for data_iter in self.iters:
            vals.append(next(data_iter))

        if isinstance(vals[0], dict):
            # Loaders that we use return dicts of batches.
            # So we have to stack those batches for each key.
            multibatch = dict()
            for key in vals[0].keys():
                key_vals = []
                for val in vals:
                    key_vals.append(val[key])
                multibatch[key] = torch.stack(key_vals)
        else:
            # A normal batch
            multibatch = torch.stack(vals)

        return multibatch

ENV_TYPES = ["Feeding", "Drinking", "ArmManipulation", "BedBathing", "ScratchItch"]

@click.command()
@click.option("--zarr_path", default="teleop_datasets/FeedingJaco-v1.zarr", type=Path, help="Path to a datset.")
@click.option("--test_parquet_path", default=None, type=Path, help="Path to a test parquet datset.")
@click.option("--test_zarr_path", default=None, type=Path, help="Path to a test zarr datset.")
@click.option("--num_epochs", default=50, help="Number of epochs to train.")
@click.option("--in_horizon", default=1, help="Number of obs/acts as input to the safety model.")
@click.option("--gap_horizon", default=1, help="Number of obs/acts skipped between in and out.")
@click.option("--out_obs_horizon", default=1, help="Number of obs as out from the safety model.")
@click.option("--max_episode_length", default=200, help="Total number of obs/acts in a sample. Used only in full_episode mode")
@click.option("--batch_size", default=256)
@click.option("--save_freq", default=50)
@click.option("--test_freq", default=8)
@click.option("--test_zarr_freq", default=0)
@click.option("--test_granularity", default=10, help="Time step size for 'playback' tests. 10 by default.")
@click.option("--env_type", default="Feeding", type=click.Choice(ENV_TYPES), help="Environment type.")
@click.option("--lr", default=3e-4, help="Learing rate.")
@click.option("--kl_weight", default=5, help="KL weight for CVAE.")
@click.option("--val_ratio", default=0.1, help="Fraction of data used for validation.")
@click.option("--save_path", default="ensemble_state_prediction_sm3.pth", help="Output checkpoint path.")
@click.option("--ensemble_size", default=5, help="Number of members of ensemble.")
@click.option("--model_type", default="state_predictor", type=click.Choice(["state_predictor", "vae", "cvae", "mvt_flow"]), help="Type of a model.")
@click.option("--metric", default=None, help="Override a metric type.")
@click.option("--experiment_path", default=None, help="Path to a tensorboard experiment folder. If not specified, TB logging is off.")
@click.option("--project", default=None, help="Name of a project (wandb).")
@click.option("--backend", default="wandb", type=click.Choice(StatsLogger.BACKENDS), help="Type of a stats logging backend.")
@click.option("--full_episode", is_flag=True, show_default=True, default=False, help="Full episode as input.")
@click.option("--device", default="cpu", help="Device.")
@click.option("--use_times", is_flag=True, show_default=True, default=False, help="Add times to inputs.")
@click.option("--use_maximum", is_flag=True, show_default=True, default=False, help="Add max score up to t as a score for t.")
def _train(
    zarr_path: Path,
    test_parquet_path: Path = None,
    test_zarr_path: Path = None,  # for debugging
    num_epochs=50,
    in_horizon=1,
    gap_horizon=1,
    out_obs_horizon=1,
    max_episode_length=200,
    batch_size=256,
    save_freq=50,
    test_freq=8,
    test_zarr_freq=0,
    test_granularity=10,
    env_type="Feeding",
    lr=3e-4,
    kl_weight=5,
    val_ratio=0.1,
    save_path="ensemble_state_prediction_sm3.pth",
    ensemble_size=5,
    model_type="state_predictor",
    metric=None,
    experiment_path=None,
    project=None,
    backend="wandb",
    full_episode=False,
    device="cpu",
    use_times=False,
    use_maximum=False,
):
    train(**locals)

def train(
    zarr_path: Path,
    test_parquet_path: Path = None,
    test_zarr_path: Path = None,  # for debugging
    num_epochs=50,
    in_horizon=1,
    gap_horizon=1,
    out_obs_horizon=1,
    max_episode_length=200,
    batch_size=256,
    save_freq=50,
    test_freq=8,
    test_zarr_freq=0,
    test_granularity=10,
    env_type="Feeding",
    lr=3e-4,
    kl_weight=5,
    val_ratio=0.1,
    save_path="ensemble_state_prediction_sm3.pth",
    ensemble_size=5,
    model_type="state_predictor",
    metric=None,
    experiment_path=None,
    project=None,
    backend="wandb",
    full_episode=False,
    device="cpu",
    use_times=False,
    use_maximum=False,
):

    def get_env_params(env_type):
        return {
            "Feeding": (25, 7),
            "Drinking": (25, 7),
            "ArmManipulation": (45, 14),
            "BedBathing": (24, 7),
            "ScratchItch": (30, 7),
        }[env_type]

    observation_size, action_size = get_env_params(env_type)
    if full_episode:
        assert max_episode_length >= in_horizon + gap_horizon + out_obs_horizon, \
            f"{max_episode_length} < {in_horizon} + {gap_horizon} + {out_obs_horizon}"
        in_horizon = 0
        pad_before = 0
        pad_after = 0
        max_horizon = max_episode_length
        min_horizon = 1
    else:
        pad_before = in_horizon - 1
        pad_after = 0  #out_obs_horizon + gap_horizon - 1
        max_horizon = in_horizon + gap_horizon + out_obs_horizon
        min_horizon = in_horizon + gap_horizon + out_obs_horizon

    def load_dataset(path, val_ratio):
        return AssistiveLowdimDataset(
            str(path),
            horizon=max_horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            val_ratio=val_ratio,
            max_train_episodes=None,
            whole_sequence_sampling=full_episode,
            gap_horizon=gap_horizon,
            out_obs_horizon=out_obs_horizon,
        )

    dataset = load_dataset(zarr_path, val_ratio=val_ratio)

    if test_zarr_path and test_zarr_freq > 0:
        test_zarr_dataset = load_dataset(test_zarr_path, val_ratio=0.0)
        test_zarr_dataloader = DataLoader(test_zarr_dataset, batch_size=batch_size, shuffle=False)
    else:
        test_zarr_dataset = None
        test_zarr_dataloader = None

    train_dataloader = MultibatchDataLoader(ensemble_size, dataset, batch_size=batch_size, shuffle=True)
    normalizer = dataset.get_normalizer()

    val_dataset = dataset.get_validation_dataset()
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if test_parquet_path and test_freq > 0:
        if env_type == "voraus_ad":
            # NOTE: not tested!
            from voraus_ad_dataset.voraus_ad import ANOMALY_CATEGORIES, Signals, load_torch_dataloaders
            # Load the dataset as torch data loaders.
            _, _, _, test_dataloader = load_torch_dataloaders(
                dataset=test_parquet_path,
                batch_size=1,
                columns=Signals.machine(),
                seed=42,
                frequency_divider=10.0,
                train_gain=1.0,
                normalize=False,
                pad=False,
            )
        else:
            from voraus_ad_dataset.assistive_ad import ANOMALY_CATEGORIES, get_signals, get_env_params, load_torch_dataloaders
            state_size, action_size = get_env_params(env_type)
            Signals = get_signals(state_size, action_size)
            _, test_dataset, _, test_dataloader = load_torch_dataloaders(
                train_dataset=test_parquet_path,  # hack
                test_dataset=test_parquet_path,
                batch_size=1,
                columns=Signals.state() + Signals.action(),
                seed=42,
                frequency_divider=1.0,
                train_gain=1.0,
                normalize=False,
                pad=False,
                env_type=env_type,
            )

    save_state_dict = False
    if "state_predictor" in model_type:
        from safety_model.state_prediction import EnsembleStatePredictionSM
        DEVICE = device
        safety_model = EnsembleStatePredictionSM(
            observation_size=observation_size,
            action_size=action_size,
            in_horizon=in_horizon,
            gap_horizon=gap_horizon,
            out_obs_horizon=out_obs_horizon,
            horizon=max_horizon,
            ensemble_size=ensemble_size,
            device=DEVICE,
            use_times=use_times,
            backbone_type="mlp" if "mlp" in model_type else "cnn",
            use_input_hack="hack" in model_type,
        ).to(DEVICE)
        safety_model.set_normalizer(normalizer.to(DEVICE))
    elif model_type == "vae":
        DEVICE=device
        from safety_model.vae import VariationalAutoEncoderSM
        safety_model = VariationalAutoEncoderSM(
            observation_size=observation_size,
            action_size=action_size,
            in_horizon=in_horizon,
            horizon=max_horizon,
            ensemble_size=ensemble_size,
            hidden_size=100,
            learning_rate=lr,
            embedding_size=64,
            device=DEVICE,
        ).to(DEVICE)
        safety_model.set_normalizer(normalizer.to(DEVICE))
    elif model_type == "cvae":
        DEVICE=device
        from safety_model.cvae import ConditionalVariationalAutoEncoderSM
        assert ensemble_size == 1, "CVAE does not use ensembling!"
        safety_model = ConditionalVariationalAutoEncoderSM(
            observation_size=observation_size,
            action_size=action_size,
            in_horizon=in_horizon,
            gap_horizon=gap_horizon,
            out_obs_horizon=out_obs_horizon,
            max_horizon=max_horizon,
            device=DEVICE,
            kl_weight=kl_weight,
            use_times=use_times,
        ).to(DEVICE)
        safety_model.set_normalizer(normalizer.to(DEVICE))
    elif model_type == "mvt_flow":
        DEVICE=device
        from safety_model.mvt_flow import MVTFlowSM
        assert ensemble_size == 1, "MVT-Flow does not use ensembling!"
        assert full_episode, "MVT-Flow does not support non full episode mode!"
        save_state_dict = True
        safety_model = MVTFlowSM(
            observation_size=observation_size,
            action_size=action_size,
            max_horizon=max_horizon,
            device=DEVICE
        ).to(DEVICE)
        safety_model.set_normalizer(normalizer.to(DEVICE))
    else:
        raise NotImplementedError()

    if metric:
        safety_model.override_params({"metric": metric})
    
    stats_logger = StatsLogger(experiment_path=experiment_path, comment="test comment", project=project, backend=backend)

    optimizer = torch.optim.Adam(safety_model.parameters(), lr=lr)
    scheduler = safety_model.get_scheduler(optimizer)

    def _val_epoch():
        with torch.no_grad():
            safety_model.eval()
            loss_dict = collections.defaultdict(list)
            for batch in val_dataloader:
                multibatch = multi_repeat(batch, times=ensemble_size)
                for k, v in multibatch.items():
                    multibatch[k] = v.to(safety_model.device)
                loss_dict_one = safety_model.compute_validation_loss(multibatch)
                for key in loss_dict_one.keys():
                    loss_dict[key].append(loss_dict_one[key])
            for key, vals in loss_dict.items():
                loss_dict[key] = np.mean(vals)
            return loss_dict

    def _test_zarr_epoch():
        with torch.no_grad():
            safety_model.eval()
            loss_dict = collections.defaultdict(list)
            for batch in tqdm.tqdm(test_zarr_dataloader):
                multibatch = multi_repeat(batch, times=ensemble_size)
                for k, v in multibatch.items():
                    multibatch[k] = v.to(safety_model.device)
                loss_dict_one = safety_model.compute_validation_loss(multibatch)
                for key in loss_dict_one.keys():
                    loss_dict[f"{key}_test_zarr"].append(loss_dict_one[key])
            for key, vals in loss_dict.items():
                loss_dict[key] = np.mean(vals)
            return loss_dict
    
    def _test_epoch():
        result_list = []
        safety_model.eval()
        test_val_loss_dict = collections.defaultdict(list)
        for _, (tensors, labels) in tqdm.tqdm(enumerate(test_dataloader)):
            assert tensors.shape[0] == 1, "Batches aren't supported yet!"
            tensor = tensors[0].float().to(safety_model.device)

            # TODO: tensors -> anomality_score
            anomality_score = torch.ones(tensors.shape[0], device=safety_model.device) * (-torch.inf)

            # for each sequence in test, we "play" it and save the anomality scores
            # we only save TIME_GRAN anomality scores at time steps that we call "milestones".
            TIME_GRAN = 21
            MAX_T = 200
            milestone_ts = np.linspace(0, MAX_T, TIME_GRAN, dtype=np.int_).tolist()[1:] + [np.inf]  # exclude 0 and add inf to the end
            milestone_id = 0

            epi_len = tensor.shape[0]
            if full_episode:
                # pad the tensor
                # [s_a0, s_a1, ..., s_aT, 0 ... 0], len=max_epi_len (e.g. 200)
                tensor_padded = torch.zeros((max_episode_length, tensor.shape[1]))
                tensor_padded[:epi_len] = tensor
                tensor = tensor_padded
                start_index = 1 + gap_horizon + out_obs_horizon
            else:
                # no need to pad, just start as soon as there is enough input/output steps.
                start_index = in_horizon + gap_horizon + out_obs_horizon

            if hasattr(safety_model, "_segment_first") and safety_model._segment_first:
                obs = tensor[None, :, :observation_size]
                act = tensor[None, :, observation_size:]
                length = torch.tensor([[epi_len]])  # length of a sample
                safety_model.segment_anomaly_and_remember(
                    {"obs": obs.clone(), "action": act.clone(), "length": length.clone()},
                )

            # start "playing the episode"
            milestone_scores = {}
            if test_granularity != milestone_ts[1] - milestone_ts[0]:
                ts_list = range(0, MAX_T+test_granularity, test_granularity)
            else:
                ts_list = milestone_ts[:-1]

            for i_step in ts_list:
                if i_step < start_index: continue
                if i_step > epi_len: continue
                if full_episode:
                    obs = torch.zeros((1, max_episode_length, observation_size), device=safety_model.device)
                    act = torch.zeros((1, max_episode_length, action_size), device=safety_model.device)
                    obs[:, :i_step] = tensor[:i_step, :observation_size]
                    act[:, :i_step] = tensor[:i_step, observation_size:]
                    if i_step < max_episode_length:
                        obs[:, i_step:] = tensor[i_step-1, :observation_size]
                    length = torch.tensor([[i_step]])  # length of a sample
                else:
                    horizon = in_horizon + gap_horizon + out_obs_horizon
                    obs = tensor[None, i_step - horizon: i_step, :observation_size]
                    act = tensor[None, i_step - horizon: i_step, observation_size:]
                    length = torch.tensor([[i_step]])  # length of a sample

                _, _, score = safety_model.check_safety_runner(
                    {"obs": obs.clone(), "action": act.clone(), "length": length.clone()},
                    return_stats=True,
                    return_score=True,
                )
                if use_maximum:
                    anomality_score = torch.maximum(anomality_score, torch.from_numpy(score).to(safety_model.device))
                else:
                    anomality_score[:] = torch.from_numpy(score).to(safety_model.device)

                multibatch = multi_repeat({"obs": obs, "action": act, "length": length},
                    safety_model.ensemble_size)
                for k, v in safety_model.compute_validation_loss(multibatch).items():
                    test_val_loss_dict[k].append(v)
                
                # fill the milestone's anomality score
                if i_step >= milestone_ts[milestone_id]:
                    while i_step >= milestone_ts[milestone_id]:
                        milestone_id += 1
                    milestone_scores[f"score_{milestone_ts[milestone_id - 1]:.2f}"] = anomality_score.cpu()

            if hasattr(safety_model, "_segment_first") and safety_model._segment_first:
                safety_model.clear_segment_anomaly()

            # Append the anomaly score and the labels to the results list.
            for j in range(anomality_score.shape[0]):
                # HACK(aty): anomalies are less than 200 timesteps in length.
                labels["anomaly"] = [(epi_len == MAX_T)]
                labels["category"] = ["FAILURE" if (epi_len == MAX_T) else "NORMAL_OPERATION"]
                result_labels = {k: v[j].item() if isinstance(v, torch.Tensor) else v[j] for k, v in labels.items()}
                result_labels.update(score=anomality_score[j].item())
                result_labels.update({k: v[j] for k, v in milestone_scores.items()})
                result_list.append(result_labels)

        # calculate metrics for each milestone
        results = pd.DataFrame(result_list)
        test_metrics = collections.defaultdict(list)
        for category in ANOMALY_CATEGORIES:
            dfn = results[(results["category"] == category.name) | (~results["anomaly"])]
            import ood_metrics
            for col in results.columns:
                if "score" in col:
                    score = dfn[col].fillna(dfn["score"]).clip(-1e9, 1e9)
                    m = ood_metrics.calc_metrics(score.values, dfn["anomaly"], pos_label=1)
                    for k, v in m.items():
                        test_metrics[f"{col}_{k}"].append(v)
        test_metrics.update(test_val_loss_dict)
        mean_test_metrics = {}
        for k, v in test_metrics.items():
            mean_test_metrics[k] = np.mean(v)
        return mean_test_metrics

    def _save(save_path, save_state_dict=False, epoch=None):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        epoch = epoch if epoch else "final"
        save_path = save_path.format(epoch=epoch)
        assert Path(save_path).parent.exists(), save_path
        if save_state_dict:
            torch.save(safety_model.state_dict(), save_path)
        else:
            torch.save(safety_model, save_path)
        print(f"[SAVE ] Model (epoch={epoch}) has been saved to {save_path}")

    val_loss_dict = _val_epoch()
    stats_logger.add_val_metrics(val_loss_dict, iteration=0)
    if test_zarr_path and test_zarr_freq > 0:
        test_loss_dict = _test_zarr_epoch()
        stats_logger.add_test_metrics(test_loss_dict, iteration=0)
    if test_parquet_path and test_freq > 0:
        test_loss_dict = _test_epoch()
        stats_logger.add_test_metrics(test_loss_dict, iteration=0)

    for i_epoch in range(1, num_epochs + 1):
        losses = []
        safety_model.train()
        for _, multibatch in enumerate(train_dataloader):
            optimizer.zero_grad()
            for k, v in multibatch.items():
                multibatch[k] = v.to(safety_model.device)
            loss, _ = safety_model.compute_loss(multibatch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        scheduler.step()

        val_loss_dict = _val_epoch()
        stats_logger.add_train_metrics({"loss": np.mean(losses)}, iteration=i_epoch)
        stats_logger.add_val_metrics(val_loss_dict, iteration=i_epoch)
        if test_zarr_path and test_zarr_freq > 0:
            if i_epoch % test_zarr_freq == 1 or i_epoch == num_epochs:
                test_loss_dict = _test_zarr_epoch()
                stats_logger.add_test_metrics(test_loss_dict, iteration=i_epoch)
        if test_parquet_path and test_freq > 0:
            if i_epoch % test_freq == 0 or i_epoch == num_epochs:
                test_loss_dict = _test_epoch()
                stats_logger.add_test_metrics(test_loss_dict, iteration=i_epoch)
        if save_freq > 0 and i_epoch % save_freq == 1:
            _save(save_path, save_state_dict, i_epoch)

    _save(save_path, save_state_dict)
    stats_logger.finish()

if __name__ == "__main__":
    _train()
