from pathlib import Path
from dataclasses import dataclass

ENVIRONMENTS = [
    "Feeding",
    "Drinking",
    "ArmManipulation",
    "BedBathing",
    "ScratchItch",    
]

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
DEFAULT_ZARR_TELEOP_ROOT = ROOT_DIR / "teleop_datasets"
DEFAULT_ZARR_ROOT = ROOT_DIR / "AssistiveDiffusion" / "Datasets"
DEFAULT_TEST_ZARR_ROOT = ROOT_DIR / "out_voraus"
DEFAULT_TEST_PARQUET_ROOT = ROOT_DIR / "parquet_datasets"


@dataclass
class DataConfig:
    # Name of the environment.
    env_name: str
    # Is dataset collected from teleoperation
    is_teleop: bool
    # number of trajectories. If None, only one type of size is available.
    size: int
    
    # Train set (collected either with PPO or teleop).
    zarr_path: Path
    # Test set (collected with imitation policy) in zarr format.
    test_zarr_path: Path
    # Test set (collected with imitation policy) in parquet format.
    test_parquet_path: Path

    max_episode_length: int = 200


def validate_catalog(catalog):
    print("Checking catalog validity.")
    is_valid = True
    for name, entry in catalog.items():
        invalid_entry = False

        if not entry.zarr_path.exists():
            print(f"[ERROR] {name} ({entry.env_name}, zarr_path) - No such file: '{entry.zarr_path}'")
            invalid_entry = True
        if not entry.test_parquet_path.exists():
            print(f"[ERROR] {name} ({entry.env_name}, test_parquet_path) - No such file: '{entry.zarr_path}'")
            invalid_entry = True
        if not entry.test_zarr_path.exists():
            print(f"[WARN ] {name} ({entry.env_name}, test_zarr_path) - No such file: '{entry.zarr_path}'")

        if invalid_entry:
            print(f"{name} ({entry.env_name}) - INVALID")
        else:
            print(f"{name} ({entry.env_name}) - OK")

        is_valid &= not invalid_entry
    return is_valid


def get_data_catalog(
        zarr_root=DEFAULT_ZARR_ROOT,
        zarr_teleop_root=DEFAULT_ZARR_TELEOP_ROOT,
        test_zarr_root=DEFAULT_TEST_ZARR_ROOT,
        test_parquet_root=DEFAULT_TEST_PARQUET_ROOT,
        invalid_ok=True,
):
    zarr_root = Path(zarr_root)
    zarr_teleop_root = Path(zarr_teleop_root)
    test_zarr_root = Path(test_zarr_root)
    test_parquet_root = Path(test_parquet_root)

    catalog = {
        "feeding_teleop_250": DataConfig(
            env_name="Feeding",
            size=250,
            is_teleop=True,
            zarr_path=zarr_teleop_root / "FeedingJaco-v1.zarr",
            test_zarr_path=test_zarr_root / "feeding_250" / "zarr_recording" / "cat.zarr",
            test_parquet_path=test_parquet_root / "feeding_250_test.parquet",
        ),
        "feeding_teleop_100": DataConfig(
            env_name="Feeding",
            size=100,
            is_teleop=True,
            zarr_path=zarr_teleop_root / "FeedingJaco-v1_100.zarr",
            test_zarr_path=test_zarr_root / "feeding_100" / "zarr_recording" / "cat.zarr",
            test_parquet_path=test_parquet_root / "feeding_100_test.parquet",
        ),
        "feeding_teleop_50": DataConfig(
            env_name="Feeding",
            size=50,
            is_teleop=True,
            zarr_path=zarr_teleop_root / "FeedingJaco-v1_50.zarr",
            test_zarr_path=test_zarr_root / "feeding_50" / "zarr_recording" / "cat.zarr",
            test_parquet_path=test_parquet_root / "feeding_50_test.parquet",
        ),

        "arm_manipulation_ppo": DataConfig(
            env_name="ArmManipulation",
            size=None,
            is_teleop=False,
            zarr_path=zarr_root / "ArmManipulation_ppo.zarr",
            test_zarr_path=test_zarr_root / "arm_manipulation" / "zarr_recording" / "cat.zarr",
            test_parquet_path=test_parquet_root / "arm_manipulation_test.parquet",
        ),
        "bed_bathing_ppo": DataConfig(
            env_name="BedBathing",
            size=None,
            is_teleop=False,
            zarr_path=zarr_root / "BedBathing_ppo.zarr",
            test_zarr_path=test_zarr_root / "bed_bathing" / "zarr_recording" / "cat.zarr",
            test_parquet_path=test_parquet_root / "bed_bathing_test.parquet",
        ),
        "drinking_ppo": DataConfig(
            env_name="Drinking",
            size=None,
            is_teleop=False,
            zarr_path=zarr_root / "Drinking_1000k_ppo.zarr",
            test_zarr_path=test_zarr_root / "drinking" / "zarr_recording" / "cat.zarr",
            test_parquet_path=test_parquet_root / "drinking_test.parquet",
        ),
        "scratch_itch_ppo": DataConfig(
            env_name="ScratchItch",
            size=None,
            is_teleop=False,
            zarr_path=zarr_root / "ScratchItch_600k_ppo.zarr",
            test_zarr_path=test_zarr_root / "scratch_itch" / "zarr_recording" / "cat.zarr",
            test_parquet_path=test_parquet_root / "scratch_itch_test.parquet",
        ),
    }

    is_valid = validate_catalog(catalog)
    if not (invalid_ok or is_valid):
        raise RuntimeError("Catalog is invalid.")

    return catalog


def make_experiment_name(script_path):
    return Path(script_path).stem