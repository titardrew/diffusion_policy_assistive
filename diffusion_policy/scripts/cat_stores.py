import random
import shutil

from collections import defaultdict
from pathlib import Path
from typing import List

import zarr

def cat_zarr_stores(input_files: List[Path], output_file: Path):
    assert output_file.suffix == ".zarr"
    GROUPS = ["data", "meta"]

    inputs = []
    topics = dict()
    shapes = defaultdict(lambda: defaultdict(list))

    if output_file.exists():
        print(f"Removing existing {output_file}!")
        shutil.rmtree(str(output_file))
        output_file.parent.mkdir(exist_ok=True, parents=True)

    for input_file in input_files:
        assert input_file.suffix == ".zarr"

        store = zarr.open(input_file, mode="r")
        inputs.append(store)

        for group in GROUPS:
            if topics.get(group) is None:
                topics[group] = tuple(store[group].keys())
            else:
                assert topics[group] == tuple(store[group].keys()), (input_file, topics, store[group].keys())

            for topic in topics[group]:
                shapes[group][topic].append(store[group][topic].shape)

    dest = zarr.open(output_file, mode="w")
    zarr.copy_all(inputs[0], dest)

    for group in GROUPS:
        for topic in topics[group]:
            total_len = sum(shape[0] for shape in shapes[group][topic])
            dest[group][topic].resize(total_len, *shapes[group][topic][0][1:])
            head = 0
            if topic == 'episode_ends':
                shift = 0
                for i, store in enumerate(inputs):
                    store_len = shapes[group][topic][i][0]
                    dest[group][topic][head:head+store_len] = store[group][topic][:] + shift
                    head += store_len
                    shift += store[group][topic][-1]
            else:
                for i, store in enumerate(inputs):
                    store_len = shapes[group][topic][i][0]
                    dest[group][topic][head:head+store_len] = store[group][topic][:]
                    head += store_len

    print(f"Resulting {str(output_file)} storage spec:")
    print(dest.tree())
     

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, nargs="+")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--max_stores", type=int, required=False, default=None)
    parser.add_argument("--shuffle_stores", action="store_true", required=False)
    args = parser.parse_args()
    input_files = list(map(Path, args.input))
    for input_file in input_files:
        if not input_file.exists():
            raise FileNotFoundError(str(input_file))
    
    if args.max_stores:
        input_files = input_files[:args.max_stores]

    if args.shuffle_stores:
        random.shuffle(input_files)

    output_file = Path(args.output)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    cat_zarr_stores(input_files=input_files, output_file=output_file)


