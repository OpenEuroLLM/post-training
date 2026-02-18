import argparse

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        type=str,
        default="list-configs-and-splits",
        choices=["list-configs-and-splits"],
        help="Which utility to run",
    )
    parser.add_argument("--dataset-name", type=str, required=True)
    return parser.parse_args()


def get_configs_and_splits(dataset_name: str) -> dict[str, list[tuple[str, int]]]:
    configs = get_dataset_config_names(dataset_name)
    return {
        config: [
            (split, len(load_dataset(dataset_name, config, split=split)))
            for split in get_dataset_split_names(dataset_name, config_name=config)
        ]
        for config in configs
    }


def list_configs_and_splits(dataset_name):
    print(f"--- Dataset: {dataset_name} ---")
    configs_and_splits = get_configs_and_splits(dataset_name)
    for config, splits in configs_and_splits.items():
        print(f"Subset: {config}")
        for split, num_samples in splits:
            print(f"  └─ Split: {split} (samples: {num_samples})")


if __name__ == "__main__":
    args = parse_args()
    if args.action == "list-configs-and-splits":
        list_configs_and_splits(args.dataset_name)
    else:
        raise ValueError(f"Unknown action: {args.action}")
