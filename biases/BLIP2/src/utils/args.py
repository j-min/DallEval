import argparse
from argparse import ArgumentParser, Namespace

def parse_args() -> Namespace:
    parser: ArgumentParser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, default=f"")
    parser.add_argument("--metadata_file", type=str, default=f"")
    parser.add_argument("--savepath", type=str, default=f"./results/blip2_results.json")
    
    parser.add_argument("--load_output", type=str, default="none")
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--task", type=str, default="gender")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()
    print(args)

    return args
