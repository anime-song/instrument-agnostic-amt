import os
import argparse
import torch


def distill_model(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print(f"Loading {input_path}...")
    checkpoint = torch.load(input_path, map_location="cpu")

    # 抽出するデータの選別
    # EMAの重みがあればそれを優先し、なければ通常の重みを使用する
    model_state_dict = checkpoint.get(
        "ema_state_dict", checkpoint.get("model_state_dict")
    )

    if model_state_dict is None:
        # もし辞書形式でなく直接state_dictが保存されている場合
        model_state_dict = checkpoint
        print(
            "Warning: Could not find 'model_state_dict' or 'ema_state_dict'. Using the whole checkpoint as state_dict."
        )

    new_checkpoint = {
        "model_state_dict": model_state_dict,
        "model_config": checkpoint.get("model_config"),
        "config": checkpoint.get("config"),
    }

    print(f"Saving to {output_path}...")
    torch.save(new_checkpoint, output_path)

    # サイズの比較
    old_size = os.path.getsize(input_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done!")
    print(f"Original size: {old_size:.2f} MB")
    print(f"Distilled size: {new_size:.2f} MB")
    print(f"Keys in new checkpoint: {list(new_checkpoint.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distill PyTorch model for distribution"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="checkpoints/bass_model.pth",
        help="Path to input checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/best_model_bass.pth",
        help="Path to output checkpoint",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    distill_model(args.input, args.output)
