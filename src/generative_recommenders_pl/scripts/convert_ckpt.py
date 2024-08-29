import argparse

import lightning as L
import torch


def convert_checkpoint(input_ckpt_path, output_ckpt_path):
    original_ckpt = torch.load(input_ckpt_path, map_location="cpu")

    # Create a new state dict for the converted checkpoint
    model_state_dict = {
        key[7:]: value for key, value in original_ckpt["model_state_dict"].items()
    }
    converted_state_dict = {}

    # Mapping of original keys to new keys
    key_mapping = {
        "_attn_mask": "sequence_encoder._attn_mask",
        "_embedding_module._item_emb.weight": "embeddings._item_emb.weight",
        "_input_features_preproc._pos_emb.weight": "preprocessor._pos_emb.weight",
    }

    # Iterate through the original state dict and convert keys
    for original_key in model_state_dict.keys():
        if original_key in key_mapping:
            new_key = key_mapping[original_key]
        else:
            if "_hstu._attention_layers." in original_key:
                new_key = original_key.replace(
                    "_hstu._attention_layers.",
                    "sequence_encoder._hstu._attention_layers.",
                )
            else:
                new_key = original_key

        converted_state_dict[new_key] = model_state_dict[original_key]

    # Create a new checkpoint dictionary
    new_ckpt = {
        "state_dict": converted_state_dict,
        "pytorch-lightning_version": L.__version__,
    }

    # Save the converted checkpoint
    torch.save(new_ckpt, output_ckpt_path)
    print(f"Converted checkpoint saved to {output_ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert checkpoint keys.")
    parser.add_argument(
        "--input_ckpt",
        type=str,
        required=True,
        help="Path to the input checkpoint file.",
    )
    parser.add_argument(
        "--output_ckpt",
        type=str,
        required=True,
        help="Path to save the converted checkpoint file.",
    )

    args = parser.parse_args()

    convert_checkpoint(args.input_ckpt, args.output_ckpt)
