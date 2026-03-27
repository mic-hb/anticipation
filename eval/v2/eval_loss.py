from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from train.v2.dataset_utils import PreTokenizedDataset
from train.v2.hf_gpt2_rewrite import GPT2LMHeadModelLite


def log_loss(
    model: GPT2LMHeadModelLite, dataset: PreTokenizedDataset, subsample_ratio: int = 10
):
    total_samples = len(dataset)
    ce = torch.empty(0)

    for sample_idx in tqdm(range(total_samples)):
        if sample_idx % subsample_ratio != 0:
            continue

        tokens = dataset[sample_idx]["input_ids"]
        tokens = tokens.unsqueeze(0).cuda()
        with torch.no_grad():
            logits = model(tokens).logits[0]
            ce = torch.cat(
                [
                    ce,
                    F.cross_entropy(logits[:-1], tokens[0, 1:], reduction="none").cpu(),
                ]
            )

    # with open(datafile, "r") as data:
    #     ce = torch.empty(0)
    #     for i, line in tqdm(list(enumerate(data))):
    #         if i % subsample != 0:
    #             continue

    #         tokens = [int(token) for token in line.split()]
    #         tokens = torch.tensor(tokens).unsqueeze(0).cuda()
    #         with torch.no_grad():
    #             logits = model(tokens).logits[0]
    #             ce = torch.cat(
    #                 [
    #                     ce,
    #                     F.cross_entropy(
    #                         logits[:-1], tokens[0, 1:], reduction="none"
    #                     ).cpu(),
    #                 ]
    #             )

    res = {}
    res["loss"] = np.round(ce.mean().item(), 3)
    res["event_ppl"] = np.round(np.exp(3 * ce.mean().item()), 3)
    res["onset_ppl"] = np.round(np.exp(ce[0::3].mean().item()), 3)
    res["dur_ppl"] = np.round(np.exp(ce[1::3].mean().item()), 3)
    res["note_ppl"] = np.round(np.exp((ce[2::3]).mean().item()), 3)
    # WARNING: hard-coded lakh dataset test split number of hours
    # test split is f
    res["bps"] = (
        subsample_ratio * ce.mean().item() * np.log2(np.e) * (len(ce) / (560.98 * 3600))
    )

    return res


def main() -> None:
    data_dir = "data/tokenized_datasets/giga_midi/6fb2094dfa7c0d16278dfaa4a401e3b8"
    data_dir_lmd = "data/tokenized_datasets/lmd_full/"
    pretrained_checkpoint_path = "/home/mf867/anticipation_isolated/anticipation/output/checkpoints/test_checkpoints/step-100"
    dataset = PreTokenizedDataset(Path(data_dir) / "train.npy")
    model = GPT2LMHeadModelLite.from_pretrained(
        pretrained_checkpoint_path,
    )
    res = log_loss(model, dataset)
    pprint(res)


if __name__ == "__main__":
    main()
