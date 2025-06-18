import argparse
import torch
from torch.utils.data import DataLoader

from datasets import load_paraphrase_data, ParaphraseDetectionDataset
from evaluation import model_eval_paraphrase
from scripts.run_knn_augmented import load_knn_model, load_datastore
from models.knn_paraphrase import KNNParaphraseClassifier


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate k-NN-augmented Paraphrase model on the dev set")

    # model / training-specific
    parser.add_argument("--epochs", type=int, default=10, help="Finetuning epochs used during training")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning-rate tag used in checkpoint name")

    # k-NN parameters
    parser.add_argument("--k", type=int, default=8, help="Number of neighbours to retrieve")
    parser.add_argument("--lambda_knn", type=float, default=0.25, help="Interpolation weight for k-NN logits")
    parser.add_argument("--knn_temperature", type=float, default=10.0, help="Temperature for distance → similarity")
    parser.add_argument("--use_quality_filter", action="store_true", help="Enable neighbour quality filtering")
    parser.add_argument("--use_adaptive_interpolation", action="store_true", help="Enable entropy-based lambda")

    # runtime / misc
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_gpu", action="store_true")

    # datastore-related (for compatibility with load_datastore helper)
    parser.add_argument("--use_wikitext", action="store_true", help="Use WikiText datastore instead of task-specific one")
    parser.add_argument("--wikitext_version", type=str, default="2", choices=["2", "103"], help="WikiText version if --use_wikitext")
    parser.add_argument("--data_dir", type=str, default="data", help="Base data directory (not used here but required by helper)")
    parser.add_argument("--max_chunks", type=int, default=3, help="Max datastore chunks to load (memory safe)")

    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    # 1) Load k-NN-augmented GPT and datastore
    knn_gpt, _ = load_knn_model("paraphrase", args)
    datastore = load_datastore("paraphrase", args)
    knn_gpt.set_datastore(datastore)

    # 2) Wrap for evaluation compatibility
    model = KNNParaphraseClassifier(knn_gpt).to(device)

    # 3) Prepare dev data
    dev_raw = load_paraphrase_data("data/quora-dev.csv")
    dev_ds = ParaphraseDetectionDataset(dev_raw, args)
    dev_dl = DataLoader(dev_ds, shuffle=False, batch_size=args.batch_size, collate_fn=dev_ds.collate_fn)

    # 4) Evaluate
    acc, f1, *_ = model_eval_paraphrase(dev_dl, model, device)
    print(f"Dev accuracy: {acc:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    main() 