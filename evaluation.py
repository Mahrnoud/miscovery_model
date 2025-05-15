"""
Evaluation and Checkpoint Management Module for Transformer Models
"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import evaluate

from enhanced_evaluation_code import generate_text_optimized


class ModelEvaluator:
    """
    Evaluator class for Transformer models with metrics tracking and visualization
    """

    def __init__(
            self,
            model,
            tokenizer,
            eval_dataloader,
            device,
            output_dir="./outputs",
            max_checkpoints=2
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.output_dir = output_dir
        self.max_checkpoints = max_checkpoints

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

        # Initialize metrics
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")

        # Metrics history
        self.metrics_history = {
            "train_loss": [],
            "eval_loss": [],
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
            "bleu": [],
            "f1": []
        }

        # Checkpoint tracking
        self.checkpoints = []
        self.best_score = -float("inf")

    def evaluate(self, epoch, global_step, train_loss=None):
        """
        Evaluate the model and save metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        # Criterion for evaluation
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        print(f"\nEvaluating model at epoch {epoch}, step {global_step}...")

        with torch.no_grad():
            for batch_idx, (src, tgt) in enumerate(self.eval_dataloader):
                src, tgt = src.to(self.device), tgt.to(self.device)

                # Get input and target sequences
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                # Forward pass
                try:
                    output = self.model(src, tgt_input)

                    # Calculate loss
                    output_flat = output.contiguous().view(-1, output.size(-1))
                    tgt_output_flat = tgt_output.contiguous().view(-1)
                    loss = criterion(output_flat, tgt_output_flat)

                    total_loss += loss.item()

                    # Generate predictions for selected samples (every 10th batch)
                    if batch_idx % 10 == 0 and batch_idx < 50:
                        for i in range(min(2, len(src))):
                            generated_text = generate_text_optimized(
                                self.model,
                                src[i:i + 1],
                                self.tokenizer,
                                max_length=512,
                                device=self.device,
                                do_sample=True
                            )

                            reference = self.tokenizer.decode(
                                tgt[i].tolist(),
                                skip_special_tokens=True
                            )

                            all_preds.append(generated_text)
                            all_labels.append(reference)

                except RuntimeError as e:
                    print(f"Error during evaluation: {e}")
                    continue

        # Calculate average loss
        avg_loss = total_loss / len(self.eval_dataloader)

        # Calculate ROUGE scores
        try:
            rouge_output = self.rouge.compute(
                predictions=all_preds,
                references=all_labels
            )
        except Exception as e:
            print(f"ROUGE computation failed: {e}")
            # Provide default values if computation fails
            rouge_output = {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0
            }

        # Calculate BLEU score
        # For BLEU in the new evaluate library, references need to be a list of lists of strings
        # But predictions should be a list of strings
        tokenized_references = [[r.split()] for r in all_labels]
        tokenized_predictions = [p.split() for p in all_preds]

        try:
            # First try with the expected format from the error message
            bleu_score = self.bleu.compute(
                predictions=[' '.join(p) for p in tokenized_predictions],
                references=[[' '.join(r[0])] for r in tokenized_references]
            )['bleu']
        except Exception as e:
            print(f"First BLEU computation attempt failed, trying alternative format: {e}")
            try:
                # Fall back to simpler format
                bleu_score = self.bleu.compute(
                    predictions=all_preds,
                    references=[[r] for r in all_labels]
                )['bleu']
            except Exception as e:
                print(f"Second BLEU computation attempt failed: {e}")
                bleu_score = 0.0  # Default value if computation fails

        # Calculate F1 score (token level)
        # This is a simplified token-level F1 calculation
        f1_scores = []
        for pred, ref in zip(all_preds, all_labels):
            pred_tokens = set(pred.split())
            ref_tokens = set(ref.split())

            if not ref_tokens:
                continue

            if not pred_tokens:
                f1_scores.append(0)
                continue

            # Calculate precision, recall, and F1
            true_positives = len(pred_tokens.intersection(ref_tokens))
            precision = true_positives / len(pred_tokens) if pred_tokens else 0
            recall = true_positives / len(ref_tokens) if ref_tokens else 0

            if precision + recall == 0:
                f1_scores.append(0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))

        avg_f1 = np.mean(f1_scores) if f1_scores else 0

        # Compile metrics
        metrics = {
            "eval_loss": avg_loss,
            "rouge1": rouge_output["rouge1"],
            "rouge2": rouge_output["rouge2"],
            "rougeL": rouge_output["rougeL"],
            "bleu": bleu_score,
            "f1": avg_f1
        }

        # Update metrics history
        for key, value in metrics.items():
            self.metrics_history[key].append(value)

        if train_loss is not None:
            self.metrics_history["train_loss"].append(train_loss)

        # Print metrics
        print(f"Evaluation metrics at epoch {epoch}, step {global_step}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  ROUGE-1: {metrics['rouge1']:.4f}")
        print(f"  ROUGE-2: {metrics['rouge2']:.4f}")
        print(f"  ROUGE-L: {metrics['rougeL']:.4f}")
        print(f"  BLEU: {metrics['bleu']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")

        # Save metrics history
        with open(os.path.join(self.output_dir, "metrics_history.json"), "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        # Save checkpoint based on a combined score
        # Use a simpler formula that's less likely to fail
        combined_score = metrics["rougeL"] * 0.6 + metrics["bleu"] * 0.3 + metrics["f1"] * 0.1

        if combined_score > self.best_score:
            self.best_score = combined_score
            self.save_checkpoint(epoch, global_step, metrics, is_best=True)
        else:
            self.save_checkpoint(epoch, global_step, metrics, is_best=False)

        # Generate and save plots
        self.plot_metrics(epoch, global_step)

        self.model.train()
        return metrics

    def save_checkpoint(self, epoch, global_step, metrics, is_best=False):
        """
        Save model checkpoint and manage the number of saved checkpoints
        """
        # Create checkpoint directory
        checkpoints_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

        # Checkpoint filename
        checkpoint_name = f"checkpoint_epoch{epoch}_step{global_step}"
        checkpoint_path = os.path.join(checkpoints_dir, f"{checkpoint_name}.pth")

        # Save model state
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
        }

        torch.save(checkpoint, checkpoint_path)

        # Add to checkpoints list
        self.checkpoints.append({
            "path": checkpoint_path,
            "score": metrics["rougeL"],
            "is_best": is_best
        })

        print(f"Saved checkpoint to {checkpoint_path}")

        # Save as best model if applicable
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

        # Manage number of checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by score (descending)
            sorted_checkpoints = sorted(
                self.checkpoints,
                key=lambda x: (x["is_best"], x["score"]),
                reverse=True
            )

            # Keep best and N-1 next best checkpoints
            checkpoints_to_keep = sorted_checkpoints[:self.max_checkpoints]
            checkpoints_to_remove = sorted_checkpoints[self.max_checkpoints:]

            # Remove old checkpoints
            for ckpt in checkpoints_to_remove:
                if os.path.exists(ckpt["path"]) and not ckpt["is_best"]:
                    os.remove(ckpt["path"])
                    print(f"Removed old checkpoint: {ckpt['path']}")

            # Update checkpoints list
            self.checkpoints = checkpoints_to_keep

    def plot_metrics(self, epoch, global_step):
        """
        Plot training and evaluation metrics
        """
        figures_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        # Plot loss curves
        plt.figure(figsize=(10, 6))

        # Plot train and eval loss if available
        epochs_range = list(range(1, len(self.metrics_history["eval_loss"]) + 1))

        if self.metrics_history["train_loss"]:
            plt.plot(
                epochs_range,
                self.metrics_history["train_loss"],
                label="Training Loss",
                marker="o"
            )

        plt.plot(
            epochs_range,
            self.metrics_history["eval_loss"],
            label="Validation Loss",
            marker="x"
        )

        plt.title("Training and Validation Loss")
        plt.xlabel("Evaluation Checkpoints")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        # Save the loss plot
        loss_plot_path = os.path.join(figures_dir, f"loss_plot_epoch{epoch}.png")
        plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Plot ROUGE and BLEU scores
        plt.figure(figsize=(10, 6))

        plt.plot(
            epochs_range,
            self.metrics_history["rouge1"],
            label="ROUGE-1",
            marker="o"
        )
        plt.plot(
            epochs_range,
            self.metrics_history["rouge2"],
            label="ROUGE-2",
            marker="s"
        )
        plt.plot(
            epochs_range,
            self.metrics_history["rougeL"],
            label="ROUGE-L",
            marker="^"
        )
        plt.plot(
            epochs_range,
            self.metrics_history["bleu"],
            label="BLEU",
            marker="d"
        )
        plt.plot(
            epochs_range,
            self.metrics_history["f1"],
            label="F1",
            marker="*"
        )

        plt.title("Evaluation Metrics")
        plt.xlabel("Evaluation Checkpoints")
        plt.ylabel("Score")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        # Save the metrics plot
        metrics_plot_path = os.path.join(figures_dir, f"metrics_plot_epoch{epoch}.png")
        plt.savefig(metrics_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved plots to {figures_dir}")


def evaluate_translations(model, test_dataloader, tokenizer, device, num_examples=5):
    """
    Evaluate the model on translation examples and print results
    """
    model.eval()

    print("\n===== Translation Examples =====")

    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(test_dataloader):
            if batch_idx >= num_examples:
                break

            src, tgt = src.to(device), tgt.to(device)

            # Original source and target
            src_text = tokenizer.decode(src[0].tolist(), skip_special_tokens=True)
            tgt_text = tokenizer.decode(tgt[0].tolist(), skip_special_tokens=True)

            # Generate translation
            generated_text = generate_text_optimized(
                model,
                src[0:1],
                tokenizer,
                max_length=512,
                device=device,
                do_sample=False  # Use greedy decoding for consistency
            )

            print(f"\nExample {batch_idx + 1}:")
            print(f"Source: {src_text}")
            print(f"Target: {tgt_text}")
            print(f"Generated: {generated_text}")

    model.train()


def load_best_model(model, output_dir, device):
    """
    Load the best checkpoint based on saved metrics
    """
    best_model_path = os.path.join(output_dir, "best_model.pth")

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from {best_model_path}")
        print(f"Best model metrics: {checkpoint['metrics']}")
    else:
        print("No best model found.")

    return model
