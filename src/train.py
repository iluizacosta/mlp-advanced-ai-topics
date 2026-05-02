import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import wandb

from src.model import MLP


def train_model(config, train_loader, test_loader, input_dim):
    """
    Train MLP model with W&B logging, early stopping, and loss tracking.
    """

    run_name = (
        f"mlp_{input_dim}feat_"
        f"{'-'.join(map(str, config['model']['hidden_sizes']))}_"
        f"lr{config['model']['learning_rate']}"
    )

    wandb.init(
        project="mlp-fetal-health",
        name=run_name,
        config=config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(
        input_dim=input_dim,
        hidden_sizes=config["model"]["hidden_sizes"],
        output_dim=3,
        dropout=config["model"]["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["model"]["learning_rate"]
    )

    best_val_loss = float("inf")
    patience = config["model"]["early_stopping_patience"]
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(config["model"]["epochs"]):

        # Training phase
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)

                preds = torch.argmax(outputs, dim=1)

                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_loss /= len(test_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": acc,
            "f1_score": f1
        })

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"F1: {f1:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(model.state_dict(), "best_model.pt")

        else:
            patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load("best_model.pt"))

    # Log final best model as artifact
    artifact = wandb.Artifact(
        "best-model",
        type="model"
    )
    artifact.add_file("best_model.pt")
    wandb.log_artifact(artifact)

    wandb.finish()

    return model, train_losses, val_losses