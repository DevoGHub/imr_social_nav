import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.dataset import TrajectoryDataset
from src.model.lstm import TrajectoryLSTM


def ade(pred, gt):
    return torch.mean(torch.norm(pred - gt, dim=-1))


def fde(pred, gt):
    return torch.mean(torch.norm(pred[:, -1] - gt[:, -1], dim=-1))


def evaluate(model, loader, device):
    model.eval()

    total_ade = 0
    total_fde = 0
    count = 0

    with torch.no_grad():
        for obs, fut in loader:
            obs = obs.to(device)
            fut = fut.to(device)

            pred = model(obs)

            # ADE: mean over time and batch
            ade_batch = torch.mean(torch.norm(pred - fut, dim=-1))

            # FDE: mean over batch
            fde_batch = torch.mean(torch.norm(pred[:, -1] - fut[:, -1], dim=-1))

            total_ade += ade_batch.item() * obs.size(0)
            total_fde += fde_batch.item() * obs.size(0)
            count += obs.size(0)

    return total_ade / count, total_fde / count

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # datasets
    train_dataset = TrajectoryDataset(
        "data/processed/bookstore_video0_traj.pkl",
        "data/splits/bookstore_video0/train.txt"
    )

    test_dataset = TrajectoryDataset(
        "data/processed/bookstore_video0_traj.pkl",
        "data/splits/bookstore_video0/test.txt"
    )

    video1_dataset = TrajectoryDataset(
        "data/processed/bookstore_video1_traj.pkl"
    )

    coupa_dataset = TrajectoryDataset(
        "data/processed/coupa_video0_traj.pkl"
    )

    # loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    video1_loader = DataLoader(video1_dataset, batch_size=64, shuffle=False)
    coupa_loader = DataLoader(coupa_dataset, batch_size=64, shuffle=False)

    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Video1 size: {len(video1_dataset)}")
    print(f"Coupa size: {len(coupa_dataset)}")

    # model
    model = TrajectoryLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for obs, fut in train_loader:
            obs = obs.to(device)
            fut = fut.to(device)

            pred = model(obs)
            loss = criterion(pred, fut)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # evaluation
        test_ade, test_fde = evaluate(model, test_loader, device)
        v1_ade, v1_fde = evaluate(model, video1_loader, device)
        coupa_ade, coupa_fde = evaluate(model, coupa_loader, device)

        print(f"""
            Epoch {epoch+1}/{epochs}
            Train Loss: {avg_loss:.4f}

            Test (video0 split): ADE {test_ade:.3f}, FDE {test_fde:.3f}
            Video1 (same scene): ADE {v1_ade:.3f}, FDE {v1_fde:.3f}
            Coupa (new scene): ADE {coupa_ade:.3f}, FDE {coupa_fde:.3f}
            """)

    torch.save(model.state_dict(), "models/trajectory_lstm.pth")
    print("Model saved as trajectory_lstm.pth")


if __name__ == "__main__":
    train()