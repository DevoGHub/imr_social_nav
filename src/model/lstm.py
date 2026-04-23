import torch
import torch.nn as nn


class TrajectoryLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1, pred_len=12):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len

        # Encoder: processes observed trajectory
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

        # Decoder: generates future trajectory
        self.decoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, obs):

        batch_size = obs.size(0)

        # Encode observed trajectory
        _, (h, c) = self.encoder(obs)

        # Start decoder with last observed position
        decoder_input = obs[:, -1:, :]  # (B, 1, 2)

        outputs = []

        for _ in range(self.pred_len):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            pred = self.fc(out)  # (B, 1, 2)

            outputs.append(pred)

            # autoregressive: feed prediction back in
            decoder_input = pred

        outputs = torch.cat(outputs, dim=1)  # (B, pred_len, 2)

        return outputs