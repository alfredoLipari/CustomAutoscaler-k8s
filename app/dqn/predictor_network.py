# predictor_network.py
import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )

        # Simpler architecture
        self.fc = nn.Linear(hidden_size, 1)
        
        # Initialize weights with slightly positive bias
        self.fc.bias.data.fill_(0.1)
    
    def forward(self, x):
        # Add residual connection
        last_value = x[:, -1, :]
        
        lstm_out, _ = self.lstm(x)
        prediction = self.fc(lstm_out[:, -1, :])
        
        # The prediction is now a difference from last value
        return last_value + prediction
