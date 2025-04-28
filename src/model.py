import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=2, dropout_rate=0.2):
        super(MLPClassifier, self).__init__()
        
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        with torch.no_grad():
            logits = self(x)
            _, predicted = torch.max(logits, 1)
            return predicted
    
    def predict_proba(self, x):
        with torch.no_grad():
            logits = self(x)
            return F.softmax(logits, dim=1) 