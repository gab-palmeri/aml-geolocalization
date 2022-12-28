import torch
from torch.optim import CosineAnnealingLR

# inside training
model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# cosine annealing learning rate scheduler
# obiettivo ridurre tempo di training senza perdere performance
T_max = iterations_per_epoch * args.epochs
eta_min = 0.001
last_epoch = -1 or epoch_num
scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch)
# tempo esecuzione testato restato uguale

# efficientnetv2 
# obiettivo ridurre tempo di training senza perdere performance
efficientnet_v2_s # troppo lento!!!

# averaging ensemble
# Definire i modelli
model1 = nn.Linear(10, 20)
model2 = nn.Linear(10, 20)

# Creare una lista di modelli
model_list = nn.ModuleList([model1, model2])

# Creare un modulo di ensemble che utilizza la lista di modelli
class EnsembleModule(nn.Module):
    def __init__(self, models):
        super(EnsembleModule, self).__init__()
        self.models = models

    def forward(self, x):
        # Fare le previsioni utilizzando i modelli nella lista e combinarle
        # ad esempio, potresti utilizzare il voto della maggioranza
        predictions = [model(x) for model in self.models]
        return torch.mean(predictions, dim=0)

ensemble_model = EnsembleModule(model_list)

# Usare il modulo di ensemble come un normale modello PyTorch
output = ensemble_model(torch.randn(10, 10))
print(output.shape)  # dovrebbe stampare (20,)

# model soup
# obiettivo migliorare performance

# fine tuning ???
