import pandas as pd
from models import get_training_settings
from ml import accuracy, fit, evaluate
import sys

try:
    model = sys.argv[1]
except:
    model = "simple"
settings = get_training_settings(model)

loss, _, acc = evaluate(settings.model, settings.loss, settings.test_dl, accuracy)

train_losses, val_losses, val_metrics = fit(
    settings.steps,
    settings.model,
    settings.loss,
    settings.train_dl,
    settings.test_dl,
    settings.optimizer,
    settings.lr,
    accuracy,
    settings.target_accuracy,
)

df = pd.DataFrame(
    {
        "val_loss": [loss] + val_losses,
        "val_acc": [acc] + val_metrics,
    }
)
df.to_csv("local_training.csv")
