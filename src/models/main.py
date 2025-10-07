import torch
import torch.nn as nn
import torch.optim as optim
from src.models.nnet import SimpleNN
import hydra
from omegaconf import DictConfig
from src.preprocessors.factory import PreprocessorFactory


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    preprocessor = PreprocessorFactory.get_preprocessor(**cfg.preprocessor)  # model_type=cfg.preprocessor.model_type

    print(preprocessor)

    data = preprocessor.preprocess()

    print(f"Data prepared for model type: {cfg.preprocessor.model_type}")

    if cfg.preprocessor.model_type.lower() in ['nn', 'neural', 'neuralnet']:
        train_loader, val_loader, artifacts = data
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # -----------------------------
        # Model
        # -----------------------------

        print(artifacts)

        print(artifacts['scaler'].mean_)

        # classes
        print("Classes:", artifacts['encoder'].classes_)
        #print("Encoded values:", y_encoded[:10])

        print(artifacts['features'])

        input_size = preprocessor.input_size
        num_classes = preprocessor.num_classes

        print(input_size)
        print(num_classes)


        model = SimpleNN(input_size, num_classes)

        # -----------------------------
        # Loss and optimizer
        # -----------------------------
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # -----------------------------
        # Training loop
        # -----------------------------
        num_epochs = 2

        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
            val_acc = correct / total
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_acc:.4f}")

    else:
        X_train, X_val, y_train, y_val = data
        print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")


if __name__==main():
    main()