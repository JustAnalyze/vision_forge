import torch
import torch.nn as nn
from tqdm.auto import tqdm


# train step function
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               optimizer: torch.optim.Optimizer,
               device):

  # Put the model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0.0, 0.0

  # Loop through data loader and data batches
  for batch, (X, y) in enumerate(dataloader):
    # Send data to target device
    X, y = X.to(device), y.to(device)

    # 1. Forward pass
    preds_logits = model(X)

    # 2. Calculate and accumulate loss
    loss = loss_fn(preds_logits, y)
    train_loss += loss

    # 3. label predictions
    preds_labels = torch.argmax(preds_logits, dim=1)

    # 3.1 Calculate and accumualte accuracy metric across all batches
    acc = accuracy_fn(preds_labels, y)
    train_acc += acc

    # 4. Optimizer zero grad
    optimizer.zero_grad()

    # 5. Loss backward
    loss.backward()

    # 6. Optimizer step
    optimizer.step()

  # Adjust metrics to get average loss and average accuracy per batch
  train_loss /= len(dataloader)
  train_acc /= len(dataloader)

  return train_loss, train_acc


# test step function
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device):

  # Put model in eval mode
  model.eval()

  # Setup the test loss and test accuracy values
  test_loss, test_acc = 0.0, 0.0

  # Loop through DataLoader batches
  for batch, (X, y) in enumerate(dataloader):
    # Send data to target device
    X, y = X.to(device), y.to(device)

    # 1. Forward pass
    # Turn on inference context manager
    with torch.inference_mode():
      preds_logits = model(X)

    # 2. Calculuate and accumulate loss
    loss = loss_fn(preds_logits, y)
    test_loss += loss

    # 3. label predictions
    preds_labels = torch.argmax(preds_logits, dim=1)

    # Calculate and accumulate accuracy
    acc = accuracy_fn(preds_labels, y)
    test_acc += acc

  # Adjust metrics to get average loss and accuracy per batch
  test_loss /= len(dataloader)
  test_acc /= len(dataloader)

  return test_loss, test_acc


# Train model for specified epoch using train step and evaluate using test step
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          accuracy_fn,
          device,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):

  """
  Trains the given model using the provided data loaders.

  Args:
      model (torch.nn.Module): The model to train.
      train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
      test_dataloader (torch.utils.data.DataLoader): DataLoader for testing/validation data.
      optimizer (torch.optim.Optimizer): Optimizer to use during training.
      accuracy_fn (callable): Function to compute accuracy.
      device: Device to use for training (e.g., 'cuda' for GPU, 'cpu' for CPU).
      loss_fn (torch.nn.Module, optional): Loss function. Default is nn.CrossEntropyLoss().
      epochs (int, optional): Number of epochs for training. Default is 5.

  Returns:
      dict: A dictionary containing training and testing metrics.
          Keys:
              - "train_loss": List of training losses for each epoch.
              - "train_acc": List of training accuracies for each epoch.
              - "test_loss": List of testing losses for each epoch.
              - "test_acc": List of testing accuracies for each epoch.
  """

  # Create results dictionary
  results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []}

  # Loop through the training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
    # Train step
    train_loss, train_acc = train_step(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       accuracy_fn=accuracy_fn,
                                       optimizer=optimizer,
                                       device=device)
    # Test step
    test_loss, test_acc = test_step(model=model,
                                    dataloader=test_dataloader,
                                    accuracy_fn=accuracy_fn,
                                    loss_fn=loss_fn,
                                    device=device)

    # Print out what's happening
    print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")

    # Update the results dictionary
    results["train_loss"].append(train_loss.detach() if device == 'cpu' else torch.Tensor.cpu(train_loss.detach()))
    results["train_acc"].append(train_acc if device == 'cpu' else torch.Tensor.cpu(train_acc))
    results["test_loss"].append(test_loss if device == 'cpu' else torch.Tensor.cpu(test_loss))
    results["test_acc"].append(test_acc if device == 'cpu' else torch.Tensor.cpu(test_acc))

  # Return the results dictionary
  return results
