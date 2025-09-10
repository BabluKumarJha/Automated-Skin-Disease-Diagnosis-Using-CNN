# import important library and local script file
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from data import transform
from data import train_dir, test_dir
from model import cnn_model


# Dataloader
transform = transform(train_dir, test_dir)
train_dataloader = transform.train_dataloader()
test_dataloader = transform.test_dataloader()


##########
## Model
########
# model = cnn_model()

######################################
# Device Agnostic code
####################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Create train step()
def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer):
    # Put the model in train mode
    model.train()

    # Setup trian loss and train accuracy values
    train_loss, train_acc = 0,0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to the target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = cnn_model(X)

        # 2. Calcualte the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate accuracy metric
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class==y).sum().item()/len(y_pred)

    # Adjust metrics to get average lossand accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc



# Create a test step
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn = torch.nn.Module):
    # Put Mdoel in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0,0

    # Turn on inference mode
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batches, (X, y) in enumerate(dataloader):
            # Send data to the target device
            X,y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate the loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate the accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels ==y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss/ len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc





results = {}
# 1. Create a train function that takes in various model parameters + optimizer + dataloader + loss function
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int= 5,
          device = device):
    # Create empty results dictionary
    results = {"train_loss":[],
               "train_acc": [],
               "test_loss": [],
               'test_acc': []}

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model= model,
                                           dataloader = train_dataloader,
                                           loss_fn = loss_fn,
                                           optimizer = optimizer
                                           )
        test_loss, test_acc = test_step(
            model= model,
            dataloader = test_dataloader,
            loss_fn = loss_fn

        )
        # 4. Print out what's happening
        print(f'Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.2f}% | Test loss: {test_loss:.4f} | Test_acc: {test_acc:.2f}%')

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

# declare loss or criterion
criterion = nn.CrossEntropyLoss()
# optimizer variable
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
# we are taking Adam, it is usually known as better for image classification.



if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()  # Optional but safe

    model_results = train(
        model=cnn_model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=criterion,
        epochs=10
    )

