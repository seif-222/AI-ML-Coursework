import torch
from tqdm.auto import tqdm

def train_step (model,
                dataloader_data,
                loss_fn,
                optimizer,
                device):
    model.train()
    train_loss , train_acc = 0,0
    for batch_idx, (data, label) in enumerate(dataloader_data):
        data, label = data.to(device), label.to(device)
        y_preds = model(data)
        loss = loss_fn(y_preds, label)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #calculate the accuracy
        y_pred_class = torch.argmax(y_preds, dim=1)
        train_acc += (y_pred_class == label).sum().item()/len(y_preds)

    train_loss = train_loss / len(dataloader_data)
    train_acc = train_acc / len(dataloader_data)
    return train_loss, train_acc



def test_step (model
               ,dataloader_data,
                loss_fn,
                device):
    model.eval()
    test_loss,test_acc = 0,0
    with torch.inference_mode():
        for data, label in dataloader_data:
            data, label = data.to(device), label.to(device)
            y_preds = model(data)
            loss = loss_fn(y_preds, label)
            test_loss += loss.item()
            y_pred_class = torch.argmax(y_preds, dim=1)
            test_acc += (y_pred_class == label).sum().item()/len(y_preds)
    test_loss = test_loss / len(dataloader_data)
    test_acc = test_acc / len(dataloader_data)
    return test_loss, test_acc


def training_fn (model,
                dataloader_train_data,
                 dataloader_test_data,
                loss_fn,
                optimizer,
                EPOCHS,
                device):
    """return results list"""
    results={"train_loss":[]
            ,"train_acc":[]
            ,"test_loss":[],
            "test_acc":[]}
    for epoch in tqdm(range(EPOCHS)):
        train_loss,train_acc = train_step (model, dataloader_train_data, loss_fn, optimizer, device)
        test_loss , test_acc = test_step (model, dataloader_test_data, loss_fn, device)
        print(f"Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Train acc: {train_acc*100:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc*100:.4f}")
        results["train_loss"].append(train_loss)  #remember that the results is dictionary of lists that is why we make append
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)


    return results
