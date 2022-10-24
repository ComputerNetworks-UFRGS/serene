from itertools import cycle
import torch
import numpy as np

from constants import GRADS_PER_PKT
import torch


def loss_batch(model, loss_fn, xb, yb, opt=None, metric=None):
    """Compute loss over a batch.
    If opt is given, perform backpropagation.
    If metric is given, evaluate the metric over the given batch

    Return:
        loss (float): loss computed over the batch
        elements (int): number of elements in batch
        metric: Result of the metric. If the metric os not given, return None
    """

    preds = model(xb)
    loss = loss_fn(preds, yb)

    if opt is not None:
        # print("Backpropagation...")
        opt.zero_grad()
        loss.backward()
        opt.step()

    metric_result = None
    if metric is not None:
        metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result


def evaluate(model, loss_fn, valid_dl, metric=None):
    """Evaluate the model over a validation dataset

    Returns:
        avg_loss, total_elements, avg_metric
    """
    with torch.no_grad():
        results = [
            loss_batch(model, loss_fn, xb, yb, metric=metric) for xb, yb in valid_dl
        ]
        losses, nums, metrics = zip(*results)
        total = np.sum(nums)
        # média ponderada pelo número de imagens em cada batch
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
        return avg_loss, total, avg_metric


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)


def num_params(model):
    return sum(t.numel() for t in model.parameters())


def fit(
    steps,
    model,
    loss_fn,
    train_dl,
    valid_dl,
    opt_fn=None,
    lr=None,
    metric=None,
    target_metric=None,
):
    train_losses = []
    train_metrics = []
    val_losses = []
    val_metrics = []

    if opt_fn is not None:
        opt = opt_fn(model.parameters(), lr=lr)
        print(opt)
        scheduler = None#torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    else:
        scheduler = None

    train_iter = cycle(train_dl)

    for step in range(steps):
        model.train()

        xb, yb = next(train_iter)
        train_loss, _, train_metric = loss_batch(model, loss_fn, xb, yb, opt, accuracy)

        if (step + 1) % 500 == 0:
            model.eval()
            val_loss, _, val_metric = evaluate(model, loss_fn, valid_dl, metric)

            train_losses.append(train_loss)
            train_metrics.append(train_metric)
            val_losses.append(val_loss)
            val_metrics.append(val_metric)

            display_eval(
                step + 1, steps, train_loss, train_metric, val_loss, val_metric
            )
            if target_metric is not None and val_metric > target_metric:
                break

            #if scheduler:
            #    scheduler.step()
            #    for param_group in opt.param_groups:
            #        print(param_group["lr"])
            #        break
    return train_losses, val_losses, val_metrics


def display_eval(step, total_steps, train_loss, train_acc, val_loss, val_acc):
    print(
        f"Step [{step}/{total_steps}], train_loss: {train_loss}, train_acc: {train_acc}, "
        f"val_loss: {val_loss}, val_acc: {val_acc}"
    )


def get_parameter_shapes(model):
    return [tensor.shape for tensor in model.parameters()]


def flatten_and_split_tensors(tensors, split_size=GRADS_PER_PKT):
    flattened = [tensor.view(-1) for tensor in tensors]
    splitted = torch.split(torch.cat(flattened), split_size)
    return splitted


def rebuild_format(tensor, shapes):
    splitted = tensor.split([shape.numel() for shape in shapes])
    rebuilt = [tensor.view(shape) for tensor, shape in zip(splitted, shapes)]
    return rebuilt


def apply_gradients(model, gradients):
    with torch.no_grad():
        for param, grad in zip(model.parameters(), gradients):
            param -= grad
    return model
