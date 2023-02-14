# the train funcs for training the model
import torch
from pathlib import Path



def train_epoch(
    flow,
    train_loader,
    optimizer,
    epoch,
    device=None,
    output_freq=50,
    args=None,
    add_noise=True,
    annealing=False,
):
    """Train model for one epoch.
    Arguments:
        flow {Flow} -- NSF model
        train_loader {DataLoader} -- train set data loader
        optimizer {Optimizer} -- model optimizer
        epoch {int} -- epoch number
    Keyword Arguments:
        device {torch.device} -- model device (CPU or GPU) (default: {None})
        output_freq {int} -- frequency for printing status (default: {50})
    Returns:
        float -- average train loss over epoch
    """

    flow.train()
    train_loss = 0.0

    for batch_idx, (z, y) in enumerate(train_loader):
        optimizer.zero_grad()

        if device is not None:
            z = z.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        # Compute log prob
        loss = -flow.log_prob(z, context=y)

        # Keep track of total loss. w is a weight to be applied to each
        # element.
        train_loss += (loss.detach()).sum()

        # loss = (w * loss).sum() / w.sum()
        loss = (loss).mean()

        loss.backward()
        optimizer.step()

        if (output_freq is not None) and (batch_idx % output_freq == 0):
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}".format(
                    epoch,
                    batch_idx * train_loader.batch_size,
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    train_loss = train_loss.item() / len(train_loader.dataset)
    print(
        "Model:{} Train Epoch: {} \tAverage Loss: {:.4f}".format(
            args.log_name, epoch, train_loss
        )
    )

    return train_loss


def test_epoch(flow, test_loader, epoch, device=None):
    """Calculate test loss for one epoch.
    Arguments:
        flow {Flow} -- NSF model
        test_loader {DataLoader} -- test set data loader
    Keyword Arguments:
        device {torch.device} -- model device (CPU or GPu) (default: {None})
    Returns:
        float -- test loss
    """

    with torch.no_grad():
        flow.eval()
        test_loss = 0.0
        for z, y in test_loader:

            if device is not None:
                z = z.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

            # Compute log prob
            loss = -flow.log_prob(z, context=y)

            # Keep track of total loss
            test_loss += (loss).sum()

        test_loss = test_loss.item() / len(test_loader.dataset)
        # test_loss = test_loss.item() / total_weight.item()
        print("Test set: Average loss: {:.4f}\n".format(test_loss))

        return test_loss

# NOTE: for the moment sampling is left to the user
# def sample(flow, test_loader, device, args):
#     """Sample from the model.
#     Arguments:
#         flow {Flow} -- NSF model
#         device {torch.device} -- model device (CPU or GPU)
#         num_samples {int} -- number of samples to generate
#     Returns:
#         torch.Tensor -- samples
#     """

#     with torch.no_grad():
#         flow.eval()
#         samples = []
#         conditioning = []
#         original = []
#         for _, (z, y) in enumerate(test_loader):
#             inputs_y = y.to(device)
#             z_sampled = flow.sample(
#                     num_samples=1, context=inputs_y.view(-1, args.y_dim)
#                 )

#             z_sampled = z_sampled.cpu().detach().numpy()
#             inputs_y = inputs_y.cpu().detach().numpy()
#             z = z.cpu().detach().numpy()

#             samples.append(z_sampled)
#             conditioning.append(inputs_y)
#             original.append(z)
        
#         samples = np.reshape(samples, (-1, args.z_dim))
#         conditioning = np.reshape(conditioning, (-1, args.y_dim))
#         original = np.reshape(original, (-1, args.z_dim))

#     return samples, conditioning, original

def train(
    model,
    train_loader,
    test_loader,
    epochs,
    optimizer,
    device,
    name,
    model_dir,
    args,
    writer=None,
    output_freq=100,
    save_freq=10,
    res_epoch=0,
    val_func=None,
    val_func_args=None,
):
    """Train the model.
    Args:
            epochs:     number of epochs to train for
            output_freq:    how many iterations between outputs
    """
    train_history = []
    test_history = []

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    for epoch in range(0 + res_epoch, epochs + 1 + res_epoch):

        print(
            "Learning rate: {}".format(optimizer.state_dict()["param_groups"][0]["lr"])
        )

        train_loss = train_epoch(
            model, train_loader, optimizer, epoch, device, output_freq, args=args
        )
        test_loss = test_epoch(model, test_loader, epoch, device)

        scheduler.step()
        train_history.append(train_loss)
        test_history.append(test_loss)

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("test/loss", test_loss, epoch)

        if epoch % args.val_freq == 0:
            if val_func is not None:
                val_func(**val_func_args)

        if epoch % save_freq == 0:

            save_model(
                epoch,
                model,
                scheduler,
                train_history,
                test_history,
                name,
                model_dir=model_dir,
                optimizer=optimizer,
            )
            print("saving model")

    return train_history, test_history


def save_model(
    epoch,
    model,
    scheduler,
    train_history,
    test_history,
    name,
    model_dir=None,
    optimizer=None,
):
    """Save a model and optimizer to file.
    Args:
        model:      model to be saved
        optimizer:  optimizer to be saved
        epoch:      current epoch number
        model_dir:  directory to save the model in
        filename:   filename for saved model
    """

    if model_dir is None:
        raise NameError("Model directory must be specified.")

    filename = name + f"_@epoch_{epoch}.pt"
    resume_filename = "checkpoint-latest.pt"

    p = Path(model_dir)
    p.mkdir(parents=True, exist_ok=True)

    dict = {
        "train_history": train_history,
        "test_history": test_history,
        "model_hyperparams": model.model_hyperparams,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        dict["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(dict, p / filename)
    torch.save(dict, p / resume_filename)


def load_model(device, model_dir=None, filename=None):
    """Load a saved model.
    Args:
        filename:       File name
    """

    if model_dir is None:
        raise NameError(
            "Model directory must be specified."
            " Store in attribute PosteriorModel.model_dir"
        )

    p = Path(model_dir)
    checkpoint = torch.load(p / filename, map_location=device)

    model_hyperparams = checkpoint["model_hyperparams"]
    train_history = checkpoint["train_history"]
    test_history = checkpoint["test_history"]

    # Load model
    model = create_NDE_model(**model_hyperparams)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
    # Failing to do this will yield inconsistent inference results.
    model.eval()

    # Load optimizer
    scheduler_present_in_checkpoint = "scheduler_state_dict" in checkpoint.keys()

    # If the optimizer has more than 1 param_group, then we built it with
    # flow_lr different from lr
    if len(checkpoint["optimizer_state_dict"]["param_groups"]) > 1:
        flow_lr = checkpoint["optimizer_state_dict"]["param_groups"][-1]["initial_lr"]
    else:
        flow_lr = None

    # Set the epoch to the correct value. This is needed to resume
    # training.
    epoch = checkpoint["epoch"]

    return (
        model,
        scheduler_present_in_checkpoint,
        flow_lr,
        epoch,
        train_history,
        test_history,
    )


