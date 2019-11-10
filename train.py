import argparse
from util import get_loaders_with_class, construct_model, train, test, save_model_checkpoint
import torch

def argument_parser():
    ap = argparse.ArgumentParser("Image classfier train application")

    ap.add_argument("data_dir", type=str, help="directory to read data from, it must contain /test, /train and /valid")
    ap.add_argument("--save_dir", type=str,default='output', help="directory to save checkpoint")
    ap.add_argument("--learning_rate", type=float, default=0.002, help="model learning rate")
    ap.add_argument("--hidden_units", type=int, default=512 , help="model hidden units")
    ap.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    ap.add_argument("--arch", type=str, default='densenet121', help="model architecture")
    ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")

    return ap.parse_args()


if __name__ == "__main__":
    a = argument_parser()
    data_dir = a.data_dir
    check_point_dir = a.save_dir
    gpu = a.gpu
    arch = a.arch
    learning_rate = a.learning_rate
    hidden_units = a.hidden_units
    device = torch.device("cuda:0" if gpu else "cpu")
    epochs = a.epochs

    device_type = "GPU" if gpu else "CPU"
    print(f"Use {device_type} for training")

    train_loader, test_loader, validation_loader, class_to_idx = get_loaders_with_class(data_dir)

    model, criterion, optimizer = construct_model(arch=arch, hidden_units=hidden_units, device=device, learning_rate=learning_rate)

    print_every=50
    train(model, optimizer, criterion, train_loader, validation_loader, device, print_every,  epochs)

    test(model=model, criterion=criterion, test_loader=test_loader, device=device)

    save_model_checkpoint(model, class_to_idx, arch, hidden_units, learning_rate, optimizer)

