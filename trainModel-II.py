import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torch
import torch.nn as nn
import torch.distributed as dist
from models import *
from datasets import *
import platform
from metrics import*
import copy
import numpy as np
from torch.utils.data import Dataset, IterableDataset, DataLoader
from milesial_unet_model import UNet, APAU_Net
from leejunhyun_unet_models import U_Net, R2U_Net, AttU_Net, R2AttU_Net, AttU_Net_S
import pickle

# Import custom loss and evaluation functions


TRAIN = 'train'
VAL = 'validation'
MASTER_RANK = 0
SAVE_INTERVAL = 1

# Make sure to change these paths!
DATASET_PATH = 'data/next-day-wildfire-spread'
SAVE_MODEL_PATH = 'savedModels'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--master', default='sardine',
                        help='master node')
    parser.add_argument('-p', '--port', default='30437',
                         help = 'master node')
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=10, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    print(f'initializing training on single GPU')
    train(0, args)

def create_data_loaders(rank, gpu, world_size, selected_features=None):
    batch_size = 64

    ALL_FEATURES = [
               'elevation','pdsi', 'pr', 'sph', 'th', 'tmmn','tmmx', 'vs', 
        'erc', 'population', 'NDVI','PrevFireMask'
    ]

    if selected_features is not None:
        feature_indices = [ALL_FEATURES.index(feature) for feature in selected_features]
        print(f"Using selected features: {selected_features}")
        print(f"Feature indices being used: {feature_indices}")
    else:
        feature_indices = list(range(len(ALL_FEATURES)))
        selected_features = ALL_FEATURES
        print("Using all features by default.")

    print(f"\nSelected features and their indices:\n{list(zip(selected_features, feature_indices))}")

    datasets = {
        TRAIN: RotatedWildfireDataset(
            f"{DATASET_PATH}/{TRAIN}.data",
            f"{DATASET_PATH}/{TRAIN}.labels",
            features=feature_indices,
            crop_size=64
        ),
        VAL: WildfireDataset(
            f"{DATASET_PATH}/{VAL}.data",
            f"{DATASET_PATH}/{VAL}.labels",
            features=feature_indices,
            crop_size=64
        )
    }

    dataLoaders = {
        TRAIN: torch.utils.data.DataLoader(
            dataset=datasets[TRAIN],
            batch_size=batch_size,
            shuffle=True,  
            num_workers=0,
            pin_memory=True,
        ),
        VAL: torch.utils.data.DataLoader(
            dataset=datasets[VAL],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    }

    return dataLoaders

def perform_validation(model, loader):
    model.eval()

    total_loss = 0
    total_iou = 0
    total_accuracy = 0
    total_f1 = 0
    total_auc = 0
    total_dice = 0
    total_precision = 0
    total_recall = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)


             # Forward pass
            outputs = model(images)

            labels = torch.flatten(labels)
            outputs = torch.flatten(outputs)

            
            # Compute metrics
            total_loss += loss(labels, outputs).item()
            total_iou += mean_iou(labels, outputs)
            total_accuracy += accuracy(labels, outputs)
            total_f1 += f1_score(labels, outputs)
            total_auc += auc_score(labels, outputs)
            total_dice += dice_score(labels, outputs)
            
            precision, recall = precision_recall(labels, outputs)
            total_precision += precision
            total_recall += recall

    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    avg_accuracy = total_accuracy / len(loader)
    avg_f1 = total_f1 / len(loader)
    avg_auc = total_auc / len(loader)
    avg_dice = total_dice / len(loader)
    avg_precision = total_precision / len(loader)
    avg_recall = total_recall / len(loader)

    print(f"Validation - Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, Accuracy: {avg_accuracy:.4f}")
    print(f"F1 Score: {avg_f1:.4f}, AUC: {avg_auc:.4f}, Dice: {avg_dice:.4f}")
    print(f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")

    return avg_loss, avg_iou, avg_accuracy, avg_f1, avg_auc, avg_dice, avg_precision, avg_recall


def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    validate = True
    print("Current GPU", gpu, "\n RANK: ", rank)

    dataLoaders = create_data_loaders(rank, gpu, args.gpus * args.nodes)

    torch.manual_seed(0)

    model = U_Net(12, 1)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)


    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([5])).cuda(gpu)
  

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.003, momentum=0.9)

    start = datetime.now()
    print(f'TRAINING ON: {platform.node()}, Starting at: {datetime.now()}')

    total_step = len(dataLoaders[TRAIN])
    best_epoch = 0
    best_f1_score = -float("inf")

    train_loss_history = []
    val_metrics_history = []

    for epoch in range(args.epochs):
        model.train()

        loss_train = 0

        for i, (images, labels) in enumerate(dataLoaders[TRAIN]):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Forward pass
            outputs = model(images)


            # Not entirely sure if this flattening is required or not
            labels = torch.flatten(labels)
            outputs = torch.flatten(outputs)

            loss = criterion(outputs, labels)
            #loss = torchvision.ops.sigmoid_focal_loss(outputs, labels, alpha=0.85, gamma=2, reduction="mean")

            loss_train += loss.item()


            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                print('Epoch [{}/{}], Steps [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1,
                    args.epochs,
                    i,
                    total_step,
                    loss.item())
                )

        train_loss_history.append(loss_train / len(dataLoaders[TRAIN]))

        if validate:
            metrics = perform_validation(model, dataLoaders[VAL])
            val_metrics_history.append(metrics)

            curr_avg_loss_val, _, _, curr_f1_score, _, _, _, _ = metrics

            if best_f1_score < curr_f1_score:
                print("Saving model...")
                best_epoch = epoch
                best_f1_score = curr_f1_score
                filename = f'model-{model.__class__.__name__}-bestF1Score-Rank-{rank}.weights'
                torch.save(model.state_dict(), f'{SAVE_MODEL_PATH}/{filename}')
                print("Model has been saved!")
            else:
                print("Model is not being saved")

    pickle.dump(train_loss_history, open(f"{SAVE_MODEL_PATH}/train_loss_history.pkl", "wb"))
    pickle.dump(val_metrics_history, open(f"{SAVE_MODEL_PATH}/val_metrics_history.pkl", "wb"))

    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
        print(f"Endtime: {datetime.now()}")
        print(f"Best epoch: {best_epoch}")
        print(f"Best F1 score: {best_f1_score}")


if __name__ == '__main__':
    main()
