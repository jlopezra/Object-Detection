import typing as t

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchsummary import summary
from matplotlib import pyplot as plt
from torch import Tensor
from utils import DEVICE, synthesize_data, score_iou

from model import Detector_FPN

class StarDataset(torch.utils.data.Dataset):
    """Return star image and labels"""

    def __init__(self, data_size=512000):
        self.data_size = data_size

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx) -> t.Tuple[torch.Tensor, torch.Tensor]:
        # if has_star is set to True, training data only contains images with stars
        # image, label = synthesize_data(has_star=True)
        image, label = synthesize_data()
        return image[None], label

def rotationloss(p_star: Tensor, preds: Tensor, label: Tensor) -> Tensor:

    # Reference: Eqn(1) & Eqn (2) https://arxiv.org/pdf/1911.08299.pdf
    x1, y1, yaw1, w1, h1 = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3], preds[:, 4]
    x2, y2, yaw2, w2, h2 = label[:, 0], label[:, 1], label[:, 2], label[:, 3], label[:, 4]

    # Center Point Loss
    lcp = torch.abs(x1 - x2) + torch.abs(y1 - y2)

    # l1 Loss
    l1 = lcp + torch.abs(w1 - w2) + torch.abs(h1 - h2) + (torch.abs(yaw1 - yaw2)*180/math.pi)

    # correction loss to make loss continuous [eliminates angular periodicity
    # and the exchangeability of height and width]
    l1_corr = lcp + torch.abs(w1 - h2) + torch.abs(h1 - w2) + \
            torch.abs(90 - (torch.abs(yaw1 - yaw2)*180/math.pi))

    # Modulated Rotation Loss
    lmr = torch.min(l1, l1_corr)

    for i, loss_i in enumerate(lmr):
        if np.isnan(loss_i.cpu().detach().numpy()):
            pred_star = np.isnan(preds[i][0].cpu().detach().numpy())
            label_star = np.isnan(label[i][0].cpu().detach().numpy())
            # lmr_max = np.nanmax(lmr.cpu().detach().numpy())  # make lmr_max the max loss in lmr
            # lmr_max = np.float32(100.0)
            if pred_star and label_star:
                l = lmr.clone()
                l[i] = 0.0
                lmr = l.clone()
            elif pred_star:
                l = lmr.clone()
                l[i] = 200 #lmr_max.item()*.6
                lmr = l.clone()
            else:
                l = lmr.clone()
                l[i] = 500 #lmr_max.item()*1.2
                lmr = l.clone()

    label_stars = torch.empty(p_star.shape)
    for i, label_i in enumerate(label):
        if np.isnan(label_i.cpu().detach().numpy()).any():
            label_stars[i] = torch.tensor([0.0])
        else:
            label_stars[i] = torch.tensor([1.0])

    label_stars = label_stars.to(DEVICE)
    l_star = nn.functional.binary_cross_entropy(p_star, label_stars, reduction="none")
    loss = lmr + l_star

    return loss

def train(model: Detector_FPN, dl: StarDataset, num_epochs: int) -> Detector_FPN:

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epoch_log = []
    loss_log = []
    accuracy_log = []
    for epoch in range(num_epochs):
        losses = []
        p_stars = []
        i = 0 # used to keep track of iterations for minibatch cycle
        for image, label in tqdm(dl, total=len(dl)): # get data from dataloader
            image = image.to(DEVICE).float() # moves data to GPU
            label = label.to(DEVICE).float() # moves label to GPU
            i+=1
            # clear the gradients before training by setting to zero. Fresh start for each epoch
            optimizer.zero_grad()
            # Forward -> backprop + optimize
            p_star, preds = model(image) # Forward Propagation
            # loss = loss_fn(preds, label) # Get Loss (diff between label and predictions)
            loss = rotationloss(p_star, preds, label)
            loss = loss.mean()
            loss.backward() # Back propagate to obtain the new gradients for all nodes
            losses.append(loss.detach().cpu().numpy())
            optimizer.step() # Update the gradients/weights
            p_star = p_star.mean()
            p_stars.append(p_star.cpu().detach().numpy())

            # print training statistics - Epoch/Iterations/Loss/Accuracy
            if i % 1000 == 0: # Show loss every 1000 mini-batches
                scores = []
                for _ in range(1024):
                    image_t, label_t = synthesize_data()
                    with torch.no_grad(): # Don't need gradients for validation. Saves memory
                        p_star, pred = model(torch.Tensor(image_t[None, None]).to(DEVICE))
                    np_pred = pred[0].detach().cpu().numpy()
                    scores.append(score_iou(np_pred, label_t))

                ious = np.asarray(scores, dtype="float")
                ious = ious[~np.isnan(ious)]  # remove true negatives

                # this section is to evaluate model as it trains
                epoch_num = epoch + 1
                actual_loss = np.mean(losses)
                p_star_cum = np.mean(p_stars)
                accuracy = (ious > 0.7).mean()*100
                fp_fn = scores.count(0)
                tp_low_iou = np.count_nonzero(ious<0.7)-fp_fn
                tn = scores.count(None)
                print(f'Epoch: {epoch_num}, Batches: {i}, Loss: {actual_loss:.3f}, P_Star: {p_star_cum:.3f}, '
                      f'Val Acc = {accuracy:.3f}%, FP & FN: {fp_fn}, TP w/ low IOU: {tp_low_iou}, TN: {tn}')

        # Store training stats after each epoch
        epoch_log.append(epoch_num)
        loss_log.append(actual_loss)
        accuracy_log.append(accuracy)

    # plot training stats
    fig, ax1 = plt.subplots()

    # x-axis label rotation
    plt.title("Accuracy & Loss vs Epoch")
    plt.xticks(rotation=45)

    # plot a secondary y axis
    ax2 = ax1.twinx()

    #plot loss_log and accuracy_log
    ax1.plot(epoch_log, loss_log, 'g-')
    ax2.plot(epoch_log, accuracy_log, 'b-')

    # Set labels
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='g')
    ax2.set_ylabel('Validation Accuracy', color='b')

    plt.show()

    return model


def main():

    model = Detector_FPN().to(DEVICE)
    inp = torch.rand((2, 1, 200, 200))
    summary(model, input_size=inp.shape[1:])
    star_model = train(
        model,
        torch.utils.data.DataLoader(StarDataset(), batch_size=64, num_workers=12),
        num_epochs=20,
    )
    torch.save(star_model.state_dict(), "model.pickle")

if __name__ == "__main__":
    main()
