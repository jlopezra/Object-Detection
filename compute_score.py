import numpy as np
import torch
from tqdm import tqdm
from model import Detector_FPN
from utils import DEVICE, score_iou, synthesize_data


def load_model():
    model = Detector_FPN() #StarModel()
    model.to(DEVICE)
    with open("model.pickle", "rb") as f:
        state_dict = torch.load(f, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def eval(*, n_examples: int = 1024) -> None:
    model = load_model()
    scores = []
    for _ in tqdm(range(n_examples)):
        image, label = synthesize_data()
        with torch.no_grad():
            p_star, pred = model(torch.Tensor(image[None, None]).to(DEVICE))
        np_pred = pred[0].detach().cpu().numpy()
        scores.append(score_iou(np_pred, label))

    ious = np.asarray(scores, dtype="float")
    ious = ious[~np.isnan(ious)]  # remove true negatives
    fp_fn = scores.count(0)
    tn = scores.count(None)
    tp_low_iou = np.count_nonzero(ious < 0.7) - fp_fn
    print(f'FP & FN: {fp_fn}, TP w/ low IOU: {tp_low_iou}, TN: {tn}')
    print((ious > 0.7).mean())


if __name__ == "__main__":
    eval()
