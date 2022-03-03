import numpy as np
import torch
from torch import nn
from torch.nn.functional import nll_loss

class MinCEMultilabelLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def generate_multilabels(soft_labels, thr, singlelabels):
        """
        New label would be everything with soft label > thr, and singlelabels
        """
        multilabels = []
        for soft_label, singlelabel in zip(soft_labels, singlelabels):
            multilabel = (soft_label > thr).float()
            multilabel[singlelabel] = 1.
            multilabels.append(multilabel)

        multilabels = torch.stack(multilabels)
        return multilabels

    @staticmethod
    def generate_multilabels_numpy(soft_labels, thr, singlelabels):
        """
        New label would be everything with soft label > thr, and singlelabels
        """
        multilabels = []
        for soft_label, singlelabel in zip(soft_labels, singlelabels):
            multilabel = (soft_label > thr).astype(int)
            multilabel[singlelabel] = 1
            multilabels.append(multilabel)

        multilabels = np.stack(multilabels)
        return multilabels
            

    def forward(self, output, multilabels):
        o = torch.log_softmax(output, dim=1)
        batch_losses = []

        # labels is expected to be of shape (batch, n_classes), containing soft_labels
        for vid_o, vid_l in zip(o, multilabels):
            targets = torch.where(vid_l == 1.)[0]

            assert len(targets) > 0, f'No target found with for multi-labels {vid_l}. ' \
                                     f'This should never happen since we include now the gt label'
            losses = []

            for t in targets:
                l = nll_loss(vid_o.view(1,-1), t.view(-1))
                losses.append(l)

            video_loss = min(losses)
            batch_losses.append(video_loss)

        assert len(batch_losses) > 0, f'No loss calculated for this batch, adjust threshold which was {thr}'
        batch_loss = sum(batch_losses) / len(batch_losses)
        return batch_loss


def subsets(arr):
    return list(chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)]))
class MinRegressionCombinationLoss(nn.Module):
    """
    Given multi-labels, get loss for all possible combinations of the targets and take the minimum loss.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, output, multilabels):

        o = torch.sigmoid(output)
        for i, (vid_o, vid_l) in enumerate(zip(o, multilabels)):
            targets = torch.where(vid_l == 1.)[0]

            all_labels_combinations = subsets(targets.tolist())
            vid_losses = []

            for g in all_labels_combinations:
                g = list(g)  # g is a tuple
                pseudo_y = torch.zeros_like(vid_l, device=vid_l.device)
                pseudo_y[g] = 1
                loss = (pseudo_y * torch.log(vid_o + eps)) + ((1 - pseudo_y) * torch.log(1 - vid_o + eps))
                loss = -loss.sum()
                vid_losses.append(loss)

            vid_loss = min(vid_losses)
            batch_losses.append(vid_loss)

        assert len(batch_losses) > 0, f'No loss calculated for this batch, adjust threshold which was {thr}'
        batch_loss = sum(batch_losses) / len(batch_losses)
        return batch_loss
