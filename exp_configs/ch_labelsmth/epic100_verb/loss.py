import torch
from torch import nn
from torch.nn.functional import cross_entropy

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

        multilabels = torch.stack(multilabels)
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
                l = cross_entropy(vid_o.view(1,-1), t.view(-1))
                losses.append(l)

            video_loss = min(losses)
            batch_losses.append(video_loss)

        assert len(batch_losses) > 0, f'No loss calculated for this batch, adjust threshold which was {thr}'
        batch_loss = sum(batch_losses) / len(batch_losses)
        return batch_loss
