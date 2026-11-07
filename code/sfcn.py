"""SFCN (Simple Fully Convolutional Network) architecture and loader.

This is the original SFCN of Peng et al. (2021), "Accurate brain age prediction
with lightweight deep neural networks" (Medical Image Analysis). Pretrained UK
Biobank weights (brain-age model) are released at:
    https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain

Das et al. (2024) used this network as a frozen feature extractor: they kept the
six convolutional blocks and discarded the final classifier, then trained a new
head on top. We do the same. The brain-age checkpoint is sufficient because only
the convolutional feature extractor is reused; the brain-age output layer is dropped.

Input geometry (must match the pretraining): 1 mm MNI T1, cropped to 160x192x160.
The feature extractor then produces a 64 x 5 x 6 x 5 feature map, which is the
"5 x 6 x 5" map named in Das et al. (2024).

The exact weights Das et al. used are not published, so this original release is
used instead. The protocol's decision log records this choice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SFCN(nn.Module):
    """Simple Fully Convolutional Network (Peng et al., 2021).

    channel_number and output_dim match the released brain-age model so that the
    pretrained state_dict loads without renaming. output_dim is the number of soft
    age bins in the pretraining task; it is irrelevant here because the classifier
    is dropped and only `feature_extractor` is reused.
    """

    def __init__(self, channel_number=(32, 64, 128, 256, 256, 64), output_dim=40, dropout=True):
        super().__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            in_channel = 1 if i == 0 else channel_number[i - 1]
            out_channel = channel_number[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module(
                    "conv_%d" % i,
                    self.conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=1),
                )
            else:
                # Final block: 1x1 conv, no maxpool. Output stays at the 5x6x5 grid.
                self.feature_extractor.add_module(
                    "conv_%d" % i,
                    self.conv_layer(in_channel, out_channel, maxpool=False, kernel_size=1, padding=0),
                )

        self.classifier = nn.Sequential()
        self.classifier.add_module("average_pool", nn.AvgPool3d((5, 6, 5)))
        if dropout:
            self.classifier.add_module("dropout", nn.Dropout(0.5))
        self.classifier.add_module(
            "conv_%d" % n_layer, nn.Conv3d(channel_number[-1], output_dim, padding=0, kernel_size=1)
        )

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool:
            return nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        return nn.Sequential(
            nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


def load_pretrained_feature_extractor(checkpoint_path, device="cpu", freeze=True):
    """Load the pretrained SFCN and return ONLY the frozen six-block feature extractor.

    The released checkpoint is a plain state_dict whose keys are prefixed
    `feature_extractor.` and `classifier.`. We load with strict=False so the
    classifier keys (which we drop) do not raise. On first run, print any missing
    or unexpected keys and confirm the feature-extractor keys all matched
    (first-run checklist in the notebook).
    """
    model = SFCN()
    state = torch.load(checkpoint_path, map_location=device)
    state = state.get("state_dict", state)
    # Some releases wrap weights under a DataParallel `module.` prefix.
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    fe_missing = [k for k in missing if k.startswith("feature_extractor.")]
    if fe_missing:
        raise RuntimeError(
            "Pretrained feature-extractor keys did not load: %s. "
            "Check the checkpoint source; the weights file may be the wrong one." % fe_missing
        )

    extractor = model.feature_extractor
    if freeze:
        for p in extractor.parameters():
            p.requires_grad_(False)
    extractor.to(device).eval()
    return extractor


class AgeSFCN(nn.Module):
    """Frozen SFCN feature extractor + a small trainable age-group classification head.

    The proxy task is age dichotomized at the cohort median (older vs younger); the
    frozen SFCN is a brain-age model, so age is the signal its features encode (sex
    was near chance and was dropped as the task, decision log D11).

    The head mirrors Das et al. (2024): a 512-unit dense layer, ReLU, dropout 0.5.
    We output two logits (softmax) instead of one sigmoid so that captum target
    indexing and the deletion/insertion faithfulness curves work directly; this is
    equivalent to the binary head. Logged in the decision log.
    """

    def __init__(self, feature_extractor, n_features=64 * 5 * 6 * 5, hidden=512, n_classes=2):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.head(self.feature_extractor(x))
