import torch.nn as nn
import torchvision.models as models
import torch
import os
import gzip
import logging
logger = logging.getLogger(__name__)

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = 'densenet121'
        self.pretrained = args.visual_extractor_pretrained
        self.cached_file = "/opt/data/ARGON/containers/models/chexpert_auc14.dict.gz"
        if os.path.exists(self.cached_file):
            self.pretrained = False
        else:
            self.pretrained = True

        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)

        if not self.pretrained:
            print("loading densnet parameters")
            logger.info("loading densnet parameters from {}".format(self.cached_file))
            with gzip.open(self.cached_file) as f:
                state_dict = torch.load(f, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)

        # modules = list(model.children())[:-2]
        #self.model = model
        self.model = nn.Sequential(*list(model.features.children()))
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.fine_tune(fine_tune=True)

    def fine_tune(self, fine_tune=False):
        """
        Allow or prevent the computation of gradients for convolutional blocks of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.model.parameters():
            p.requires_grad = fine_tune


    def forward(self, images):
        # CNN features
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape # torch.Size([16, 1024, 8, 8])

        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)#feature_sizetorch.Size([16, 64, 1024])    
        return patch_feats, avg_feats