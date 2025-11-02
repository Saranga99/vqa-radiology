import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import BertModel

class MedVQA(nn.Module):
    def __init__(self, num_answers):
        super(MedVQA, self).__init__()
        self.vision = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.vision.fc = nn.Identity()
        self.text = BertModel.from_pretrained("bert-base-uncased")
        self.fc1 = nn.Linear(2048 + 768, 512)
        self.fc2 = nn.Linear(512, num_answers)

    def forward(self, images, questions):
        v_feat = self.vision(images)
        t_feat = self.text(**questions).pooler_output
        combined = torch.cat((v_feat, t_feat), dim=1)
        x = F.relu(self.fc1(combined))
        return self.fc2(x)