import torch
from torch import nn
from transformers import AutoModel

class PhoBERT_MultiLabel(nn.Module):
    def __init__(self,n_classes,model= ""):
        super(PhoBERT_MultiLabel, self).__init__()
        self.bert = AutoModel.from_pretrained(model,return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids,attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output