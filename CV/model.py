import torch
import torch.nn as nn
from torch.nn import functional as func
import torchvision.models as models
import torchmetrics
import pytorch_lightning as pl
from lion import Lion

def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


class InnerModel(nn.Module):    
    def __init__(self, class_size):
        super(InnerModel, self).__init__()
        self.class_size = class_size
        self.maxvit_t = models.maxvit_t(pretrained=True)
        num_features = self.maxvit_t.classifier[-1].in_features
        self.maxvit_t.classifier[-1] = nn.Linear(num_features, num_features).apply(init_weights_xavier)
        self.maxvit_t.classifier.append(nn.Tanh())
        self.maxvit_t.classifier.append(nn.Linear(num_features, class_size, bias=False))
        
    def forward(self, x):
        return self.maxvit_t(x)

class ImageClassifier(pl.LightningModule):
    def __init__(self, class_size, eval_class_size, lr):
        super(ImageClassifier, self).__init__()
        self.model = InnerModel(class_size)
        self.save_hyperparameters()
        self.lr = lr
        self.class_size = class_size
        self.eval_class_size = eval_class_size

        self.train_precision = torchmetrics.classification.BinaryPrecision()
        self.train_recall = torchmetrics.classification.BinaryRecall()
        self.train_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.train_f1_score = torchmetrics.classification.BinaryF1Score()

        self.valid_precision = torchmetrics.classification.BinaryPrecision()
        self.valid_recall = torchmetrics.classification.BinaryRecall()
        self.valid_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.valid_f1_score = torchmetrics.classification.BinaryF1Score()

        self.test_precision = torchmetrics.classification.BinaryPrecision()
        self.test_recall = torchmetrics.classification.BinaryRecall()
        self.test_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.test_f1_score = torchmetrics.classification.BinaryF1Score()


    def forward(self, x):
        return self.model(x)
    
    def compute_loss(self, logits, y):
        return func.cross_entropy(logits, y)

    def convert_to_labels(self, logits, y):
        y_pred_class = torch.argmax(logits, dim=-1)
        y_pred = (y_pred_class != 1).to(torch.int8)
        y_label = (y != 1).to(torch.int8)
        return y_pred, y_label
    

    def base_step(self, batch, metrics):
        x, y = batch
        logits = self(x)

        loss = self.compute_loss(logits, y)

        y_pred, y_label = self.convert_to_labels(logits, y)
        
        for _, metric in metrics.items():
            metric(y_pred, y_label)
        
        return loss
        
    def training_step(self, batch, _):
        metrics = {
            'train_prec': self.train_precision,
            'train_recall': self.train_recall,
            'train_acc': self.train_accuracy,
            'train_f1': self.train_f1_score,
        }
        loss = self.base_step(batch, metrics)
        logs = {
            'train_loss': loss,
            'step': self.current_epoch,
            **metrics
        }
        self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, _):
        metrics = {
            'valid_prec': self.valid_precision,
            'valid_recall': self.valid_recall,
            'valid_acc': self.valid_accuracy,
            'valid_f1': self.valid_f1_score,
        }
        loss = self.base_step(batch, metrics)
        logs = {
            'valid_loss': loss,
            'step': self.current_epoch,
            **metrics
        }
        self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, _):
        metrics = {
            'test_prec': self.test_precision,
            'test_recall': self.test_recall,
            'test_acc': self.test_accuracy,
            'test_f1': self.test_f1_score,
        }
        loss = self.base_step(batch, metrics)
        logs = {
            'test_loss': loss,
            'step': self.current_epoch,
            **metrics
        }
        self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = Lion(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)

        return [optimizer], [scheduler]









