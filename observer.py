import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,  patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class Runtime_Observer:
    def __init__(self, log_dir, device='cuda', **kwargs):
        """
        The Observer of training, which contains a log file(.txt), computing tools(torchmetrics) and tensorboard writer
        :author windbell
        :param log_dir: output dir
        :param device: Default to 'cuda'
        :param kwargs: Contains the experiment name and random number seed
        """
        self.best_dicts = {'epoch': 0, 'acc': 0, 'auc': 0, 'f1': 0, 'p': 0, 'recall': 0, 'spe': 0}
        self.log_dir = str(log_dir)
        self.log_ptr = open(self.log_dir + '/log.txt', 'w')
        _kwargs = {'name': kwargs['name'] if kwargs.__contains__('name') else 'None',
                   'seed': kwargs['seed'] if kwargs.__contains__('seed') else 'None'}

        self.test_acc = torchmetrics.Accuracy(num_classes=2, task='binary').to(device)
        self.test_recall = torchmetrics.Recall(num_classes=2, task='binary').to(device)
        self.test_precision = torchmetrics.Precision(num_classes=2, task='binary').to(device)
        self.test_auc = torchmetrics.AUROC(num_classes=2, task='binary').to(device)
        self.test_F1 = torchmetrics.F1Score(num_classes=2, task='binary').to(device)
        self.summary = SummaryWriter(log_dir=self.log_dir + '/summery')
        self.log_ptr.write('exp:' + str(_kwargs['name']) + '  seed -> ' + str(_kwargs['seed']))
        self.early_stopping = EarlyStopping(patience=50, verbose=True)
        self.test_spe = torchmetrics.Specificity(num_classes=2, task='binary').to(device)
        self.model_save_path = self.log_dir + '/best_model.pth'
    def update(self, prediction, label):
        self.test_acc.update(prediction, label)
        self.test_auc.update(prediction, label)
        self.test_recall.update(prediction, label)
        self.test_precision.update(prediction, label)
        self.test_F1.update(prediction, label)
        self.test_spe.update(prediction, label)

    def log(self, info: str):
        print(info)
        self.log_ptr.write(info)

    def save_model(self, model):
        """保存模型的最佳权重"""
        torch.save(model.state_dict(), self.model_save_path)
        self.log(f"Best model saved to {self.model_save_path}\n")

    def excute(self, epoch, model=None):
        def _save():
            self.best_dicts['acc'] = total_acc
            self.best_dicts['epoch'] = epoch
            self.best_dicts['auc'] = total_auc
            self.best_dicts['f1'] = total_F1
            self.best_dicts['p'] = total_precision
            self.best_dicts['recall'] = total_recall
            self.best_dicts['spe'] = total_spe

        total_acc = self.test_acc.compute()
        total_recall = self.test_recall.compute()
        total_precision = self.test_precision.compute()
        total_auc = self.test_auc.compute()
        total_F1 = self.test_F1.compute()
        total_spe = self.test_spe.compute()

        self.early_stopping(total_acc)

        self.summary.add_scalar('val_acc', total_acc, epoch)
        self.summary.add_scalar('val_recall', total_recall, epoch)
        self.summary.add_scalar('val_precision', total_precision, epoch)
        self.summary.add_scalar('val_auc', total_auc, epoch)
        self.summary.add_scalar('val_f1', total_F1, epoch)
        self.summary.add_scalar('val_spe', total_spe, epoch)

        if total_acc >= self.best_dicts['acc']:
            _save()
            self.save_model(model)
            if total_auc >= self.best_dicts['auc']:
                _save()
                self.save_model(model)
                if total_auc >= self.best_dicts['f1']:
                    _save()
                    self.save_model(model)
                    if abs(total_precision - total_recall) <= abs(self.best_dicts['p'] - self.best_dicts['recall']):
                        _save()
                        self.save_model(model)

        log_info = "-------\n" + "Epoch %d:\n" % (epoch + 1) \
                   + "Val Accuracy: %4.2f%%  || " % (total_acc * 100) + \
                   "best accuracy : %4.2f%%" % (self.best_dicts['acc'] * 100) \
                   + " produced @epoch %3d\n" % (self.best_dicts['epoch'] + 1)
        self.log(log_info)

        return self.early_stopping.early_stop

    def record(self, epoch, train_loss, val_loss):
        self.summary.add_scalar('train_loss', train_loss, epoch)
        self.summary.add_scalar('val_loss', val_loss, epoch)
        # self.summary.add_scalar('Linear_acc', test_acc, epoch)

        self.log(f"Epoch {epoch + 1}, Average train Loss: {train_loss}\n" \
                 + f'Average val Loss:{val_loss}')

    def record_loss(self, epoch, loss, tloss):
        self.summary.add_scalar('train_loss', loss, epoch)
        self.summary.add_scalar('test_loss', tloss, epoch)

    def reset(self):
        self.test_acc.reset()
        self.test_auc.reset()
        self.test_recall.reset()
        self.test_precision.reset()
        self.test_F1.reset()
        self.test_spe.reset()

    def finish(self):
        finish_info = "---experiment ended---\n" \
                      + "Best Epoch %d:\n" % (self.best_dicts['epoch'] + 1) \
                      + "Accuracy : %4.2f%%" % (self.best_dicts['acc'] * 100) \
                      + "Precision : %4.2f%%\n" % (self.best_dicts['p'] * 100) \
                      + "F1 score : %4.2f%%" % (self.best_dicts['f1'] * 100) \
                      + "AUC : %4.2f%%" % (self.best_dicts['auc'] * 100) \
                      + "Recall : %4.2f%%\n" % (self.best_dicts['recall'] * 100) \
                      + "Specificity (SPE): %4.2f%%\n" % (self.best_dicts['spe'] * 100) \
                      + "exiting..."
        self.log(finish_info)
        self.log_ptr.close()