import time
import numpy as np
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
from tqdm import tqdm
from datetime import datetime
from pprint import PrettyPrinter

cudnn.benchmark = True

class Trainer:
    def __init__(self, dtPATH:str, savePATH:str, device:str, workers:int, label_map:dict, keep_difficult:bool,
                 optimizer:str, lr:float, momentum:float, weight_decay:float,
                 start_epoch:int, epochs:int, batch_size:int, dt:str, checkpoint:str, decay_lr_at:int):

        self.dtPATH = dtPATH
        self.savePATH = savePATH
        self.n_classes = len(label_map)
        self.batch_size = batch_size
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # Good formatting when printing the APs for each class and mAP
        self.pp = PrettyPrinter()

        # Initialize model or load checkpoint
        if checkpoint is None:
            start_epoch = 0
            self.model = SSD300(n_classes=self.n_classes)
            
            # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
            biases, not_biases = [], []
            for param_name, param in self.model.named_parameters():
                if param.requires_grad:
                    if param_name.endswith('.bias'):
                        biases.append(param)
                    else:
                        not_biases.append(param)
            
            if optimizer.lower() == 'sgd':
                self.optimizer = torch.optim.SGD(
                    params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                    lr=lr, momentum=momentum, weight_decay=weight_decay
                )
            elif optimizer.lower() == 'adam':
                self.optimizer = torch.optim.Adam(
                    params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                    lr=lr, weight_decay=weight_decay
                )
        else:
            checkpoint = torch.load(self.savePATH + dt + checkpoint)
            start_epoch = checkpoint['epoch'] + 1
            print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
            self.model = checkpoint['model']
            self.optimizer = checkpoint['optimizer']

        # Move to default device
        self.model = self.model.to(self.device)
        self.criterion = MultiBoxLoss(priors_cxcy=self.model.priors_cxcy).to(self.device)

        # Custom dataloaders
        dataset_tr = PascalVOCDataset(data_folder=self.dtPATH, subset='tr', keep_difficult=keep_difficult)
        dataset_va = PascalVOCDataset(data_folder=self.dtPATH, subset='va', keep_difficult=keep_difficult)
        dataset_te = PascalVOCDataset(data_folder=self.dtPATH, subset='te', keep_difficult=keep_difficult)

        self.loader_tr = torch.utils.data.DataLoader(
            dataset_tr, batch_size=batch_size, shuffle=True,
            collate_fn=dataset_tr.collate_fn,
            num_workers=workers, pin_memory=True
        )  # note that we're passing the collate function here
        
        self.loader_va = torch.utils.data.DataLoader(
            dataset_va, batch_size=batch_size, shuffle=False,
            collate_fn=dataset_va.collate_fn,
            num_workers=workers, pin_memory=True
        )
        
        self.loader_te = torch.utils.data.DataLoader(
            dataset_te, batch_size=batch_size, shuffle=True,
            collate_fn=dataset_tr.collate_fn,
            num_workers=workers, pin_memory=True
        )

        self.decay_lr_at = [it // (len(dataset_tr) // 32) for it in decay_lr_at]
        # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
        # To convert iterations to epochs, divide iterations by the number of iterations per epoch
        # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
        # epochs = iterations // (len(dataset_tr) // 32)
        # decay_lr_at = [it // (len(dataset_tr) // 32) for it in decay_lr_at]

    def train(self, decay_lr_to=.1, print_freq=10, print_freq_epoch=500):
        
        self.dt = datetime.now().strftime('%d-%H-%M-%S')
        self.folder_name = self.savePATH + self.dt + '_' + '_bs=' + str(self.batch_size) + '_epochs=' + str(self.epochs) + '/'
        self.lst_mAP_va = []
        self.lst_loss_tr = []
        self.best_mAP_va = 0
        mAP_va = 0

        for epoch in range(self.start_epoch, self.epochs):
            # Decay learning rate at particular epochs
            if epoch in self.decay_lr_at:
                adjust_learning_rate(self.optimizer, decay_lr_to)
            # One epoch's training
            self.train_single(epoch, print_freq=print_freq)
            
            if (epoch+1) % print_freq_epoch == 0:
                mAP_va = self.evaluate(self.loader_va, 'validation')
                self.lst_mAP_va.append(mAP_va)

            # Save checkpoint
            if mAP_va == 0 or mAP_va > self.best_mAP_va:
                save_checkpoint(epoch=epoch, model=self.model, optimizer=self.optimizer, folder=self.folder_name)


    def train_single(self, epoch, print_freq, grad_clip=None):
        '''
        One epoch's training.
        :param train_loader: DataLoader for training data
        :param model: model
        :param criterion: MultiBox loss
        :param optimizer: optimizer
        :param epoch: epoch number
        '''
        self.model.train()           # training mode enables dropout
        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()   # data loading time
        losses = AverageMeter()      # loss
        tStart = time.time()

        # Batches
        for i, (images, boxes, labels, _) in enumerate(self.loader_tr):
            data_time.update(time.time() - tStart)

            # Move to default device
            images = images.to(self.device)  # (batch_size (N), 3, 300, 300)
            boxes = [b.to(self.device) for b in boxes]
            labels = [l.to(self.device) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores = self.model(images)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            loss = self.criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

            # Backward prop.
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients, if necessary
            if grad_clip is not None:
                clip_gradient(self.optimizer, grad_clip)

            # Update model
            self.optimizer.step()
            self.lst_loss_tr.append(loss.cpu().detach().numpy())
            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - tStart)

            tStart = time.time()

            # Print status
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(self.loader_tr), batch_time=batch_time, data_time=data_time, loss=losses))

        del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


    def evaluate(self, test_loader, subset_name):
        '''
        Evaluate.
        :param test_loader: DataLoader for test data
        :param model: model
        '''
        # Make sure it's in eval mode
        self.model.eval()

        # Lists to store detected and true boxes, labels, scores
        det_boxes, det_labels, det_scores = [], [], []
        true_boxes, true_labels, true_difficulties = [], [], [] # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

        with torch.no_grad():
            # Batches
            for (images, boxes, labels, difficulties) in tqdm(test_loader, desc='Evaluating'):
                images = images.to(self.device)  # (N, 3, 300, 300)

                # Forward prop.
                predicted_locs, predicted_scores = self.model(images)

                # Detect objects in SSD output
                det_boxes_batch, det_labels_batch, det_scores_batch = self.model.detect_objects(
                    predicted_locs, predicted_scores,
                    min_score=0.01, max_overlap=0.45, top_k=200
                )
                # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

                # Store this batch's results for mAP calculation
                boxes = [b.to(self.device) for b in boxes]
                labels = [l.to(self.device) for l in labels]
                difficulties = [d.to(self.device) for d in difficulties]

                det_boxes.extend(det_boxes_batch)
                det_labels.extend(det_labels_batch)
                det_scores.extend(det_scores_batch)
                true_boxes.extend(boxes)
                true_labels.extend(labels)
                true_difficulties.extend(difficulties)

            # Calculate mAP
            APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

        # Print AP for each class
        self.pp.pprint(APs)
        print('\nmAP on the %s dataset: %.3f' % (subset_name, mAP))
        return mAP


    def plot_training(self, plot_type:str, figsize:tuple, save_plot=True):
        plt.figure(figsize=figsize)
        if plot_type == 'loss':
            plt.plot(np.arange(len(self.lst_loss_tr)), self.lst_loss_tr)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Plot of Loss')
        elif plot_type == 'mAP':
            print(self.lst_mAP_va)
            plt.plot(np.arange(len(self.lst_mAP_va)), self.lst_mAP_va)
            plt.xlabel('Epoch')
            plt.ylabel('mAP')
            plt.title('Plot of mAP')
        plt.grid()
            
        if save_plot:
            fn = self.folder_name + plot_type + '_plot.png'
            plt.savefig(fn)
            print('The', plot_type, 'plot is saved as', fn)
