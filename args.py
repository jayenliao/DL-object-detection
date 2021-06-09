import argparse
#from easydict import EasyDict

def get_args():
    parser = argparse.ArgumentParser(prog='Deep Learning - HW6: Object Detection')

   # Data
    parser.add_argument('-Pdt', '--dtPATH', type=str, default='../data_tables/')
    parser.add_argument('-xml', '--FOLDERxml', type=str, default='../train_cdc/train_annotations/')
    parser.add_argument('-Ptr', '--trainPATH', type=str, default='../train_cdc/train_images/')
    parser.add_argument('-Pte', '--testPATH', type=str, default='../test_cdc/test_images/')

    # General
    parser.add_argument('-s', '--seed', type=int, default=4028)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-Ps', '--savePATH', type=str, default='../output/', help='The path to store the outputs, including models, plots, and training and evalution results.')
    parser.add_argument('-pm', '--pretrained_model', type=str, default='yolov4', choices=['yolov4'])

    # Data augmentation
    parser.add_argument('--pretrained_size', type=int, default=96)
    parser.add_argument('--padding', type=int, default=10)
    parser.add_argument('--rotation', type=int, default=5)
    parser.add_argument('--horizontal_flip', type=float, default=.5)

    # Training
    parser.add_argument('-cp', '--checkpoint', type=str, default=None)
    parser.add_argument('-se', '--start_epoch', type=int, default=0)
    parser.add_argument('-e', '--epochs', type=int, default=10, help='No. of epochs')
    parser.add_argument('-d', '--device', type=str, default='cuda:1', choices=['cpu', 'cuda', 'cuda:1'], help='Device name')
    parser.add_argument('-w', '--workers', type=int, default=4)
    parser.add_argument('-o', '--optimizer', type=str, default='adam', choices=['sgd', 'adam'], help='Optimizer')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-mt', '--momentum', type=float, default=.9)
    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-4)
    parser.add_argument('-dla', '--decay_lr_at', type=int, nargs='+', default=[80000, 100000])
    parser.add_argument('-dlt', '--decay_lr_to', type=float, default=.1)
    parser.add_argument('-vs', '--val_size', type=float, default=0.15)
    parser.add_argument('-ts', '--test_size', type=float, default=0.15)
    parser.add_argument('-fs', '--figsize', nargs='+', type=int, default=[8, 6], help='Figure size of model performance plot. Its length should be 2.')
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('-pp', '--print_freq', type=int, default=50)
    parser.add_argument('-ppe', '--print_freq_epoch', type=int, default=100)
    
    # Prediction
    parser.add_argument('-dt', '--dt', type=str)
    parser.add_argument('-ms', '--min_score', type=float, default=.2)
    parser.add_argument('-tr', '--transform', action='store_true')
    parser.add_argument('-trial', '--trial_and_error', action='store_true')
    parser.add_argument('-detect', '--detect', action='store_true')

    return parser

'''
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()
Cfg.use_darknet_cfg = True
Cfg.cfgfile = os.path.join(_BASE_DIR, 'cfg', 'yolov4.cfg')
Cfg.batch = 64
Cfg.subdivisions = 16
Cfg.width = 304
Cfg.height = 304
Cfg.channels = 3
Cfg.momentum = 0.949
Cfg.decay = 0.0005
Cfg.angle = 0
Cfg.saturation = 1.5
Cfg.exposure = 1.5
Cfg.hue = .1
Cfg.learning_rate = 0.00261
Cfg.burn_in = 1000
Cfg.max_batches = 500500
Cfg.steps = [400000, 450000]
Cfg.policy = Cfg.steps
Cfg.scales = .1, .1
Cfg.cutmix = 0
Cfg.mosaic = 1
Cfg.letter_box = 0
Cfg.jitter = 0.2
Cfg.classes = 13
Cfg.track = 0
Cfg.w = Cfg.width
Cfg.h = Cfg.height
Cfg.flip = 1
Cfg.blur = 0
Cfg.gaussian = 0
Cfg.boxes = 60  # box num
Cfg.TRAIN_EPOCHS = 3
Cfg.train_label = os.path.join(_BASE_DIR, './data_tables/', 'table_tr.txt')
Cfg.val_label = os.path.join(_BASE_DIR, './data_tables/' ,'table_va.txt')
Cfg.TRAIN_OPTIMIZER = 'adam'

if Cfg.mosaic and Cfg.cutmix:
    Cfg.mixup = 4
elif Cfg.cutmix:
    Cfg.mixup = 2
elif Cfg.mosaic:
    Cfg.mixup = 3

Cfg.checkpoints = os.path.join(_BASE_DIR, 'checkpoints')
Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, 'log')
Cfg.iou_type = 'iou'  # 'giou', 'diou', 'ciou'
Cfg.keep_checkpoint_max = 10

def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(
        prog='Deep Learning - HW6: Object Detection',
        description='Train the Model on images and target masks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Data
    parser.add_argument('-xml', '--FOLDERxml', type=str, default='./train_cdc/train_annotations/')
    parser.add_argument('-Pdt', '--dataPATH', type=str, default='./data_tables/')
    parser.add_argument('--imagesPATH', type=str, default='')
    parser.add_argument('-Ptr', '--trainPATH', type=str, default='./train_cdc/train_images/')
    parser.add_argument('-Pte', '--testPATH', type=str, default='./test_cdc/test_images/')
    
    # General
    parser.add_argument('-s', '--seed', type=int, default=4028)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-Ps', '--savePATH', type=str, default='./output/', help='The path to store the outputs, including models, plots, and training and evalution results.')
    parser.add_argument('-pm', '--pretrained_model', type=str, default='yolov4', choices=['yolov4'])

    # Model and Training
    parser.add_argument('-e', '--epochs', type=int, default=10, help='No. of epochs')
    parser.add_argument('-pp', '--print_result_per_epochs', type=int, default=10)

    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2, help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001, help='Learning rate', dest='learning_rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='0', help='GPU', dest='gpu')
    #parser.add_argument('-dir', '--data-dir', type=str, default=None, help='dataset dir', dest='dataset_dir')
    parser.add_argument('-pretrained', type=str, default=None, help='pretrained yolov4.conv.137')
    parser.add_argument('-classes', type=int, default=13, help='dataset classes')
    parser.add_argument('-optimizer', '--TRAIN_OPTIMIZER', type=str, default='adam', help='training optimizer')
    parser.add_argument('-iou-type', type=str, default='iou', help='iou type (iou, giou, diou, ciou)', dest='iou_type')
    parser.add_argument('-keep-checkpoint-max', type=int, default=10, help='maximum number of checkpoints to keep. If set 0, all checkpoints will be kept', dest='keep_checkpoint_max')
    args = vars(parser.parse_args())

    # for k in args.keys():
    #     cfg[k] = args.get(k)
    #print(args)
    cfg.update(args)
    #print('\n\n\n')
    #print(cfg)

    return EasyDict(cfg)

'''