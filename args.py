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
