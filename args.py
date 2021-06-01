import argparse

def init_arguments():
    parser = argparse.ArgumentParser(prog='Deep Learning - HW6: Object Detection')

    # Data
    parser.add_argument('-xml', '--FOLDERxml', type=str, default='./train_cdc/train_annotations/')
    parser.add_argument('-Pdt', '--dataPATH', type=str, default='./data_tables/')
    parser.add_argument('-Ptr', '--trainPATH', type=str, default='./train_cdc/train_images/')
    parser.add_argument('-Pte', '--testPATH', type=str, default='./test_cdc/test_images/')
    
    # General
    parser.add_argument('-s', '--seed', type=int, default=4028)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-Ps', '--savePATH', type=str, default='./output/', help='The path to store the outputs, including models, plots, and training and evalution results.')
    parser.add_argument('-pm', '--pretrained_model', type=str, default='yolov4', choices=['yolov4'])

    # Data augmentation
    parser.add_argument('--pretrained_size', type=int, default=96)
    parser.add_argument('--padding', type=int, default=10)
    parser.add_argument('--rotation', type=int, default=5)
    parser.add_argument('--horizontal_flip', type=float, default=.5)

    # Training
    parser.add_argument('-d', '--device', type=str, default='cuda:1', choices=['cpu', 'cuda', 'cuda:1'], help='Device name')
    parser.add_argument('-o', '--optimizer', type=str, default='adam', choices=['sgd', 'adam'], help='Optimizer')
    parser.add_argument('-lr', '--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-vs', '--val_size', type=float, default=0.15)
    parser.add_argument('-ts', '--test_size', type=float, default=0.15)
    parser.add_argument('-fs', '--figsize', nargs='+', type=int, default=[8, 6], help='Figure size of model performance plot. Its length should be 2.')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='No. of epochs')
    parser.add_argument('-pp', '--print_result_per_epochs', type=int, default=10)
    
    return parser