import argparse

def init_arguments():
    parser = argparse.ArgumentParser(prog='Deep Learning - HW6: Object Detection')

    # General
    parser.add_argument('-xml', '--FOLDERxml', type=str, default='./train_cdc/train_annotations')

    parser.add_argument('--NonCNN', action='store_true')
    parser.add_argument('--seed', type=int, default=4028)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--label', type=str, default='Label')
    parser.add_argument('--dataPATH', type=str, default='./train_images/', help='The path where should the data be loaded in.')
    parser.add_argument('--trainDATA', type=str, default='./train.csv')
    parser.add_argument('--testDATA', type=str, default='./test.csv')
    parser.add_argument('--savePATH', type=str, default='./output_torch/', help='The path to store the outputs, including models, plots, and training and evalution results.')
    parser.add_argument('--pretrained_model', type=str, default='resnet50', choices=['resnet18', 'resnet50', 'resnet101', 'vgg16', 'efficientnet-b7'])

    # Data augmentation
    parser.add_argument('--pretrained_size', type=int, default=96)
    parser.add_argument('--padding', type=int, default=10)
    parser.add_argument('--rotation', type=int, default=5)
    parser.add_argument('--horizontal_flip', type=float, default=.5)

    # Training
    parser.add_argument('--val_size', type=float, default=0.15)
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--device', type=str, default='cuda:1', choices=['cpu', 'cuda', 'cuda:1'], help='Device name')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'], help='Optimizer')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='No. of epochs')
    parser.add_argument('--figsize', nargs='+', type=int, default=[8, 6], help='Figure size of model performance plot. Its length should be 2.')
    parser.add_argument('--print_result_per_epochs', type=int, default=10)
    
    return parser