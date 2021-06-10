from args import get_args
from trainer import Trainer
from utils import *
import os, time

def main(args):
    # If the dataset has not been splitted yet,
    if not os.path.exists(args.dtPATH + 'table_tr.txt'):
        out = load_xml(args.FOLDERxml)
        out_tr, out_va, out_te = data_splitting(out, args.val_size, args.test_size)
        save_txt(out_tr, args.dtPATH, 'table_tr')
        save_txt(out_va, args.dtPATH, 'table_va')
        save_txt(out_te, args.dtPATH, 'table_te')
    
    # If the json files have not been produced,
    if not os.path.exists(args.dtPATH + 'objects_tr.json'):
        create_data_lists(
            subsets=['tr', 'va', 'te'],
            dtPATH=args.dtPATH,
            labelPATH=args.FOLDERxml,
            imgPATH=args.trainPATH
        )

    voc_labels = ('aquarium', 'bottle', 'bowl', 'box', 'bucket', 'plastic_bag', 'plate', 'styrofoam', 'tire', 'toilet', 'tub', 'washing_machine', 'water_tower')
    label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
    label_map['background'] = 0
    rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

    print('Initializing the trainer ...')
    trainer = Trainer(
        dtPATH=args.dtPATH, savePATH=args.savePATH,
        device=args.device, workers=args.workers,
        label_map=label_map, keep_difficult=True,
        optimizer=args.optimizer, lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
        start_epoch=args.start_epoch, epochs=args.epochs,
        batch_size=args.batch_size, dt=args.dt, checkpoint=args.checkpoint,
        decay_lr_at=args.decay_lr_at
    )

    print('\nTraining and evaluating on the validation set ...')
    t0 = time.time()
    trainer.train(decay_lr_to=args.decay_lr_to, print_freq=args.print_freq, print_freq_epoch=args.print_freq_epoch)
    tCost = time.time() - t0
    
    print('Finish training! Time cost: %8.2f' % tCost)
    print('Testing performance:')
    mAP_te = trainer.evaluate(trainer.loader_te, 'testing')

    print('Ploting ...')
    trainer.plot_training(plot_type='loss', figsize=args.figsize, save_plot=True)
    trainer.plot_training(plot_type='mAP', figsize=args.figsize, save_plot=True)

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)