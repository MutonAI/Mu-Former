import argparse
import sys
import os
from pathlib import Path
import torch
import pandas as pd
import traceback
import csv

sys.path.append(os.path.join(Path(__file__).parent, "./protein/"))
from src.trainer import Trainer

from utils import Logger
logger = Logger()

def main(args):
    trainer = Trainer(
        output_dir=args.output_dir,
        train_tsv=args.train,
        valid_tsv=args.valid,
        test_tsv=args.test,
        fasta=args.fasta,
        split_ratio=args.split_ratio,
        num_ensembles=args.num_ensembles,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        save_log=args.save_log,
        num_workers=args.num_workers,
        aug_ratio=args.aug_ratio,
        encoder_name=args.encoder_name,
        pretrained_model=args.pretrained_model,
        loss=args.loss,
        lr=args.lr,
        label_oper=args.label_oper,
        decoder_name=args.decoder_name,
        freeze_lm=args.freeze_lm,
        local_rank=args.local_rank,
        random_init=args.random_init,
        random_seed=args.seed,
        args=args,
    )

    if args.saved_model_dir:
        if args.strip_load_checkpoint:
            trainer.load_checkpoint_strip(args.saved_model_dir)
        else:
            trainer.load_checkpoint(args.saved_model_dir)
        test_results = trainer.test(
            model_label='Test', mode='ensemble',
            save_prediction=args.save_prediction,
        )
    else:
        trainer.train(
            epochs=args.epochs, 
            patience=args.patience,
            eval_freq=args.eval_freq,
            save_checkpoint=args.save_checkpoint,
            resume_path=args.resume_path,
        )
        test_results = trainer.test(
            mode='ensemble',
            save_prediction=args.save_prediction,
            checkpoint_dir=args.output_dir,
        )

    mode = args.test if args.test else args.train if args.train else None  
    test_res_msg = f'Test on {mode} \n' if mode else ''  
    test_res_msg += f'Ensembling Model - Loss: {test_results["loss"]:.4f}\t'
    test_res_msg += f'\t'.join([f'Test {k}: {v:.4f}' for (k, v) in test_results['metric'].items()]) + '\n'
    trainer.logger.info(test_res_msg)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', action='store', help='training data (TSV format)')
    parser.add_argument('--valid', action='store', help='valid data (TSV format)')
    parser.add_argument('--test', action='store', help='test data (TSV format)')
    parser.add_argument('--fasta', action='store', help='native sequence (FASTA format)')

    parser.add_argument('--num-ensembles', action='store', type=int, default=1, help='number of models in ensemble')
    parser.add_argument('--split-ratio', action='store', type=float, nargs='+', default=[0.7, 0.1, 0.2],
                        help='ratio to split training data. [train, valid] or [train, valid, test]')
    
    parser.add_argument('--hidden-size', action='store', type=int, default=256, help='hidden dimension in top layer')
    parser.add_argument('--kernel-size', action='store', type=int, default=3, help='kernel size of the convolution module')
    parser.add_argument('--conv-layers', action='store', type=int, default=1, help='layers of the convolution module')

    parser.add_argument('--epochs', action='store', type=int, default=300, help='total epochs')
    parser.add_argument('--patience', action='store', type=int, help='patience for early stopping')
    parser.add_argument('--batch-size', action='store', type=int, default=8, help='batch size')
    parser.add_argument('--eval-freq', action='store', type=int, default=1,
                        help='evaluate (on validation set) per N epochs')

    parser.add_argument('--saved-model-dir', action='store', help='directory of trained models')
    parser.add_argument('--output-dir', action='store', help='directory to save model, prediction, etc.')
    parser.add_argument('--save-checkpoint', action='store_true', default=True, help='save pytorch model checkpoint')
    parser.add_argument('--save-prediction', action='store_true', default=True, help='save prediction')
    parser.add_argument('--save-log', action='store_true', default=True, help='save log file')

    parser.add_argument('--dataset-name', action='store', default='', help='dataset name for parallel training')
    parser.add_argument('--label-column', action='store', default='', help='dataset column used')
    parser.add_argument('--skip-bad-lines', action='store_true', help='Skip bad lines')
    parser.add_argument('--datapath', action='store', default='', help='dataset root folder')
    parser.add_argument('--resume-path', action='store', help='load ckpt to continue')
    parser.add_argument('--num-workers', action='store', type=int, default=4)
    parser.add_argument('--lr', action='store', type=float, default=0.0001)
    parser.add_argument('--encoder-lr', action='store',type=float, default=0.00001)
    parser.add_argument('--decoder-lr', action='store',type=float, default=0.001)
    parser.add_argument('--weight-decay', action='store',type=float,default=0.0)
    parser.add_argument('--dropout', action='store',type=float, default=0.2)
    parser.add_argument('--pretrained-model', action='store', default='', help='the specific path to load the pretrained model from')

    parser.add_argument('--freeze-lm', action='store_true', default=False, help='freeze the language model weights')
    parser.add_argument('--loss', action='store', default='mae', help='loss function used for training')
    parser.add_argument('--activate', action='store', default='tanh', help='activation function')
    parser.add_argument('--warmup-epochs', action='store', type=int, default=5)

    parser.add_argument('--label-oper', action='store', default='minus', help='data operate, choose from [divide, minus]')
    parser.add_argument('--aug-ratio', action='store', default=-1, type=float, help='if augment, the ratio of wt and mt')
    parser.add_argument('--decoder-name', action='store', default='siamese', help='decoder name')
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument('--metric', action='store', default='corr', help='the metric used to choose the best ckpt')
    parser.add_argument('--seed', action='store', type=int, default=42, help='the random seed')

    parser.add_argument('--shuffle', action='store', default=True, help='shuffle training data')
    
    parser.add_argument('--random-init', action='store', default=False, help='w/o pretraining?')
    
    parser.add_argument('--encoder-name', action='store', default='pmlm', help='decoder name, choose from [pmlm, esm1, esm/esm2]')
    parser.add_argument('--mix-precision', action='store', default=False, help='use mix precision?')
    
    parser.add_argument('--strip-load-checkpoint', action='store', default=False, help='Strip "module." prefix when loading the finetinued model')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)


