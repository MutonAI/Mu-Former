import os
import warnings
from functools import lru_cache
import numpy as np
import pandas as pd
from Bio import SeqIO
import torch.utils.data

import torch
from torch.utils.data import Dataset

from utils import Logger
logger = Logger()

import vocab

def index_encoding(sequences):
    '''
    Modified from https://github.com/openvax/mhcflurry/blob/master/mhcflurry/amino_acid.py#L110-L130

    Parameters
    ----------
    sequences: list of equal-length sequences

    Returns
    -------
    np.array with shape (#sequences, length of sequences)
    '''
    df = pd.DataFrame(iter(s) for s in sequences)
    encoding = df.replace(vocab.AMINO_ACID_INDEX)
    encoding = encoding.values.astype(np.int)
    return encoding

def _pad_sequences(sequences, constant_value=1,length=0):
    batch_size = len(sequences)
    if not length:
        shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()
    else:
        shape = [batch_size] + [length]
    array = np.zeros(shape, sequences[0].dtype) + constant_value

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array

class FairseqLMEncoder(object):
    def __init__(self,
        alphabet,
        batch_size: int = 2,
        full_sequence_embed: bool = True,
        num_workers: int = 4,
        progress_bar: bool = True,
        ):
        
        self.tokenizer = alphabet   # task.source_dictionary
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.full_sequence_embed = full_sequence_embed
        self.progress_bar = progress_bar

    def encode(self, sequences,length=0):

        encoding = []
        for item in sequences:
            temp = '<s> '+' '.join(item)
            token_ids = self.tokenizer.encode_line(temp).numpy()  # seq -> <cls> <seq> encoder
            encoding.append(token_ids)
        # return torch.from_numpy(_pad_sequences(np.array(encoding))).long()
        
        res_comb = {}
        mask = [np.ones(i.shape[0]) for i in encoding]
        if len(encoding) > 1:
            res = torch.from_numpy(_pad_sequences(np.array(encoding,dtype=object),length=length).astype(int)).long()
            res_mask = torch.from_numpy(_pad_sequences(np.array(mask,dtype=object),length=length,constant_value=0).astype(int)).long()
        else:
            res = torch.from_numpy(_pad_sequences(np.array(encoding),length=length).astype(int)).long()
            res_mask = torch.from_numpy(_pad_sequences(np.array(mask),length=length,constant_value=0).astype(int)).long()
        res_comb['x'] = res
        res_comb['mask'] = res_mask
        
        return res_comb

class MetagenesisData(Dataset):
    def __init__(self, data, mode,
        native_sample,
        fasta=None, 
        label_oper="minus",
        aug_ratio=0.0
    ):

        self.data = data
        self.fasta = fasta
        self.mode = mode
        self.native_sample = native_sample
        # self.native_sequence = self._read_native_sequence()
        if self.fasta is not None:
            self.native_sequence = self._read_native_sequence()
        self.label_oper = label_oper
        self.aug_ratio = aug_ratio
        self.data_augment = self.aug_ratio > 0

    def __len__(self):
        if (self.mode == 'test') or (not self.data_augment):
        # if (self.mode != 'train') or (not self.data_augment):
            return len(self.data)
        else:
            return int(len(self.data) * (1 + self.aug_ratio))
    
    def _read_native_sequence(self):
        fasta = SeqIO.read(self.fasta, 'fasta')
        native_sequence = str(fasta.seq)
        return native_sequence
    
    @lru_cache(maxsize=32)
    def __getitem__(self, index):
        rec1 = self.data[index % len(self.data)].copy()

        # if (self.mode != 'train') or (not self.data_augment) or (index<len(self.data)):
        if (self.mode == 'test') or (not self.data_augment) or (index < len(self.data)):
            rand_index = -1
        else:
            rand_index = torch.randint(0,len(self.data),(1,))[0]
            while rand_index == index % len(self.data):
                rand_index = torch.randint(0, len(self.data), (1,))[0]
        
        if rand_index < 0:
            rec2 = self.native_sample
        else:
            rec2 = self.data[rand_index]
        
        if self.label_oper == "divide":
            rec1['label_af'] = rec1['label'] / rec2['label']
        elif self.label_oper == "minus":
            rec1['label_af'] = rec1['label'] - rec2['label']
        return (rec1, rec2)

class PairedDataset(object):
    def __init__(self,
            train_tsv=None, valid_tsv=None, test_tsv=None,
            fasta=None,
            split_ratio=[0.9, 0.1],
            random_seed=42,
            label_oper="minus",
            aug_ratio=-1,
            alphabet=None,
            distributed=False,
            encoder_name='pmlm',
            label_column='',
            skip_bad_lines=False,
            ):
        """
        split_ratio: [train, valid] or [train, valid, test]
        """

        self.train_tsv = train_tsv
        self.valid_tsv = valid_tsv
        self.test_tsv = test_tsv
        self.fasta = fasta
        self.split_ratio = split_ratio
        self.rng = np.random.RandomState(random_seed)
        self.label_oper = label_oper
        self.aug_ratio = aug_ratio
        self.lm_encoder = FairseqLMEncoder(alphabet)
        self.distributed = distributed

        self.label_column = label_column
        self.skip_bad_lines = skip_bad_lines
            
        if self.fasta is not None:
            self.native_sequence = self._read_native_sequence()

        self.train_valid_df = None
        self.full_df = None
        
        self.can_resample = False
        
        if test_tsv is None:
            if len(split_ratio) != 3:
                split_ratio = [0.7, 0.1, 0.2]
                logger.warning(# "split_ratio should have 3 elements if test_tsv is None." + \
                    f"Changing split_ratio to {split_ratio}. " + \
                    "Set values using --split_ratio.")
            self.full_df = self._read_mutation_df(train_tsv)
            self.train_df, self.valid_df, self.test_df = \
                self._split_dataset_df(self.full_df, split_ratio)
            self.can_resample = True
        elif valid_tsv is None:
            if len(split_ratio) != 2:
                split_ratio = [0.9, 0.1]
                logger.warning(# "split_ratio should have 2 elements if test_tsv is provided. " + \
                    f"Changing split_ratio to {split_ratio}. " + \
                    "Set values using --split_ratio.")
            self.test_df = self._read_mutation_df(test_tsv)
            self.train_valid_df = self._read_mutation_df(train_tsv)
            self.train_df, self.valid_df, _ = \
                self._split_dataset_df(self.train_valid_df, split_ratio)
            self.can_resample = True
        else:
            self.train_df = self._read_mutation_df(train_tsv)
            self.valid_df = self._read_mutation_df(valid_tsv)
            self.test_df = self._read_mutation_df(test_tsv)
            self.can_resample = False
        
        if self.train_valid_df is None:
            self.train_valid_df = pd.concat([self.train_df, self.valid_df]).reset_index(drop=True)
        
        if self.full_df is None:
            self.full_df = pd.concat([self.train_df, self.valid_df, self.test_df]).reset_index(drop=True)

    @property
    def max_len(self):
        return max(len(seq) for seq in self.full_df['sequence'].values)

    def _read_native_sequence(self):
        fasta = SeqIO.read(self.fasta, 'fasta')
        native_sequence = str(fasta.seq)
        return native_sequence
    
    def _check_split_ratio(self, split_ratio):
        """
        Modified from: https://github.com/pytorch/text/blob/3d28b1b7c1fb2ddac4adc771207318b0a0f4e4f9/torchtext/data/dataset.py#L284-L311
        """
        test_ratio = 0.
        if isinstance(split_ratio, float):
            assert 0. < split_ratio < 1., (
                "Split ratio {} not between 0 and 1".format(split_ratio))
            valid_ratio = 1. - split_ratio
            return (split_ratio, valid_ratio, test_ratio)
        elif isinstance(split_ratio, list):
            length = len(split_ratio)
            assert length == 2 or length == 3, (
                "Length of split ratio list should be 2 or 3, got {}".format(split_ratio))
            ratio_sum = sum(split_ratio)
            if not ratio_sum == 1.:
                if ratio_sum > 1:
                    split_ratio = [float(ratio) / ratio_sum for ratio in split_ratio]
            if length == 2:
                return tuple(split_ratio + [test_ratio])
            return tuple(split_ratio)
        else:
            raise ValueError('Split ratio must be float or a list, got {}'
                            .format(type(split_ratio)))


    def _split_dataset_df(self, input_df, split_ratio, resample_split=False):
        """
        Modified from:
        https://github.com/pytorch/text/blob/3d28b1b7c1fb2ddac4adc771207318b0a0f4e4f9/torchtext/data/dataset.py#L86-L136
        """
        _rng = self.rng.randint(512) if resample_split else self.rng
        df = input_df.copy()
        df = df.sample(frac=1, random_state=_rng).reset_index(drop=True)
        N = len(df)
        train_ratio, valid_ratio, test_ratio = self._check_split_ratio(split_ratio)
        train_len = int(round(train_ratio * N))
        valid_len = int(round(valid_ratio * N)) if int(round(valid_ratio * N)) < N-train_len else N-train_len
        test_len = int(round(test_ratio * N)) if int(round(test_ratio * N)) < N-train_len-valid_len else N-train_len-valid_len

        train_df = df.iloc[:train_len].reset_index(drop=True)
        valid_df = df.iloc[-1-test_len:-1-test_len-valid_len:-1].reset_index(drop=True)
        test_df = df.iloc[-1:-1-test_len:-1].reset_index(drop=True)

        return train_df, valid_df, test_df

    def _mutation_to_sequence(self, mutation):
        '''
        Parameters
        ----------
        mutation: ';'.join(WiM) (wide-type W at position i mutated to M)
        '''
        sequence = self.native_sequence
        for mut in mutation.split(';'):
            if mut.endswith('del'):
                wt_aa = mut[0]
                mt_aa = '-'
                pos = int(mut[1:-3])
            else:
                wt_aa = mut[0]
                mt_aa = mut[-1]
                pos = int(mut[1:-1])
            assert wt_aa == sequence[pos - 1],\
                    "%s: %s->%s (fasta WT: %s)"%(pos, wt_aa, mt_aa, sequence[pos - 1])
            sequence = sequence[:(pos - 1)] + mt_aa + sequence[pos:]
        sequence = sequence.replace('-', '')
        return sequence

    def _mutations_to_sequences(self, mutations):
        return [self._mutation_to_sequence(m) for m in mutations]

    def _drop_invalid_mutation(self, df):
        '''
        Drop mutations WiM where
        - W is incosistent with the i-th AA in native_sequence
        - M is ambiguous, e.g., 'X'
        '''
        flags = []
        for mutation in df['mutation'].values:
            for mut in mutation.split(';'):
                if mut.endswith('del'):
                    wt_aa = mut[0]
                    mt_aa = ''
                    pos = int(mut[1:-3])
                else:
                    wt_aa = mut[0]
                    mt_aa = mut[-1]
                    pos = int(mut[1:-1])
                valid = True if wt_aa == self.native_sequence[pos - 1] else False
                valid = valid and (mt_aa not in ['X'])
            flags.append(valid)
        df = df[flags].reset_index(drop=True)
        return df

    def _read_mutation_df(self, tsv):
        df = pd.read_table(tsv)

        if self.skip_bad_lines:
            df_filter = df[df[self.label_column].notnull()]
            logger.info(f'Skip bad lines on, {len(df)} -> {len(df_filter)}')
            df = df_filter

        if 'mutated_sequence' in df.columns:
            df['sequence'] = df['mutated_sequence']

        if self.label_column != '' and self.label_column in df.columns:
            logger.info('Use label column:' + self.label_column)
            df['score'] = df[self.label_column]
            return df
        elif 'sequence' in df.columns:
            return df
        else:
            df = self._drop_invalid_mutation(df)
            df['sequence'] = self._mutations_to_sequences(df['mutation'].values)
            return df

    def encode_seq_enc(self, sequences):
        seq_enc = index_encoding(sequences)
        seq_enc = torch.from_numpy(seq_enc.astype(np.int))
        return seq_enc

    def encode_sequence(self, sequences, length=0):
        feat = self.lm_encoder.encode(sequences, length)
        return feat

    def build_data(self, mode, return_df=False):
        if mode == 'train':
            df = self.train_df.copy()
        elif mode == 'valid':
            df = self.valid_df.copy()
        elif mode == 'test':
            df = self.test_df.copy()
        elif mode == 'zeroshot':
            df = self.full_df.copy()
        else:
            raise NotImplementedError()

        sequences = df['sequence'].values
        
        maxlen_native = len(self.native_sequence)
        data_maxlen = np.max([len(seq) for seq in sequences], 0)
        length = data_maxlen + 2 if maxlen_native <= data_maxlen else maxlen_native + 2
        rec = self.encode_sequence(sequences, length=length)
        
        if 'score' not in df.columns:
            logger.warning(f'No score column found, filling with 0s.')
            df['score'] = 0.0
        labels = df['score'].values
        labels = torch.from_numpy(labels.astype(np.float32))
        samples = []
        for i in range(len(df)):
            sample = {
                'sequence':sequences[i],
                'label':labels[i],
                # 'seq_enc': seq_enc[i],
            }
            sample['x'] = rec['x'][i]
            sample['mask'] = rec['mask'][i]
            samples.append(sample)

        sequence = self.native_sequence
        rec = self.encode_sequence(np.array([self.native_sequence]), length=rec['x'].shape[1])
        native_sample = {
            'sequence':sequence,
            'x': rec['x'][0],
            }
            
        if self.label_oper=="divide":
            native_sample['label'] = torch.tensor(1.0)
        elif self.label_oper =="minus":
            native_sample['label'] = torch.tensor(0.0)

        native_sample['mask'] = rec['mask'][0]
        
        data = MetagenesisData(samples,mode,
            native_sample,
            fasta=self.fasta,
            label_oper=self.label_oper,
            aug_ratio=self.aug_ratio)
            
        if return_df:
            return data, df
        else:
            return data

    def get_dataloader(self, mode, batch_size=128,
            return_df=False, resample_train_valid=False, num_workers=1, shuffle=False):
        if os.cpu_count() < num_workers:
            num_workers = 2
        else:
            num_workers = 4
        
        if resample_train_valid and self.can_resample:
            self.train_df, self.valid_df, _ = \
                self._split_dataset_df(
                    self.train_valid_df, self.split_ratio[:2], resample_split=True)
                
        if shuffle:
            # logger.info('Shuffle..')
            self.train_df = self.train_df.sample(frac=1.0)
            
        if mode == 'train_valid':
            train_data, train_df = self.build_data('train', return_df=True)
            valid_data, valid_df = self.build_data('valid', return_df=True)
            if self.distributed:
                self.sampler_train = torch.utils.data.DistributedSampler(train_data)
            else:
                self.sampler_train = torch.utils.data.RandomSampler(train_data)
            batch_sampler_train = torch.utils.data.BatchSampler(self.sampler_train, batch_size=batch_size, drop_last=True)
            train_loader = torch.utils.data.DataLoader(train_data, batch_sampler=batch_sampler_train, pin_memory=False, num_workers=num_workers)
            # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, pin_memory=False, drop_last=True)
            valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, pin_memory=False)
            if return_df:
                return (train_loader, train_df), (valid_loader, valid_df)
            else:
                return train_loader, valid_loader
        elif mode == 'test':
            test_data, test_df = self.build_data('test', return_df=True)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=False)
            if return_df:
                return test_loader, test_df
            else:
                return test_loader
        elif mode == 'zeroshot':
            test_data, test_df = self.build_data('zeroshot', return_df=True)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=False)
            if return_df:
                return test_loader, test_df
            else:
                return test_loader
        else:
            raise NotImplementedError

