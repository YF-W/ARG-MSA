import logging
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = ['MMDataLoader']

logger = logging.getLogger('MSA')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATASET_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
            'simsv2':self.__init_simsv2,
        }
        DATASET_MAP[args['dataset_name']]()

    def __init_mosi(self):
        if self.args['custom_feature']:
            # use custom feature file extracted with MSA-FET
            with open(self.args['custom_feature'], 'rb') as f:
                data = pickle.load(f)
        else:
            # use deault feature file specified in config file
            with open(self.args['featurePath'], 'rb') as f:
                data = pickle.load(f)
        
        if self.args.get('use_bert', None):
            self.text = data[self.mode]['text_bert'].astype(np.float32)
            self.args['feature_dims'][0] = 768
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
            self.args['feature_dims'][0] = self.text.shape[2]
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.args['feature_dims'][1] = self.audio.shape[2]
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.args['feature_dims'][2] = self.vision.shape[2]
        self.raw_text = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        # Overide with custom modality features
        if self.args['feature_T']:
            with open(self.args['feature_T'], 'rb') as f:
                data_T = pickle.load(f)
            if self.args.get('use_bert', None):
                self.text = data_T[self.mode]['text_bert'].astype(np.float32)
                self.args['feature_dims'][0] = 768
            else:
                self.text = data_T[self.mode]['text'].astype(np.float32)
                self.args['feature_dims'][0] = self.text.shape[2]
        if self.args['feature_A']:
            with open(self.args['feature_A'], 'rb') as f:
                data_A = pickle.load(f)
            self.audio = data_A[self.mode]['audio'].astype(np.float32)
            self.args['feature_dims'][1] = self.audio.shape[2]
        if self.args['feature_V']:
            with open(self.args['feature_V'], 'rb') as f:
                data_V = pickle.load(f)
            self.vision = data_V[self.mode]['vision'].astype(np.float32)
            self.args['feature_dims'][2] = self.vision.shape[2]

        self.labels = {
            # 'M': data[self.mode][self.args['train_mode']+'_labels'].astype(np.float32)
            'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)
        }
        if 'sims' in self.args['dataset_name']:
            for m in "TAV":
                self.labels[m] = data[self.mode]['regression' + '_labels_' + m].astype(np.float32)

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not self.args['need_data_aligned']:
            # TFN averages audio/video over time before its forward pass and
            # never consumes sequence lengths. Some released MOSEI feature
            # files omit that metadata; preserve the features without inventing
            # lengths or changing the alignment setting.
            lengths_optional = (
                self.args.get('model_name') == 'tfn'
                and self.args.get('need_normalized', False)
            )
            if self.args['feature_A']:
                audio_source = data_A[self.mode]
            else:
                audio_source = data[self.mode]
            if self.args['feature_V']:
                vision_source = data_V[self.mode]
            else:
                vision_source = data[self.mode]
            if not lengths_optional or 'audio_lengths' in audio_source:
                self.audio_lengths = audio_source['audio_lengths']
                if self.args['feature_A']:
                    self.audio_lengths = list(self.audio_lengths)
            if not lengths_optional or 'vision_lengths' in vision_source:
                self.vision_lengths = vision_source['vision_lengths']
                if self.args['feature_V']:
                    self.vision_lengths = list(self.vision_lengths)
        self.audio[self.audio == -np.inf] = 0


        if self.args.get('need_normalized'):
            self.__normalize()
    
    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()
    
    def __init_simsv2(self):
        return self.__init_mosi()



    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # For visual and audio modality, we average across time:
        # The original data has shape (max_len, num_examples, feature_dim)
        # After averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if 'use_bert' in self.args and self.args['use_bert']:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'raw_text': self.raw_text[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 
        if not self.args['need_data_aligned']:
            if hasattr(self, 'audio_lengths'):
                sample['audio_lengths'] = self.audio_lengths[index]
            if hasattr(self, 'vision_lengths'):
                sample['vision_lengths'] = self.vision_lengths[index]

        return sample

def MMDataLoader(args, num_workers):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args['seq_lens'] = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['batch_size'],
                       num_workers=num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    
    return dataLoader
