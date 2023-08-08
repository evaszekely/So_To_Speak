import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow as tf
from .text.symbols import symbols

class HParams(object):
    hparamdict = []
    def __init__(self, **hparams):
        self.hparamdict = hparams
        for k, v in hparams.items():
            setattr(self, k, v)
    def __repr__(self):
        return "HParams(" + repr([(k, v) for k, v in self.hparamdict.items()]) + ")"
    def __str__(self):
        return ','.join([(k + '=' + str(v)) for k, v in self.hparamdict.items()])
    def parse(self, params):
        for s in params.split(","):
            k, v = s.split("=", 1)
            k = k.strip()
            t = type(self.hparamdict[k])
            if t == bool:
                v = v.strip().lower()
                if v in ['true', '1']:
                    v = True
                elif v in ['false', '0']:
                    v = False
                else:
                    raise ValueError(v)
            else:
                v = t(v)
            self.hparamdict[k] = v
            setattr(self, k, v)
        return self

def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

#    hparams = tf.contrib.training.HParams(
    hparams = HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=50000,
        iters_per_checkpoint=5000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=True,
        dist_backend="nccl",
        dist_url="tcp://localhost:54218",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        #ignore_layers=['speaker_embedding.weight'],
        ignore_layers=[''],

        ################################
        # Additional Factor Parameters #
        ################################        
        # Prosodic feature embedding
        prosodic=True,
        feat_dim=2,
        feat_max_bg=1,
        # Speaker embedding
        speakers=True,
        n_speakers=2,
        speaker_embedding_dim=8,
        
        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        training_files='filelists/joe_tien_train_filelist.txt',
        validation_files='filelists/joe_tien_val_filelist.txt',
        #training_files='filelists/duo_train_filelist.txt',
        #validation_files='filelists/duo_val_filelist.txt',
        text_cleaners=['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=5e-6,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=28,
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_string:
        #tf.logging.info('Parsing command line hparams: %s', hparams_string)
        tf.compat.v1.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        #tf.logging.info('Final parsed hparams: %s', hparams.values())
        tf.compat.v1.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
