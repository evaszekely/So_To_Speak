from .model import Tacotron2
from .hparams import create_hparams

def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model

if __name__ == '__main__':
    hparams = create_hparams(args.hparams)
