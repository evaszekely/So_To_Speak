# So-to-Speak: an exploratory platform for investigating the interplay between style and prosody in TTS
Interspeech 2023 Demonstration

[tacotron2_link]: https://github.com/NVIDIA/tacotron2
[hifigan_link]: https://github.com/jik876/hifi-gan
[automos_link]: https://github.com/nii-yamagishilab/mos-finetune-ssl


We introduce So-to-Speak, a customisable interface tailored for showcasing the capabilities of different controllable TTS systems. The interface allows for the generation, synthesis, and playback of hundreds of samples simultaneously, displayed on an interactive grid, with variation both low level prosodic features and high level style controls. To offer insights into speech quality, automatic estimates of MOS scores are presented for each sample. So-to-Speak facilitates the audiovisual exploration of the interaction between various speech features, which can be useful in a range of applications in speech technology.

## Acknowledgements
The code implementation is using an adaptation of [Nvidia's implementation of Tacotron 2][tacotron2_link] as a synthesis engine, [HiFi-GAN][hifigan_link] as vocoder, and code and the trained model for the [automatic MOS predictor][automos_link].


## Setup
### synthesis model and vocoder
Pretrained models for the synthesis model and the HiFi-GAN model are provided in the release and should be placed in the models/tronduo and models/hifigan folders respectively.

### Automatic MOS predictor
Run the script `run_inference.py` from the [repository][automos_link] should download both a base wav2vec model and a fine-tuned version, both of which are required to run MOS prediction script here. Both should be placed in the models folder.

### Start
Run the code in the Jupyter Notebook SoToSpeak_launch_interface.ipynb to start the demo; CUDA required.
