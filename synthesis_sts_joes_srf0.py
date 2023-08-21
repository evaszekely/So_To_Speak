# packages used
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import IPython.display as ipd
import torch
from g2p_en import G2p
import re
import pandas as pd
import librosa
import ipywidgets as widgets
from torch import nn
from scipy.io import wavfile
from IPython.display import display, HTML
display(HTML("<style>div.output_scroll { height: 42em; }</style>"))
g2p = G2p()
import os
import json
import math
import sys
from tqdm.notebook import tqdm

# Tacotron2
from tronduo.model import Tacotron2
from tronduo.layers import TacotronSTFT, STFT
from tronduo.model_util import load_model
from tronduo import text_to_sequence

# HiFi-GAN
from hifigan.env import AttrDict
from hifigan.models import Generator
from tronduo.hifigandenoiser import Denoiser
MAX_WAV_VALUE = 32768.0
device = 'cuda'

# ssl_mos
mos_dir = "./models/mos_ssl/"
#sys.path.append(mos_dir)
from mos_ssl.infer_util import *


# load generic parameters
from tronduo.hparams import create_hparams
hparams = create_hparams()
hparams.global_mean = None
hparams.distributed_run = False
hparams.feat_max_bg = 4
hparams.speaker_embedding_dim = 8


# define model settings
class Config:
    def __init__(self):
        # Tacotron
        self.tacotron_checkpoint_path = 'models/tronduo/'
        self.tacotron_iterations = '100000'
        # prosodic control
        self.prosodic = True
        self.feat_dim = 2 # nr of prosodic features controlled
        self.prosodic_factors = ['speech rate', 'pitch'] # nr of items should match feat_dim
        # speaker embedding
        self.speakers = True
        self.n_speakers= 2
        self.speaker_ids = ['read speech', 'spontaneous'] # nr of items should match n_speakers
        
        # HiFi-GAN
        self.hifigan_checkpoint_path = 'models/hifigan/'
        self.hifigan_iterations = 3860000


# grapheme to phoneme preparation
def preptext(b):
    txt = re.sub('[\!]+',',',startdict['intext'].value)
    txt = re.sub('-',' ',txt)
    txt = re.sub(';','-',txt)
    txt = re.sub('\|','. ',txt)
    phon = g2p(txt)
    for j, n in enumerate(phon):
        if n == ' ':
            phon[j] = '} {'
    transcript = '{ '+' '.join(phon)+' }'
    transcript = re.sub(r' ?{ ?- ?} ?',';', transcript)
    transcript = re.sub(r' ?{ ?, ?} ?',',', transcript)
    transcript = re.sub(r' ?{ ?\. ?} ?','.', transcript)
    transcript = re.sub(r' ?{ ?\? ?} ?','?', transcript)
    transcript = re.sub(r'{ ?','{', transcript)
    transcript = re.sub(r' ?}','}', transcript)
    if transcript.strip()[-1:] == '}':
        transcript = transcript.strip()+'.'
    b.transcript = transcript

def showgrid(s, spk_steps, sp_steps, f0_steps):
    size = boxdict['feat_steps'].value # number of items on each row/column
    start = int(sp_steps[0]*100) # first setting (%)
    step = int((sp_steps[1]-sp_steps[0])*100) # distance between settings (%)
    high = "100%"
    res = results[results['rd'] == spk_steps[s]*100]
    res = res.reset_index(drop=True)

    # create filelist
    filelist = res["Filename"]
    # create buttons
    buttons = [widgets.Button(description=f"{np.round(res['ssl_mos'][i],1)}", tooltip=f"{i+s*size*size:03}_{filelist[i]}", 
                              layout=widgets.Layout(width="100%", height=high), 
                              button_style='').add_class('myclass') for i in range(size*size)]
    for i, button in enumerate(buttons):
        button.style.button_color = set_colour(0, 5, res['ssl_mos'][i], [173, 255, 47], [255, 69, 0])
        button.on_click(play_sound2)
    # create the grid
    grid_out = widgets.Output(layout={"display": "flex", "flex_flow": "row wrap", "align_items": "flex-start", "margin": "0"})
    for i in range(size):
        grid = widgets.GridBox([widgets.Button(description=f"{str(start+(size-i-1)*step)}", 
                                               layout=widgets.Layout(width="100%", height=high)).add_class('myclass')]\
                               + buttons[(size-1-i)*size:(size-i)*size],
                               layout=widgets.Layout(grid_template_columns=f"repeat({size+1},1fr)"), grid_gap='0px 0px')
        grid_out.append_display_data(grid)
    # add x-axis
    xax = [widgets.Button(description=f"{str(x)}", layout=widgets.Layout(width="100%", height=high)).add_class('myclass') for x in range(start,start+size*step,step)]
    grid = widgets.GridBox([widgets.Button(description='sr/f0', layout=widgets.Layout(width="150%", height=high)).add_class('myclass')]\
                               +xax, layout=widgets.Layout(grid_template_columns=f"repeat({size+1},1fr)"), grid_gap='0px 0px')
    grid_out.append_display_data(grid)
    samples.clear_output()
    with samples:
        print(f'Conversational vs Read Speech - [{1-spk_steps[s]:.0%},{(spk_steps[s]):.0%}]:')
        display(grid_out)


def synth(b):
    samples.clear_output()
    with samples:
        display(widgets.HTML(value=f"<b>Output:</b>"))                                                           
    sequence = np.array(text_to_sequence(boxdict['trns'].value, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).to(device).long()
    spk_steps = np.linspace(boxdict['speakers'].value[0], boxdict['speakers'].value[1], boxdict['spk_steps'].value)
    sp_steps = np.linspace(boxdict['speechrate'].value[0], boxdict['speechrate'].value[1], boxdict['feat_steps'].value)
    f0_steps = np.linspace(boxdict['pitch'].value[0], boxdict['pitch'].value[1], boxdict['feat_steps'].value)
    global results
    results = pd.DataFrame(columns = ["wav", "Filename", "rd", "sp", "sr_in", "f0_in", "dur", "ssl_mos"])

    # generate samples
    total_iterations=boxdict['spk_steps'].value*boxdict['feat_steps'].value*boxdict['feat_steps'].value
    print(f'Generating {total_iterations} samples...')
    pbar = tqdm.tqdm(total=total_iterations)
    with torch.no_grad():
        for spk in spk_steps:
            for j in sp_steps:
                for k in f0_steps:
                    speaks = torch.as_tensor([max(spk,0.2), 1-spk]).unsqueeze(0).to(device)
                    pros = torch.as_tensor([j,k]).unsqueeze(0).half().to(device)
                    durat = 1000
                    cnt = 0
                    while durat > 890 and cnt < 3:
                        try:
                            _, mel_outputs_postnet, _, _ = model.inference(sequence, speaks=speaks, pros=pros)
                            durat = mel_outputs_postnet[0].size()[1]
                            cnt += 1
                        except:
                            pass
                    melfl = mel_outputs_postnet.float()
                    y_g_hat = generator(melfl)
                    audio = denoiser(y_g_hat[0], strength=0.015).squeeze().half()
                    audio_out = audio.cpu().detach().numpy()
                    # generate output
                    pbar.update(1)
                    filename = f'RD{int(100*spk):04}SP{int(100*(1-spk)):04}_sr_{int(100*j):04}_f0_{int(100*k):04}'
                    results = results.append({"wav":audio_out, "Filename":filename, 
                                              "rd":100*spk,"sp":100*(1-spk),
                                              "sr_in":np.round(j,2),"f0_in":np.round(k,2),
                                              "dur":np.round(len(audio_out)/hparams.sampling_rate,3)}, ignore_index = True)
    pbar.close()
                                                           
    # run ssl_mos
    print(f'Running evaluation...')
    for i in tqdm.tqdm(range(len(results))):
        audio = torch.tensor(results["wav"][i], dtype=torch.float32).to(device).unsqueeze(0)
        audio = torchaudio.functional.resample(audio, hparams.sampling_rate, 16000)
        with torch.no_grad():
            results.at[i, "ssl_mos"] = mos_predictor(audio).cpu().detach().numpy()[0]
    
    # create slider for style selection
    gridselect.clear_output()
    with gridselect:
        label = 'Conversational'
        gridslider = widgets.IntSlider(m=0, max=len(spk_steps)-1, value=0,
                                                      description='', disabled=False,
                                                      continuous_update=False, orientation='horizontal',
                                                      readout=False, layout=widgets.Layout(width='300px'))
        gridslider.observe(lambda change: showgrid(change.new, spk_steps, sp_steps, f0_steps), names='value')
        gridbox = widgets.HBox([widgets.Label(value='Conversational Speech'), gridslider, widgets.Label(value='Read Speech')])
        display(gridbox)
        
    showgrid(0, spk_steps, sp_steps, f0_steps)


def synth_settings(b):
    global boxdict
    styleselect.clear_output()
    text1 = widgets.HTML
    boxdict = {'txt0':widgets.HTML(value=f"Transcript:"),
               'trns': widgets.Textarea(value=startdict['prep'].transcript, placeholder='Transcript', 
                      description='', disabled=False, layout=widgets.Layout(width='600px')),
               'txt1': widgets.HTML(value="Text:"),
               'txt1b': widgets.HTML(value=f"<b>{startdict['intext'].value}</b>"),
               'txt2': widgets.HTML(value="Nr. of grids: "),
               'spk_steps': widgets.Dropdown(value=3, options=[2,3,4,5,6,7,8,9], 
                                              description='', disabled=False, continuous_update=False,
                                              #orientation='horizontal', readout=True, readout_format='d',
                                             layout=widgets.Layout(width='50px')),
               'txt3': widgets.HTML(value="%Read Speech:"),
               'speakers': widgets.FloatRangeSlider(value=[0., 1.], min=-0.5, max=1.5, step=0.05,
                                                    description='',
                                                    disabled=False, continuous_update=False,
                                                    orientation='horizontal', readout=True,
                                                    readout_format='.0%', layout=widgets.Layout(width='600px')),
               'txt4': widgets.HTML(value="Nr. of features:"),
               'feat_steps': widgets.Dropdown(value=5, options=[2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
                                              description='', disabled=False, continuous_update=False,
                                              #orientation='horizontal', readout=True, readout_format='d',
                                             layout=widgets.Layout(width='50px')),
               'txt5': widgets.HTML(value="Speech Rate:"),
               'speechrate': widgets.FloatRangeSlider(value=[-2.0, 2.0], min=-3.0, max=3.0, step=0.1,
                                                    description='',
                                                    disabled=False, continuous_update=False,
                                                    orientation='horizontal', readout=True,
                                                    readout_format='.1f', layout=widgets.Layout(width='600px')),
               'txt6': widgets.HTML(value="Pitch:"),
               'pitch': widgets.FloatRangeSlider(value=[-2.0, 2.0], min=-3.0, max=3.0, step=0.1,
                                                    description='',
                                                    disabled=False, continuous_update=False,
                                                    orientation='horizontal', readout=True,
                                                    readout_format='.1f', layout=widgets.Layout(width='600px')),
               'txt7': widgets.HTML(value=""),
              }
    boxdict['synth'] = widgets.Button(description='Generate Speech', disabled=False,
        button_style='success', tooltip='Synthesize using selected feature value ranges', icon='check')
    boxdict['synth'].on_click(synth)
    column_widths = ['1fr', '4fr']
    sb = widgets.GridBox(children=[boxdict[x] for x in boxdict],
                              layout=widgets.Layout(grid_template_columns=' '.join(column_widths),
                                                    grid_template_rows='repeat(8, auto)'))
    with styleselect:
        display(sb)

def load_tacotron(hparams, config):
    model = load_model(hparams)
    checkpoint_path = config.tacotron_checkpoint_path + "checkpoint_" + config.tacotron_iterations
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval().half()
    return model


def load_hifigan(config, device):
    config_file = config.hifigan_checkpoint_path + 'config.json'
    checkpoint_file = config.hifigan_checkpoint_path + 'g_' + str(config.hifigan_iterations).zfill(8)
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    generator = Generator(h).to(device)
    state_dict_g = torch.load(checkpoint_file, map_location=device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator

# grid support functions
def play_sound2(b):
    i = int(b.tooltip[:3])
    audios.clear_output(wait=True)
    with audios:
        display(ipd.Audio(results["wav"][i], rate=hparams.sampling_rate, autoplay=True))

def set_colour(mn, mx, value, mn_colour, mx_colour):
    if np.isnan(value):
        col = 'rgb(250, 250, 210)'
    else:
        perc = (value-mn)/(mx-mn)
        col = f'rgb({int((1-perc)*mx_colour[0]+perc*mn_colour[0])},{int((1-perc)*mx_colour[1]+perc*mn_colour[1])},{int((1-perc)*mx_colour[2]+perc*mn_colour[2])})'
    return col                                                           

def start():
    #load parameter settings
    global config
    config = Config()
    hparams.prosodic = config.prosodic
    hparams.speakers = config.speakers
    hparams.feat_dim = config.feat_dim
    hparams.n_speakers= config.n_speakers
    
    global model
    global generator
    global denoiser
    global mos_predictor
    global speaker_embedding
    global speaker0
    global speaker1
    speaker_embedding = nn.Embedding(
                hparams.n_speakers, hparams.speaker_embedding_dim).to(torch.device('cuda:0')).half()
    speaker0 = speaker_embedding(torch.as_tensor(0).unsqueeze(0).cuda())[:, None]
    speaker1 = speaker_embedding(torch.as_tensor(1).unsqueeze(0).cuda())[:, None]
    
    model = load_tacotron(hparams, config)
    generator = load_hifigan(config, device)
    denoiser = Denoiser(generator, mode='zeros') # The other mode is normal
    
    mos_predictor = get_mos_model(cp_path = mos_dir+"fairseq/wav2vec_small.pt" , my_checkpoint = mos_dir+"pretrained/ckpt_w2vsmall", device=device)
    mos_predictor.eval()

    # set layout
    global box_layout_2
    box_layout_2 = widgets.Layout(display='flex',
                    flex_flow='column', 
                    align_items='center',
                    justify_content='center',
                    #border='solid 3px palevioletred',
                    border='solid 3px gainsboro',
                    width='80%')
    
    # create starting point
    global startpoint
    global startdict
    global styleselect
    global gridselect
    global audios
    global samples
    startdict = dict()
    startdict['label2'] = widgets.HTML(value=f"<b>Enter text to synthesise:</b>")
    startdict['intext'] = widgets.Textarea(value='', placeholder='', 
                                           description='Input Text:', disabled=False, 
                                           layout=widgets.Layout(width='600px'))
    startdict['prep'] = widgets.Button(description='Preprocess Text', disabled=False,
                                        button_style='success', tooltip='G2P and prepare style units', icon='check')
    startdict['prep'].on_click(preptext)
    startdict['prep'].on_click(synth_settings)
    startpoint = widgets.VBox([startdict[name] for name in startdict])
    display(startpoint)
    lnk0 = widgets.link((startdict['intext'], 'value'), (startdict['prep'], 'tooltip'))
    styleselect = widgets.Output(layout=box_layout_2)
    display(styleselect)
    gridselect = widgets.Output(layout=box_layout_2)
    display(gridselect)
    samples = widgets.Output(layout=box_layout_2)
    display(samples)
    audios = widgets.Output(layout=box_layout_2)
    display(audios)

