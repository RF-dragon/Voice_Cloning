import os
import pathlib

import numpy as np
import paddle
import soundfile as sf
import yaml
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.t2s.modules.normalizer import ZScore
from yacs.config import CfgNode

text = "text.txt"
output_dir = "output"
vl_dir = "voice library"
config = "Voice Cloning.yaml"

with open(text) as f:
    text = f.read()
with open(config) as f:
    config = CfgNode(yaml.safe_load(f))
with open(config.am_config) as f:
    am_config = CfgNode(yaml.safe_load(f))
with open(config.voc_config) as f:
    voc_config = CfgNode(yaml.safe_load(f))

output_dir = pathlib.Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
vl_dir = pathlib.Path(vl_dir)

with open(config.phones_dict, "r") as f:
    phn_id = [line.strip().split() for line in f.readlines()]
vocab_size = len(phn_id)
am_name = config.am[:config.am.rindex('_')]
am_class = dynamic_import(am_name, config.model_alias)
am_inference_class = dynamic_import(am_name + '_inference', config.model_alias)
am = am_class(idim=vocab_size,
              odim=am_config.n_mels,
              spk_num=None,
              **am_config["model"])
am.set_state_dict(paddle.load(config.am_ckpt)["main_params"])
am.eval()
am_mu, am_std = np.load(config.am_stat)
am_mu = paddle.to_tensor(am_mu)
am_std = paddle.to_tensor(am_std)
am_normalizer = ZScore(am_mu, am_std)
am_inference = am_inference_class(am_normalizer, am)
am_inference.eval()

voc_name = config.voc[:config.voc.rindex('_')]
voc_class = dynamic_import(voc_name, config.model_alias)
voc_inference_class = dynamic_import(voc_name + '_inference',
                                     config.model_alias)
voc = voc_class(**voc_config["generator_params"])
voc.set_state_dict(paddle.load(config.voc_ckpt)["generator_params"])
voc.remove_weight_norm()
voc.eval()
voc_mu, voc_std = np.load(config.voc_stat)
voc_mu = paddle.to_tensor(voc_mu)
voc_std = paddle.to_tensor(voc_std)
voc_normalizer = ZScore(voc_mu, voc_std)
voc_inference = voc_inference_class(voc_normalizer, voc)
voc_inference.eval()

frontend = paddle.load("frontend.fe")

input_ids = frontend.get_input_ids(text)
for name in os.listdir(vl_dir):
    utt_id = name.split(".")[0]
    spk_emb = paddle.load(str(vl_dir / name))
    with paddle.no_grad():
        wav = voc_inference(
            am_inference(input_ids["phone_ids"][0], spk_emb=spk_emb))
    sf.write(str(output_dir / (utt_id + ".wav")), wav.numpy(), am_config.fs)
    print(f"{utt_id} done!")