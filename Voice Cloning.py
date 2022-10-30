import os
from pathlib import Path

import paddle
import yaml
from paddlespeech.vector.exps.ge2e.audio_processor import SpeakerVerificationPreprocessor
from paddlespeech.vector.models.lstm_speaker_encoder import LSTMSpeakerEncoder
from yacs.config import CfgNode

config = "Voice Cloning.yaml"
input_dir = "input"
vl_dir = "voice library"
vl_dir = Path(vl_dir)
vl_dir.mkdir(parents=True, exist_ok=True)
with open(config) as f:
    config = CfgNode(yaml.safe_load(f))
p = SpeakerVerificationPreprocessor(16000, -30, 30, 8, 6, 25, 10, 40, 160)
print("Audio Processor Done!")
speaker_encoder = LSTMSpeakerEncoder(40, 3, 256, 256)
speaker_encoder.set_state_dict(paddle.load(config.ge2e_params_path))
speaker_encoder.eval()
print("GE2E Done!")
for name in os.listdir(input_dir):
    utt_id = name.split(".")[0]
    os.system('python -m spleeter separate -p spleeter:2stems -o input "' +
              os.path.join(input_dir, name) + '"')
    wav = p.preprocess_wav(os.path.join(input_dir, utt_id, "vocals.wav"))
    mel_sequences = p.extract_mel_partials(wav)
    with paddle.no_grad():
        spk_emb = speaker_encoder.embed_utterance(
            paddle.to_tensor(mel_sequences))
        paddle.save(spk_emb, str(vl_dir / (utt_id + ".voice")))
    print(f"{utt_id} done!")