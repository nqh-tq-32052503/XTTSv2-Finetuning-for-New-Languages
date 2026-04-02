import torch
import torchaudio
from tqdm import tqdm
import os
from underthesea import sent_tokenize

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"

class TTSInference(object):
    def __init__(self, checkpoint_path, vocab_path, config_path):
        self.xtts_checkpoint = checkpoint_path
        self.xtts_config = config_path
        self.xtts_vocab = vocab_path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        self.config = XttsConfig()
        self.config.load_json(self.xtts_config)
        self.XTTS_MODEL = Xtts.init_from_config(self.config)
        self.XTTS_MODEL.load_checkpoint(self.config, checkpoint_path=self.xtts_checkpoint, vocab_path=self.xtts_vocab, use_deepspeed=False)
        self.XTTS_MODEL.to(self.device)
        self.lang = "vi"
        print("Model loaded successfully!")

    def infer(self, text, audio_ref, output_path):
        gpt_cond_latent, speaker_embedding = self.XTTS_MODEL.get_conditioning_latents(
            audio_path=audio_ref,
            gpt_cond_len=self.XTTS_MODEL.config.gpt_cond_len,
            max_ref_length=self.XTTS_MODEL.config.max_ref_len,
            sound_norm_refs=self.XTTS_MODEL.config.sound_norm_refs,
        )
        tts_texts = sent_tokenize(text)

        wav_chunks = []
        for text in tts_texts:
            wav_chunk = self.XTTS_MODEL.inference(
                text=text,
                language=self.lang,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.1,
                length_penalty=1.0,
                repetition_penalty=10.0,
                top_k=10,
                top_p=0.3,
            )
            wav_chunks.append(torch.tensor(wav_chunk["wav"]))

        out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()

        torchaudio.save(output_path, out_wav, 22050)

    def infer_batch(self, audio_folder, metadata_file, dest_folder):
        with open(metadata_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert os.path.exists(audio_folder), "[ERROR] Audio folder {0} not exists".format(audio_folder)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        for line in tqdm(lines):
            audio_name, transcript = line.split(",")
            transcript = transcript.replace("\n", "")
            audio_ref = os.path.join(audio_folder, audio_name)
            assert os.path.exists(audio_ref), "[ERROR] Audio path {0} not exists".format(audio_ref)
            output_path = os.path.join(dest_folder, audio_name)
            self.infer(transcript, audio_ref, output_path)
        
        print("Test successfully!")

if __name__ == "__main__":
    checkpoint_path = "checkpoints/GPT_XTTS_FT-April-01-2026_10+20AM-3240401/best_model_77350.pth"
    config_path = "checkpoints/GPT_XTTS_FT-April-01-2026_10+20AM-3240401/config.json"
    vocab_path = "checkpoints/XTTS_v2.0_original_model_files/vocab.json"
    obj = TTSInference(checkpoint_path=checkpoint_path, vocab_path=vocab_path, config_path=config_path)
    audio_folder = "data_4-5s/wavs"
    metadata_file = "data_4-5s/metadata_test.csv"
    dest_folder = "outputs_best_model_77350"
    obj.infer_batch(audio_folder=audio_folder, metadata_file=metadata_file, dest_folder=dest_folder)