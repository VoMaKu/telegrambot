import numpy as np # generate random
import os          # get current directory path
import subprocess  # execute ffmpeg
import telebot     # run telegram bot
from scipy.io.wavfile import read, write
import time
import librosa

from datetime import datetime # generate log
from split_by_vad import sec2samples, get_segments_energy, get_vad_mask, mask_compress  

import pickle
# @audio_digits_dataset_bot
# Get a token from @BotFather and put it in the environment:
#     export TG_RECOGNITION_BOT_TOKEN="your token"
bot = telebot.TeleBot(os.environ.get("TG_RECOGNITION_BOT_TOKEN", "YOUR TOKEN HERE"))
root = os.getcwd() + "/inference"
# Git does not store empty directories, so a fresh clone has none of these.
for folder in ("ogg", "wav", "splitted"):
	os.makedirs(root + "/" + folder, exist_ok=True)
filename = "model.pkl"
if not os.path.exists(filename):
	# The model is not kept in the repository, it is trained from the dataset.
	print(f"{filename} not found. Run 'make training' to build it from dataset/splitted_final.")
	exit(1)
with open(filename, 'rb') as f:
	model_pickled = f.read()
model = pickle.loads(model_pickled)


class BadRecording(Exception):
	"""Raised when the voice message does not hold exactly one clear digit."""


def save_ogg(ogg_data, ogg_path):
	with open(ogg_path, "wb") as file:
		file.write(ogg_data)


def convert_ogg_wav(ogg_path, dst_path):
	rate = 48000
	cmd = f"ffmpeg -i {ogg_path} -ar {rate} {dst_path} -y -loglevel panic"
	log(cmd)
	with subprocess.Popen(cmd.split()) as p:
		try:
			p.wait(timeout=2)
		except subprocess.TimeoutExpired:
			p.kill()
			p.wait()
			return "timeout"



def log(text):
	time_stamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
	print(time_stamp + " " + text)

def vad(wav_file_path, user):
	segment_duration = 0.1
	vad_threshold = 0.005
	sample_rate, audio = read(wav_file_path)
	segment_duration_samples = sec2samples(segment_duration, sample_rate)
	segments_energy = get_segments_energy(audio, segment_duration_samples)
	vad_mask = get_vad_mask(segments_energy, vad_threshold)
	segments = mask_compress(vad_mask)
	if len(segments) != 1:
		raise BadRecording(f"I heard {len(segments)} sounds instead of one digit. Please record a single digit in a quiet room.")

	max_duration = 0
	min_duration = 1
	for segment in segments:
		duration = (segment.stop - segment.start) * segment_duration_samples / sample_rate
		if duration > max_duration:
			max_duration = duration
		if duration < min_duration:
			min_duration = duration
	if max_duration > 0.8:
		raise BadRecording("That sounded too long for a single digit. Please try again.")
	if min_duration < 0.1:
		raise BadRecording("That sounded too short for a single digit. Please try again.")
	wav_path_after_vad = root + f"/splitted/unk_{user}.wav"
	start = segment.start * segment_duration_samples
	stop = segment.stop * segment_duration_samples
	write(wav_path_after_vad, sample_rate, audio[start:stop])
	return wav_path_after_vad

def predict(wav_path_after_vad, user):
	file_path = root + f"/splitted/unk_{user}.wav"
	sample_rate, audio = read(file_path)

	max_duration_sec = 0.8
	max_duration = int(max_duration_sec * sample_rate + 1e-6)
	if len(audio) < max_duration:
		audio = np.pad(audio, (0, max_duration - len(audio)), constant_values=0)
	feature = librosa.feature.melspectrogram(y=audio.astype(float), sr=sample_rate, n_mels=32, fmax=4096)
	features_flatten = feature.reshape(-1)

	answer = model.predict([features_flatten])[0]
	return answer


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    user = message.from_user.id
    text = message.text
    log(f"User ({user}): {text}")

    bot.send_message(user,
        "Please send a voice message with a digit.")


@bot.message_handler(content_types=['voice'])
def get_voice_messages(message):
	user = message.from_user.id
	voice = message.voice
	log(f"User ({user}): voice")

	tele_file = bot.get_file(voice.file_id)
	ogg_data = bot.download_file(tele_file.file_path)
	file_name = f"unk_{user}" # need to generate uniq name
	ogg_path = root + "/ogg/" + file_name + ".ogg"
	wav_path = root + "/wav/" + file_name + ".wav"
	save_ogg(ogg_data, ogg_path)
	if convert_ogg_wav(ogg_path, wav_path) == "timeout":
		bot.send_message(user, "Converting your recording took too long. Please try again.")
		return
	try:
		wav_path_after_vad = vad(wav_path, user)
		answer = predict(wav_path_after_vad, user)
	except BadRecording as e:
		bot.send_message(user, str(e))
		return
	bot.send_message(user, "You said " + str(answer))


while True:
	try:
		bot.polling(none_stop=True, interval=0)
	except KeyboardInterrupt:
		exit(0)
	except Exception as e:
		print(e)
	time.sleep(5)
	print("LOOP")
	print("pid", os.getpid())