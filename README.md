# Spoken digit recognition over Telegram

Two Telegram bots and the pipeline between them. The first one collects a
dataset of spoken digits from voice messages; the second one takes a voice
message with a single digit and answers with the digit it heard.

Written for a second-year programming course.

```
voice message ──▶ dataset bot ──▶ VAD split ──▶ training ──▶ model.pkl ──▶ recognition bot
```

Everything in that chain runs at **48 kHz mono**. Both bots hand each incoming
voice message to `ffmpeg` and convert it to that rate before anything else, and
the model is fitted on the result, so the length of a feature vector follows
from the rate. Audio at any other rate produces a vector of a different length
that the model cannot accept.

## Setup

```sh
pip3 install -r requirements.txt
```

`ffmpeg` must be on the `PATH` as well — both bots shell out to it to convert
Telegram's `.ogg` voice messages into `.wav`.

The bots are yours to create: this repository ships no tokens, and the two bots
it was originally written against are gone. Make a pair with
[@BotFather](https://t.me/BotFather) and export their tokens before starting:

```sh
export TG_DATASET_BOT_TOKEN="your token"
export TG_RECOGNITION_BOT_TOKEN="your token"
```

## Collecting a dataset

```sh
make bot_dataset
```

Send the bot any message and it answers with five random digits to read aloud;
reply with a voice message saying them, one at a time, with a pause between
each. The recording is stored under `dataset/ogg` and `dataset/wav`, named after
the digits it contains — `4_1_0_6_9.wav` — which is how the next step knows what
it is listening to. Five is the sequence length, set by `DIGITS_PER_TASK` in
`audio_digits_dataset_bot.py`; the rest of the pipeline reads the count back
from the file name and follows along.

## Splitting the recordings into single digits

```sh
make vad_sort     # voice activity detection: cut each recording into segments
make splitted     # move the segments into dataset/splitted_final and clean up
```

`split_by_vad.py` finds the segments by energy: it takes a window of 0.1 s with
a 0.005 s step, marks the parts above the energy threshold as speech and cuts on
the pauses between digits. Each segment is written to `dataset/splitted/<digit>/`
and, once you are happy with the split, moved into `dataset/splitted_final/`,
which is the directory training reads.

## Training

```sh
make training
```

`training.py` extracts features with `librosa` and fits an `MLPClassifier` with
hidden layers of 888, 777 and 666 neurons over ten classes. The result is
pickled into `model.pkl`. A `RandomForestClassifier` is left in the file,
commented out, as the alternative that was tried.

`model.pkl` is not kept in the repository — it weighs 77 MB, and every clone
would pay for it. Train your own; the dataset it is built from is committed, so
this works straight after a clone. Both the recognition bot and `inference.py`
say so plainly if the file is missing.

## Recognition

```sh
make bot_recognition
```

Send a voice message with one digit and the bot replies with its prediction.

## Recognising one file without Telegram

`inference.py` runs the same model over a single 48 kHz `.wav` that has already
been cut down to one digit:

```sh
python3 inference.py inference/splitted/unk.wav
```

## Repository layout

```
audio_digits_dataset_bot.py       collects recordings
audio_digits_recognition_bot.py   answers with a prediction
split_by_vad.py                   voice activity detection and segmentation
training.py                       feature extraction and model fitting
inference.py                      prediction on a single file, without Telegram
dataset/splitted_final/           the labelled digits training reads
model.pkl                         built by 'make training', not tracked
```
