# Spoken digit recognition over Telegram

Two Telegram bots and the pipeline between them. The first one collects a
dataset of spoken digits from voice messages; the second one takes a voice
message with a single digit and answers with the digit it heard.

Written for a second-year programming course.

```
voice message ──▶ dataset bot ──▶ VAD split ──▶ training ──▶ model.pkl ──▶ recognition bot
```

## Setup

```sh
pip3 install pyTelegramBotAPI numpy scipy scikit-learn librosa matplotlib
```

`ffmpeg` must be on the `PATH` as well — both bots shell out to it to convert
Telegram's `.ogg` voice messages into `.wav`.

Each bot reads its token from the environment. Create the bots with
[@BotFather](https://t.me/BotFather) and export the tokens before starting:

```sh
export TG_DATASET_BOT_TOKEN="your token"
export TG_RECOGNITION_BOT_TOKEN="your token"
```

## Collecting a dataset

```sh
make bot_dataset
```

The bot asks for a voice message with five digits and stores the recording in
`dataset/ogg` and `dataset/wav`.

## Splitting the recordings into single digits

```sh
make vad_sort     # voice activity detection: cut each recording into segments
make splitted     # move the segments into dataset/splitted_final and clean up
```

`split_by_vad.py` finds the segments by energy: it takes a window of 0.1 s with
a 0.005 s step, marks the parts above the energy threshold as speech and cuts on
the pauses between digits.

## Training

```sh
make training
```

`training.py` extracts features with `librosa` and fits an `MLPClassifier` with
hidden layers of 888, 777 and 666 neurons over ten classes. The result is
pickled into `model.pkl`. A `RandomForestClassifier` is left in the file,
commented out, as the alternative that was tried.

## Recognition

```sh
make bot_recognition
```

Send a voice message with one digit and the bot replies with its prediction.

## Repository layout

```
audio_digits_dataset_bot.py       collects recordings
audio_digits_recognition_bot.py   answers with a prediction
split_by_vad.py                   voice activity detection and segmentation
training.py                       feature extraction and model fitting
inference.py                      prediction on a single file, without Telegram
model.pkl                         the trained MLPClassifier (77 MB)
dataset/                          collected recordings
```

## Known issues

- **The two bots this README used to link to no longer exist.** The tokens that
  used to be hardcoded here were revoked and the bots deleted. To run any of
  this you have to create your own bots, as described in Setup.
- `model.pkl` weighs 77 MB and is committed to the repository, which is what
  makes cloning slow. It stays tracked because the size is already paid for in
  the history — deleting it now would shrink nothing and would leave the
  recognition bot without a model.
- There is no `requirements.txt`; the dependency list above was reconstructed
  from the imports.
- The dataset bot expects exactly five digits per message, and the number is
  hardcoded rather than configurable.
