import numpy as np # generate random
import os          # get current directory path
import subprocess  # execute ffmpeg
import telebot     # run telegram bot
import time

from datetime import datetime # generate log

users_tasks = dict()
# How many digits one recording holds. split_by_vad.py reads it back from the
# file name, so changing it here is enough for the whole pipeline.
DIGITS_PER_TASK = 5
# Get a token from @BotFather and put it in the environment:
#     export TG_DATASET_BOT_TOKEN="your token"
bot = telebot.TeleBot(os.environ.get("TG_DATASET_BOT_TOKEN", "YOUR TOKEN HERE"))
root = os.getcwd() + "/dataset"
# Git does not store empty directories, so a fresh clone has none of these.
for folder in ("ogg", "wav"):
    os.makedirs(root + "/" + folder, exist_ok=True)


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


def generate_task():
    return ' '.join(list(map(str, np.random.randint(10, size=DIGITS_PER_TASK))))


def log(text):
    time_stamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    print(time_stamp + " " + text)


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    user = message.from_user.id
    text = message.text
    log(f"User ({user}): {text}")

    users_tasks[user] = generate_task()
    bot.send_message(user,
        f"Please say the following {DIGITS_PER_TASK} digits, with a pause between each:"
        f"\n{users_tasks[user]}")


@bot.message_handler(content_types=['voice'])
def get_voice_messages(message):
    user = message.from_user.id
    voice = message.voice
    log(f"User ({user}): voice")
    # Without a task there is no digit sequence to name the recording after.
    if user not in users_tasks:
        bot.send_message(user,
            f"Send me any message first and I will give you {DIGITS_PER_TASK} digits to say.")
        return

    tele_file = bot.get_file(voice.file_id)
    ogg_data = bot.download_file(tele_file.file_path)
    file_name = users_tasks[user].replace(" ", "_")
    ogg_path = root + "/ogg/" + file_name + ".ogg"
    wav_path = root + "/wav/" + file_name + ".wav"
    save_ogg(ogg_data, ogg_path)
    if convert_ogg_wav(ogg_path, wav_path) == "timeout":
        bot.send_message(user, "Converting your recording took too long. Please try again.")
        return
    bot.send_message(user, "Thank you")


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