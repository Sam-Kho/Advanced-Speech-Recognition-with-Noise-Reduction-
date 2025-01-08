import numpy as np
import librosa
import librosa.display
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
import vosk
import wave
import logging

# تنظیمات برای مخفی کردن ارورهای Vosk
logging.getLogger("VoskAPI").setLevel(logging.WARNING)

# اصلاح برای نوع داده‌های قدیمی numpy complex
if not hasattr(np, 'complex'):  # بررسی سازگاری
    np.complex = np.complex128

# بارگذاری فایل صوتی
audio_path = 'd:/Users/HOME/Downloads/Music/cmvol4bk2_01_mason_64kb-[AudioTrimmer.com].mp3'
y, sr_rate = librosa.load(audio_path, sr=None)  # بارگذاری صوت با نمونه‌برداری اصلی

# مرحله 1: اضافه کردن نویز به فایل صوتی
np.random.seed(0)  # تنظیم بذر تصادفی برای تولید نویز
noise = np.random.normal(0, 0.02, y.shape)  # تولید نویز گوسی
y_noisy = y + noise  # اضافه کردن نویز به سیگنال اصلی

# ذخیره‌سازی فایل صوتی نویزدار
noisy_audio_path = "C:/Users/HOME/noisy_audio.wav"
sf.write(noisy_audio_path, y_noisy, sr_rate)

# مرحله 2: حذف نویز از فایل صوتی
y_denoised = nr.reduce_noise(y=y_noisy, sr=sr_rate, stationary=True)  # استفاده از کتابخانه noisereduce برای کاهش نویز

# ذخیره‌سازی فایل صوتی بدون نویز
denoised_audio_path = "C:/Users/HOME/denoised_audio.wav"
sf.write(denoised_audio_path, y_denoised, sr_rate)

# مرحله 3: تبدیل صوت به متن با استفاده از Vosk (آفلاین)

# تبدیل فایل صوتی WAV به فرمت سازگار با Vosk
sound = AudioSegment.from_wav(denoised_audio_path)  # بارگذاری فایل صوتی با pydub
sound = sound.set_channels(1)  # اطمینان از اینکه سیگنال تک‌کاناله باشد
sound = sound.set_frame_rate(16000)  # اطمینان از اینکه نرخ نمونه‌برداری 16 کیلوهرتز باشد
wav_output_path = "C:/Users/HOME/final_audio.wav"
sound.export(wav_output_path, format="wav")  # ذخیره‌سازی فایل صوتی با فرمت WAV

# بارگذاری مدل Vosk
model_path = "D:/vosk-model-small-en-us-0.15"
from vosk import Model, KaldiRecognizer

try:
    model = Model(model_path)  # بارگذاری مدل Vosk
    wf = wave.open(wav_output_path, "rb")  # باز کردن فایل صوتی
    # بررسی صحت فرمت فایل صوتی
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000]:
        raise ValueError("Audio file must be WAV format mono PCM with a sample rate of 8000 or 16000 Hz.")

    recognizer = KaldiRecognizer(model, wf.getframerate())  # ایجاد شیء Recognizer برای شناسایی صدا
    recognizer.SetWords(True)  # تنظیم شناسایی کلمات

    text = ""  # متغیر برای ذخیره متن شناسایی شده
    while True:
        data = wf.readframes(4000)  # خواندن داده‌ها از فایل صوتی
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):  # بررسی اینکه آیا داده‌ها به طور کامل پردازش شده‌اند
            result = recognizer.Result()  # گرفتن نتیجه شناسایی
            result_dict = eval(result)  # تبدیل نتیجه به دیکشنری
            if 'text' in result_dict:
                text += result_dict['text'] + " "  # اضافه کردن متن شناسایی شده به متغیر text

    # گرفتن نتیجه نهایی
    final_result = recognizer.FinalResult()  # گرفتن نتیجه نهایی
    final_result_dict = eval(final_result)  # تبدیل به دیکشنری
    if 'text' in final_result_dict:
        text += final_result_dict['text']  # اضافه کردن متن نهایی به متغیر text

    # چاپ تمامی متن‌های شناسایی شده
    print(text)

except Exception as e:
    print("Error during speech recognition:", str(e))  # در صورتی که خطایی پیش آید، پیام خطا نمایش داده می‌شود
