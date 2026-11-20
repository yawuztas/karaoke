# 🎤 Karaoke Pipeline — Song In, Karaoke Video Out

A single-file Python pipeline that turns any song into a word-by-word karaoke
video: downloads the track, separates vocals, fetches/aligns lyrics with
word-level timing, and renders a 720p video with live waveform, Ken Burns
cover zoom and synchronized karaoke subtitles.

Built as a gift for my daughter — say the song name, everything else is
automatic.

> **Türkçe özet için aşağı kaydırın** — ayrıntılı Türkçe dokümantasyon:
> [README_TR.md](README_TR.md)

## ✨ Features

- **Word-by-word karaoke highlight** — sung words turn white, upcoming words
  stay amber (ASS `{\k}` subtitles)
- **Smart lyrics pipeline** (priority order):
  1. **LDDC** — true word-level timing from QQ/Kugou/NetEase
  2. **LRCLIB** — synced lyrics (drops to line-level if needed)
  3. **Force-align** — known plain-text lyrics aligned to vocals with WhisperX
  4. **WhisperX** — large-v3 transcription + alignment as last resort
- **Never fakes timing** — if the lyric *text* is known, it is force-aligned to
  the vocals; noticeably more accurate than raw Whisper for Turkish repertoire
- **Live waveform + Ken Burns cover zoom** in the rendered video
- **`--transpose`** — pitch shift without changing duration (subtitles survive)
- **One-click GUI** (`launch.sh`, zenity) — queue multiple songs back-to-back
- **Stem reuse** — change subtitles and re-render in ~30 seconds

## 🚀 Getting Started

Requirements: Python 3.12, ffmpeg, (optional) NVIDIA GPU + CUDA

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt   # first WhisperX/Demucs run downloads ~3 GB
./launch.sh                                   # GUI
```

```bash
# CLI examples
./venv/bin/python karaoke.py                          # process everything in input/
./venv/bin/python karaoke.py "input/song.mp3" --transpose -2   # one tone down
./venv/bin/python karaoke.py "input/song.mp3" --srt-only       # subtitles only
```

Output: `output/<song>/<song>_karaoke.mp4`

## 🛠 Built On

[Demucs](https://github.com/facebookresearch/demucs) (vocal separation) ·
[WhisperX](https://github.com/m-bain/whisperX) (alignment/ASR) ·
[LDDC](https://github.com/chenmozhijin/LDDC) (word-level lyrics) ·
[LRCLIB](https://lrclib.net) · [spotdl](https://github.com/spotDL/spotify-downloader)
· Turkish alignment model: `wav2vec2-xls-r-300m-cv7-turkish`

## 📄 License & Notes

Downloaded music is copyrighted — **personal use only**. Personal/educational
project; see repository settings.

---

## 🇹🇷 Türkçe Özet

Kızım için yazdığım tek dosyalık karaoke hazırlayıcısı: şarkı adını söyle,
gerisini halleder. Şarkıyı indirir → vokali enstrümandan ayırır (Demucs) →
sözleri getirir (LDDC / LRCLIB / force-align / WhisperX) → kelime kelime
senkronlu karaoke videosu üretir (FFmpeg: dalga formu + Ken Burns kapak).

- Söz *metni* biliniyorsa asla sahte zamanlama kullanılmaz — vokallere
  force-align edilir; Türkçe repertuvarda saf Whisper'dan belirgin şekilde doğru
- `--transpose` ile perde kaydırma (süre sabit), `--srt-only` ile sadece altyazı
- Mevcut stem'ler yeniden kullanılır: altyazıyı değiştirip 30 saniyede yeniden render
- Kişisel kullanım içindir; indirilen müzik teliflidir

Ayrıntılı akış, kurulum ve kullanım: [README_TR.md](README_TR.md)
