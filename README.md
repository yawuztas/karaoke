# 🎤 Karaoke

Kızım için yazdığım tek dosyalık karaoke hazırlayıcı: şarkı adını söyle, gerisini halleder.

Şarkıyı indirir → vokali enstrümantalden ayırır → sözleri getirir → kelime kelime senkronlu karaoke videosu üretir.

## Akış

```
şarkı adı → spotdl (indirme) → Demucs (vokal/enstrüman ayrımı)
          → söz kaldıracı (aşağıda) → ASS {\k} karaoke altyazı
          → FFmpeg (720p video: dalga formu + Ken Burns kapak + ton kaydırma)
```

### Söz kaldıracı (öncelik sırasıyla)

1. **LDDC** — QQ/Kugou/NetEase'ten gerçek kelime-düzeyi (harf harf) zamanlama
2. **LRCLIB** — senkronlu sözler (satır düzeyi ise 3'e düşer)
3. **Force-align** — bilinen düz metin sözler, vokallere WhisperX ile hizalanır
4. **WhisperX** — söz bulunamazsa large-v3 ile transkripsiyon + hizalama

Kritik nokta: sözün *metni* biliniyorsa asla sahte zamanlama kullanılmaz — metin vokallere force-align edilir. Türkçe repertuvar için bu, saf Whisper transkripsiyonundan belirgin şekilde daha doğru.

## Özellikler

- Kelime kelime karaoke vurgusu (söylenen beyaz, gelecek sarı)
- Canlı dalga formu + kapak resminde yavaş zoom (Ken Burns)
- `--transpose` ile perde kaydırma (süre sabit kalır, altyazı bozulmaz)
- Çift tıkla çalışan GUI (`launch.sh`, zenity) — art arda çok şarkı
- Mevcut stem'ler yeniden kullanılır: altyazıyı değiştirip 30 saniyede yeniden render

## Kurulum

Gereksinimler: Python 3.12, ffmpeg, (isteğe bağlı) NVIDIA GPU + CUDA

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

İlk WhisperX/Demucs çalıştırması modelleri indirir (~3 GB, `~/.cache`).

## Kullanım

GUI (çift tık veya terminal):

```bash
./launch.sh
```

CLI:

```bash
# input/ klasörüne şarkı koy, hepsini işle
./venv/bin/python karaoke.py

# tek şarkı, 1 tam ses aşağı
./venv/bin/python karaoke.py "input/Tarkan - Şımarık.mp3" --transpose -2

# altyazıyı yeniden üret (video render yok)
./venv/bin/python karaoke.py "input/şarkı.mp3" --srt-only

# çevrimiçi söz kaynaklarını atla, doğrudan transkripsiyon
./venv/bin/python karaoke.py "input/şarkı.mp3" --no-lrclib
```

Çıktı: `output/<Şarkı>/<Şarkı>_karaoke.mp4`

## Klasör yapısı

```
input/      işlenecek şarkılar (repoya girmez)
output/     bitmiş videolar + stem'ler (repoya girmez)
models/     nltk verisi + son çalışma logu (repoya girmez)
launch.sh   zenity GUI başlatıcı
karaoke.py  tüm pipeline (tek dosya)
```

## Notlar

- `spotdl` YouTube/Spotify'dan indirir; indirilen müzik teliflidir, **kişisel kullanım** içindir.
- Türkçe hizalama modeli: `wav2vec2-xls-r-300m-cv7-turkish`; İngilizce ve 90+ dil otomatik desteklenir.
- Bu repo kişisel/öğretici amaçlıdır; temelde [Demucs](https://github.com/facebookresearch/demucs), [WhisperX](https://github.com/m-bain/whisperX), [LDDC](https://github.com/chenmozhijin/LDDC), [LRCLIB](https://lrclib.net) ve [spotdl](https://github.com/spotDL/spotify-downloader) kullanır.
