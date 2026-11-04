#!/usr/bin/env python3
"""Karaoke pipeline: şarkı (MP3/FLAC/WAV) -> karaoke videosu (MP4).

Adımlar:
  1) Orijinal kopyalanır.
  2) Sözler zamanlanır (ilk başarılı kaynak kullanılır):
     LDDC kelime-düzeyi (VERBATIM) → LRCLIB A2 kelime-düzeyi → bilinen
     sözleri WhisperX ile force-align → saf WhisperX transkripsiyonu.
  3) Demucs ile vokal/instrumental ayrımı yapılır (söz kaynağı gerektirirse).
  4) FFmpeg ile instrumental + altyazı birleştirilerek karaoke videosu oluşturulur.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import torch

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"

# NLTK veri yolunu garanti et (WhisperX hizalama için gerekli)
import nltk
_LOCAL_NLTK = ROOT / "models" / "nltk_data"
if _LOCAL_NLTK.exists() and str(_LOCAL_NLTK) not in nltk.data.path:
    nltk.data.path.insert(0, str(_LOCAL_NLTK))
DEMUCS_MODEL = "htdemucs"
WHISPER_MODEL = "large-v3"
DEMUCS_SHIFTS = 2  # 2 daha temiz vokal verir (WhisperX doğruluğunu artırır)
DEFAULT_OFFSET = 0.0  # altyazı zamanlama ofseti (saniye)


def song_slug(path: Path) -> str:
    return path.stem.replace(" ", "_")


def find_inputs() -> list[Path]:
    exts = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".opus"}
    return sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in exts)


def find_cover(audio: Path) -> Path | None:
    for name in ("cover.jpg", "cover.png", "folder.jpg", "album.jpg"):
        cand = audio.parent / name
        if cand.exists():
            return cand
    return None


def probe_duration(audio: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(audio),
    ]).decode().strip()
    return float(out)


def save_wav(path: Path, tensor: torch.Tensor, sr: int) -> None:
    """torch ses tensörünü soundfile ile diske yazar (torchcodec/torchaudio I/O sorunundan kaçınır)."""
    arr = tensor.detach().cpu().float().numpy()
    if arr.ndim == 1:
        arr = arr[:, None]
    else:
        arr = arr.T  # (channels, samples) -> (samples, channels)
    arr = np.clip(arr, -1.0, 1.0)
    sf.write(str(path), arr, sr, subtype="PCM_16")


def separate(audio: Path, out_dir: Path) -> tuple[Path, Path]:
    from demucs.api import Separator

    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sep = Separator(
        model=DEMUCS_MODEL,
        device=device,
        shifts=DEMUCS_SHIFTS,
        overlap=0.25,
        segment=7.8,
        jobs=1,
    )
    _, stems = sep.separate_audio_file(str(audio))
    sr = sep.samplerate

    vocals = stems["vocals"].cpu()
    instrumental = sum(stems[s] for s in stems if s != "vocals").cpu()

    vocals_path = out_dir / "vocals.wav"
    instr_path = out_dir / "instrumental.wav"
    save_wav(vocals_path, vocals, sr)
    save_wav(instr_path, instrumental, sr)
    return vocals_path, instr_path


def fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


# LRCLIB API — synced lyrics veritabanı
LRCLIB_BASE = "https://lrclib.net/api"
LRCLIB_TIMEOUT = 10
LRCLIB_RETRIES = 3


def http_get_json(url: str, retries: int = LRCLIB_RETRIES):
    """JSON döndüren GET. 5xx hatalarda tekrar dener."""
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "karaoke/1.0"})
            with urllib.request.urlopen(req, timeout=LRCLIB_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise last_err if last_err else RuntimeError("LRCLIB request failed")


def fetch_lrclib(artist: str, title: str, duration: float | None = None) -> dict | None:
    """LRCLIB'de şarkı ara, en uygun sonucu döndür.

    Önce /api/get ile tam eşleşme dene, yoksa /api/search ile arama yap.
    duration verilmişse en yakın süreyi seç.
    """
    artist = (artist or "").strip()
    title = (title or "").strip()
    if not artist or not title:
        return None

    # 1) Direkt eşleşme dene
    try:
        qs = urllib.parse.urlencode({"artist_name": artist, "track_name": title})
        url = f"{LRCLIB_BASE}/get?{qs}"
        try:
            data = http_get_json(url)
            if data and data.get("syncedLyrics"):
                return data
        except urllib.error.HTTPError:
            pass  # tam eşleşme yok (404 vb.); arama yoluna düş
    except Exception:
        pass

    # 2) Arama ile dene
    try:
        url = f"{LRCLIB_BASE}/search?q={urllib.parse.quote(f'{artist} {title}')}"
        data = http_get_json(url)
    except Exception as e:
        print(f"   LRCLIB arama hatası: {e}", file=sys.stderr)
        return None

    if not isinstance(data, list) or not data:
        return None

    candidates = [d for d in data if d.get("syncedLyrics")]
    if not candidates:
        return None

    if duration is not None and len(candidates) > 1:
        candidates.sort(key=lambda d: abs((d.get("duration") or 0) - duration))
    return candidates[0]


def parse_artist_title(audio: Path) -> tuple[str, str]:
    """Dosya adından veya mutagen metadata'sından sanatçı/şarkı adı çıkar."""
    # Önce mutagen ile dene (daha güvenilir)
    try:
        from mutagen import File
        f = File(str(audio))
        if f is not None:
            artist = (f.get("artist") or [None])[0] if f.get("artist") else None
            title = (f.get("title") or [None])[0] if f.get("title") else None
            if artist and title:
                return str(artist), str(title)
    except Exception:
        pass

    # Dosya adından parse et: "Artist - Title.ext"
    stem = audio.stem
    if " - " in stem:
        a, t = stem.split(" - ", 1)
        return a.strip(), t.strip()
    return "", stem


_LRC_LINE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")
_LRC_WORD = re.compile(r"<(\d+):(\d+(?:\.\d+)?)>")


def parse_lrc_to_segments(lrc_text: str) -> list[dict]:
    """LRC formatını WhisperX benzeri segment listesine çevirir.

    Hem satır-düzeyi ([mm:ss.xx]text) hem de kelime-düzeyi A2 formatını
    ([mm:ss.xx]w<mm:ss.xx>w<mm:ss.xx>w veya [mm:ss.xx]w[mm:ss.xx]w) destekler.
    Çıktı: [{'start','end','text','words':[{start,end,word}]}, ...]
    """
    lines_raw: list[tuple[float, str]] = []
    for line in lrc_text.splitlines():
        line = line.strip()
        m = _LRC_LINE.match(line)
        if not m:
            continue
        start = int(m.group(1)) * 60 + float(m.group(2))
        rest = line[m.end():].strip()
        lines_raw.append((start, rest))

    if not lines_raw:
        return []

    # Son satır için sahte end time
    last_end = lines_raw[-1][0] + 5.0

    segments = []
    for i, (start, text) in enumerate(lines_raw):
        end = lines_raw[i + 1][0] if i + 1 < len(lines_raw) else last_end

        # Kelime-düzeyi taraması
        word_matches = list(_LRC_WORD.finditer(text))
        words: list[dict] = []
        if word_matches:
            # Kelimeleri timestamp'lerden ayır
            # Format A: word<ts>word<ts>word (timestamp önceki kelime için)
            # Format B: <ts>word<ts>word (timestamp sonraki kelime için)
            # Basit yaklaşım: her <ts>'den sonraki kelime o ts'de başlar
            for j, wm in enumerate(word_matches):
                ws = int(wm.group(1)) * 60 + float(wm.group(2))
                # bu timestamp'ten sonraki ilk kelimeyi bul
                after = text[wm.end():].lstrip()
                # kelime sınırı: boşluk veya özel
                word_match = re.match(r"\S+", after)
                if not word_match:
                    continue
                word = word_match.group(0)
                # kelime bitişi: bir sonraki word timestamp'i (pozisyon->zaman dönüşümü yok)
                we_time = (int(word_matches[j + 1].group(1)) * 60 + float(word_matches[j + 1].group(2))
                           if j + 1 < len(word_matches) else end)
                words.append({"start": ws, "end": we_time, "word": word})
        else:
            # Satır-düzeyi: kelimeleri eşit dağıt
            tokens = text.split()
            if tokens:
                dur = (end - start) / max(1, len(tokens))
                for k, tok in enumerate(tokens):
                    ws = start + dur * k
                    we = start + dur * (k + 1)
                    words.append({"start": ws, "end": we, "word": tok})

        if words:
            segments.append({
                "start": start,
                "end": end,
                "text": text,
                "words": words,
            })

    return segments


def _lddc_lines_to_segments(lines) -> list[dict]:
    """LDDC LyricsLine/LyricsWord listesini WhisperX-benzeri segment sözlüklerine çevirir.

    LDDC tokenları arasında boşluk ve noktalama ayrı 'kelime' olarak gelir:
    saf boşluklar önceki kelimeyi uzatır, harf içermeyen tokenlar (noktalama)
    önceki kelimeyle birleştirilir — ASS {\\k} vurgusu kelime bazlı çalıştığı için.
    """
    segments: list[dict] = []
    for line in lines:
        words: list[dict] = []
        for w in line.words:
            text = w.text or ""
            if not text.strip():
                if words:
                    words[-1]["end"] = max(words[-1]["end"], w.end / 1000.0)
                continue
            if not any(ch.isalnum() for ch in text):
                if words:
                    words[-1]["end"] = max(words[-1]["end"], w.end / 1000.0)
                    words[-1]["word"] += text
                continue
            words.append({"start": w.start / 1000.0, "end": w.end / 1000.0, "word": text})
        if words:
            segments.append({
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "text": "".join(w["word"] for w in words).strip(),
                "words": words,
            })
    return segments


def fetch_lddc_segments(artist: str, title: str, duration: float | None = None) -> list[dict] | None:
    """LDDC (QQ/Kugou/NetEase) 'den kelime-düzeyi (VERBATIM) karaoke sözleri getirir.

    Gerçek harf-harf zamanlama bulunursa WhisperX-benzeri segment listesi döner;
    hiçbir kaynakta yoksa None (sıradaki söz kaynağına düşülür).
    """
    try:
        from LDDC.core.api.lyrics import get_lyrics, search
        from LDDC.common.models import LyricsType, SearchType, Source
    except ImportError as e:
        print(f"   LDDC kurulu değil ({e}), atlanıyor", file=sys.stderr)
        return None

    query = f"{artist} {title}"
    for src in (Source.KG, Source.NE, Source.QM):
        try:
            results = search(src, query, SearchType.SONG)
            if not results:
                continue
            candidates = list(results)
            if duration is not None:
                # Aynı şarkının birden çok yüklemesi olur; bazıları VERBATIM içerir.
                # Süreye göre sırala, ilk N adayı tek tek dene.
                candidates.sort(key=lambda r: abs((r.duration or 0) / 1000.0 - duration))
            for picked in candidates[:3]:
                if duration is not None and abs((picked.duration or 0) / 1000.0 - duration) >= 8.0:
                    break  # sıralı liste: gerisi daha da uzak
                lyrics = get_lyrics(picked)
                if lyrics.types.get("orig") != LyricsType.VERBATIM:
                    continue
                segments = _lddc_lines_to_segments(lyrics.data.get("orig") or [])
                if segments:
                    print(f"   LDDC: kelime-düzeyi sözler bulundu ({src.name}: {picked.artist} — {picked.title})")
                    return segments
        except Exception as e:
            print(f"   LDDC {getattr(src, 'name', src)} hatası: {e}", file=sys.stderr)
            continue
    return None


def _timed_lines_from_lrc(lrc_text: str) -> list[dict]:
    """Satır-düzeyi LRC'yi [{'start','end','text'}] segmentlerine çevirir (force-align girdisi)."""
    lines_raw: list[tuple[float, str]] = []
    for line in lrc_text.splitlines():
        m = _LRC_LINE.match(line.strip())
        if not m:
            continue
        start = int(m.group(1)) * 60 + float(m.group(2))
        rest = line[m.end():].strip()
        if rest:
            lines_raw.append((start, rest))
    segments = []
    for i, (start, text) in enumerate(lines_raw):
        end = lines_raw[i + 1][0] if i + 1 < len(lines_raw) else start + 5.0
        segments.append({"start": start, "end": end, "text": text})
    return segments


def fetch_lrclib_lyrics(audio: Path, artist: str, title: str,
                        duration: float | None) -> tuple[list[dict] | None, list[dict] | None, str | None]:
    """LRCLIB'den sözleri zamanlama kalitesine göre üç formda getirir.

    Returns: (a2_segments, line_segments, plain_text)
      a2_segments   — A2 kelime tag'li synced lyrics (gerçek kelime zamanlaması)
      line_segments — satır-düzeyi synced lyrics [{'start','end','text'}] (satır zamanı gerçek)
      plain_text    — zamanı olmayan düz metin (force-align iskeleti gerekir)
    """
    if not artist or not title:
        print("   LRCLIB: şarkı metadata'sı okunamadı", file=sys.stderr)
        return None, None, None
    print(f"   LRCLIB aranıyor: {artist} — {title}")
    try:
        data = fetch_lrclib(artist, title, duration)
    except Exception as e:
        print(f"   LRCLIB hatası: {e}", file=sys.stderr)
        return None, None, None
    if not data:
        print("   LRCLIB: sonuç yok")
        return None, None, None

    print(f"   LRCLIB: bulundu (id={data.get('id')}, dur={data.get('duration')}s)")
    synced = (data.get("syncedLyrics") or "").strip()
    plain = (data.get("plainLyrics") or "").strip()

    if synced and _LRC_WORD.search(synced):
        segments = parse_lrc_to_segments(synced)
        if segments:
            return segments, None, None

    if synced:
        lines = _timed_lines_from_lrc(synced)
        if lines:
            return None, lines, None

    if plain:
        return None, None, plain
    return None, None, None


def write_karaoke_lyrics(segments: list[dict], work: Path) -> int:
    """Söz segmentlerini lyrics.json + lyrics.ass olarak yazar (cue sayısını döndürür)."""
    json_path = work / "lyrics.json"
    json.dump(segments, json_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return write_ass_karaoke(segments, work / "lyrics.ass")


def fmt_ass_ts(seconds: float) -> str:
    """ASS zaman formatı: H:MM:SS.cc (santisaniye)."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


ASS_HEADER = """[Script Info]
Title: Karaoke
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,40,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,40,40,160,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def write_ass_karaoke(segments: list[dict], path: Path) -> int:
    r"""Gerçek karaoke: cümle ekranda, kelimeler sırayla vurgulanır (ASS {\k} tag'leri).

    - PrimaryColour (FFFFFF) = vurgulanmamış metin (beyaz)
    - SecondaryColour (00FFFF) = vurgulanan metin (sarı)
    - {\k<santisaniye>} bir sonraki kelimenin vurgu süresi
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    events: list[str] = []
    for seg in segments:
        words = seg.get("words") or []
        valid = [w for w in words if w.get("start") is not None and w.get("end") is not None]
        if not valid:
            continue

        seg_start = valid[0]["start"]
        seg_end = valid[-1]["end"]

        parts: list[str] = []
        prev_end = seg_start
        for w in valid:
            w_start = w["start"]
            w_end = w["end"]
            duration_cs = max(1, int(round((w_end - prev_end) * 100)))
            token = w["word"].strip()
            if not token:
                continue
            parts.append(f"{{\\k{duration_cs}}}{token}")
            prev_end = w_end

        if not parts:
            continue
        text = " ".join(parts)
        events.append(
            f"Dialogue: 0,{fmt_ass_ts(seg_start)},{fmt_ass_ts(seg_end)},Default,,0,0,0,,{text}"
        )

    with path.open("w", encoding="utf-8") as f:
        f.write(ASS_HEADER)
        for e in events:
            f.write(e + "\n")
    return len(events)


def write_srt(segments: list[dict], path: Path, word_level: bool = True, offset: float = 0.0) -> int:
    """WhisperX hizalama çıktısını SRT formatına yazar.

    word_level=True ise her kelime ayrı bir altyazı kuyruğu olur.
    offset saniye cinsinden tüm kuyruklara eklenir.
    Yazılan kuyruk sayısını döndürür.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cues: list[tuple[float, float, str]] = []
    for seg in segments:
        words = seg.get("words") or []
        if word_level and words:
            for w in words:
                start, end, text = w.get("start"), w.get("end"), w.get("word", "")
                if start is None or end is None:
                    continue
                t = text.strip()
                if t:
                    cues.append((start + offset, end + offset, t))
        else:
            start, end, text = seg.get("start"), seg.get("end"), seg.get("text", "")
            if start is None or end is None:
                continue
            t = text.strip()
            if t:
                cues.append((start + offset, end + offset, t))

    cues.sort(key=lambda c: c[0])
    cues = [(max(0.0, s), max(0.0, e) if e > 0 else e, t) for s, e, t in cues]
    with path.open("w", encoding="utf-8") as f:
        for i, (s, e, t) in enumerate(cues, 1):
            f.write(f"{i}\n{fmt_ts(s)} --> {fmt_ts(e)}\n{t}\n\n")
    return len(cues)


def _asr_options(initial_prompt: str | None) -> dict:
    return {
        "beam_size": 10,
        "best_of": 5,
        "temperatures": [0.0],
        "condition_on_previous_text": True,
        "initial_prompt": initial_prompt,
        "suppress_blank": True,
    }


def _load_whisper(language: str, initial_prompt: str | None = None):
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return whisperx.load_model(
        WHISPER_MODEL,
        device=device,
        language=language,
        compute_type=compute_type,
        asr_options=_asr_options(initial_prompt),
    )


def write_subtitle_outputs(segments: list[dict], out_dir: Path, style: str) -> tuple[Path, Path, int]:
    """Segments'i lyrics.json + altyazı dosyasına (ASS/SRT) yazar."""
    json_path = out_dir / "lyrics.json"
    json.dump(segments, json_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if style == "karaoke":
        sub_path = out_dir / "lyrics.ass"
        n = write_ass_karaoke(segments, sub_path)
    else:
        sub_path = out_dir / "lyrics.srt"
        n = write_srt(segments, sub_path, word_level=(style == "word"))
    return sub_path, json_path, n


def transcribe(vocals: Path, out_dir: Path, language: str, style: str = "karaoke", initial_prompt: str | None = None) -> tuple[Path, Path, int]:
    """WhisperX transkripsiyon + altyazı üretimi.

    style:
      - "karaoke" -> ASS dosyası (gerçek karaoke: cümle + sıralı kelime vurgusu)
      - "word"    -> SRT dosyası (her kelime ayrı cue)
      - "sentence"-> SRT dosyası (her cümle ayrı cue)
    Returns: (altyazı_dosyası, json_dosyası, cue_sayısı)
    """
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_whisper(language, initial_prompt)
    audio = whisperx.load_audio(str(vocals))

    result = model.transcribe(audio, language=language, batch_size=4)
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(
        result["segments"], align_model, metadata, audio, device=device,
        interpolate_method="linear",
    )

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return write_subtitle_outputs(aligned["segments"], out_dir, style)


def _map_known_text_to_segments(segments: list[dict], known_text: str) -> list[dict]:
    """Bilinen sözleri Whisper segment iskeletine kelime sayısına oransal dağıtır.

    Whisper'ın kendi metni yanlış olabilir; segment sınırları sadece ZAMAN
    iskeleti olarak kullanılır. Bilinen kelimeler, segmentlerin kelime sayılarına
    orantılı kümülatif payla bölünür: segment i'nin hedefi =
    round(toplam_bilinen * kümülatif_whisper_kelimesi / toplam_whisper_kelimesi).
    """
    known_words = [t for line in known_text.splitlines() for t in line.split() if t]
    whisper_counts = [len(s.get("text", "").split()) for s in segments]
    total_whisper = sum(whisper_counts)
    if not known_words or total_whisper == 0:
        return []

    pseudo: list[dict] = []
    pos = 0
    acc = 0
    for seg, cnt in zip(segments, whisper_counts):
        acc += cnt
        target = round(len(known_words) * acc / total_whisper)
        chunk = known_words[pos:target]
        pos = target
        if chunk:
            pseudo.append({"start": seg["start"], "end": seg["end"], "text": " ".join(chunk)})
    return pseudo


def align_known_text(vocals: Path, out_dir: Path, language: str,
                     known_text: str | None = None,
                     known_lines: list[dict] | None = None) -> tuple[Path, Path, int] | None:
    """Bilinen sözleri WhisperX hizalama modeliyle vokallere zorla hizalar (force-align).

    known_lines: satır-düzeyi zamanlı sözler ([{'start','end','text'}]) — transkripsiyon
    atlanır, satır sınırları doğrudan hizalama penceresi olur.
    known_text: zamanı olmayan düz metin — önce Whisper zaman iskeleti çıkarılır,
    bilinen kelimeler bu iskelete dağıtılıp hizalanır.

    Hizalama başarısız olursa None döner (çağıran saf WhisperX'e düşer).
    """
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio = whisperx.load_audio(str(vocals))

    if known_lines:
        pseudo = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in known_lines]
    elif known_text:
        model = _load_whisper(language)
        result = model.transcribe(audio, language=language, batch_size=4)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        pseudo = _map_known_text_to_segments(result["segments"], known_text)
    else:
        return None
    if not pseudo:
        print("   force-align: bilinen sözler zaman iskeletine oturmadı", file=sys.stderr)
        return None

    try:
        align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
        aligned = whisperx.align(
            pseudo, align_model, metadata, audio, device=device,
            interpolate_method="linear",
        )
    except Exception as e:
        print(f"   force-align hatası: {e}", file=sys.stderr)
        return None
    return write_subtitle_outputs(aligned["segments"], out_dir, "karaoke")


def make_video(instr: Path, sub: Path | None, cover: Path | None, mp4: Path,
               n_cues: int, style: str, transpose: int = 0) -> None:
    """Instrumental + altyazıyı karaoke videosuna birleştirir.

    Kapak varsa Ken Burns (bütün şarkı boyunca yavaş zoom) uygulanır ve alta
    canlı dalga formu bindirilir. transpose: yarım ses aralığı cinsinden perde
    kaydırma — asetrate perdeyi kaydırırken tempo da değiştirir; atempo ile
    tempo eski haline getirilir, dolayısıyla süre ve altyazı zamanlaması sabit kalır.
    """
    mp4.parent.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(instr)
    frames = max(2, int(duration * 25) + 2)

    if cover and cover.exists():
        # 1472x828 ön ölçek (=1280x720 x 1.15): zoom sonunda bile yukarı
        # ölçekleme yapılmadığı için görüntü yumuşamaz
        bg = (
            "[0:v]scale=1280:720:force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,"
            "scale=1472:828,setsar=1,"
            f"zoompan=z='min(1+0.10*on/{frames},1.10)':"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={frames}:s=1280x720:fps=25[bg]"
        )
        inputs = ["-i", str(cover)]
    else:
        bg = "[0:v]fps=25,format=yuv420p[bg]"
        inputs = ["-f", "lavfi", "-i", "color=c=black:s=1280x720:r=25"]

    if transpose:
        sr_out = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate",
            "-of", "default=noprint_wrappers=1:nokey=1", str(instr),
        ]).decode().strip()
        sr = int(sr_out) if sr_out.isdigit() else 44100
        semis = 2.0 ** (transpose / 12.0)
        audiofx = (f"asetrate={int(sr * semis)},aresample={sr},"
                   f"atempo={1.0 / semis:.6f},")
    else:
        audiofx = ""

    sub_filter = None
    if n_cues > 0 and sub is not None:
        sub_escaped = str(sub).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        if style == "karaoke":
            # ASS karaoke — {\k} tag'leri libass tarafından işlenir
            sub_filter = f"ass='{sub_escaped}'"
        else:
            # SRT — force_style ile manuel stil
            force_style = (
                "FontName=DejaVu Sans,FontSize=28,"
                "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                "BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=150"
            )
            sub_filter = f"subtitles='{sub_escaped}':force_style='{force_style}'"

    graph = (
        f"{bg};"
        f"[1:a]{audiofx}asplit=2[aw][aout];"
        "[aw]showwaves=s=1280x140:mode=cline:colors=white@0.45:rate=25[wv];"
        "[bg][wv]overlay=x=0:y=main_h-overlay_h[withwave]"
    )
    out_v = "[withwave]"
    if sub_filter:
        graph += f";[withwave]{sub_filter}[vout]"
        out_v = "[vout]"

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *inputs, "-i", str(instr),
        "-filter_complex", graph,
        "-map", out_v, "-map", "[aout]",
        # -shortest filter_complex'te sonsuz lavfi kaynağını durdurmaz; -t zorunlu
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "320k",
        str(mp4),
    ]
    subprocess.run(cmd, check=True)


def process(audio: Path, language: str, style: str = "karaoke", initial_prompt: str | None = None,
            srt_only: bool = False, use_lrclib: bool = True, transpose: int = 0) -> Path:
    slug = song_slug(audio)
    work = OUTPUT_DIR / slug
    work.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {audio.name} ===")
    original = work / f"original{audio.suffix.lower()}"
    if not original.exists():
        shutil.copy2(audio, original)
    print(f"[1/4] Orijinal -> {original.name}")

    vocals = work / "vocals.wav"
    instr = work / "instrumental.wav"
    separated = vocals.exists() and instr.exists()

    n_cues = 0
    sub: Path | None = None
    known_text: str | None = None          # zamanı olmayan bilinen söz (force-align girdisi)
    known_lines: list[dict] | None = None  # satır-düzeyi zamanlı bilinen söz

    if style == "karaoke" and use_lrclib and not srt_only:
        print("[2/4] Söz kaynakları: LDDC → LRCLIB → force-align → WhisperX")
        artist, title = parse_artist_title(audio)
        try:
            duration = probe_duration(audio) if audio.exists() else None
        except Exception as e:
            print(f"   süre okunamadı ({e}), süre-eşleşmesi atlanıyor", file=sys.stderr)
            duration = None

        # 1) LDDC: QQ/Kugou/NetEase gerçek kelime-düzeyi (VERBATIM) zamanlama
        if artist and title:
            lddc_segs = fetch_lddc_segments(artist, title, duration)
            if lddc_segs:
                n_cues = write_karaoke_lyrics(lddc_segs, work)
                sub = work / "lyrics.ass"

        # 2) LRCLIB: A2 kelime-düzeyi varsa direkt kullan;
        #    satır-düzeyi/düz metin force-align girdisi olarak saklanır
        if sub is None:
            a2_segs, known_lines, known_text = fetch_lrclib_lyrics(audio, artist, title, duration)
            if a2_segs:
                n_cues = write_karaoke_lyrics(a2_segs, work)
                sub = work / "lyrics.ass"

        # 3) Force-align: bilinen sözler WhisperX hizalayıcıyla vokallere oturtulur
        if sub is None and (known_lines is not None or known_text):
            if separated:
                print("[3/4] (mevcut stems kullanılıyor)")
            else:
                print("[3/4] Demucs ile vokal ayrımı (force-align için)")
                vocals, instr = separate(audio, work)
                separated = True
            print(f"[3/4] Bilinen sözler hizalanıyor (dil: {language})")
            res = align_known_text(vocals, work, language, known_text, known_lines)
            if res:
                sub, _, n_cues = res
                print(f"   -> {n_cues} altyazı kuyruğu ({sub.name})")
            else:
                print("   force-align başarısız, saf WhisperX'e düşülüyor")

    if sub is None:
        # 4) Saf WhisperX transkripsiyon (ayrıca srt_only ve word/sentence stilleri)
        if separated:
            print("[3/4] (atlandı — mevcut stems kullanılıyor)")
        else:
            print("[3/4] Demucs ile ayrım")
            vocals, instr = separate(audio, work)
            separated = True
        print(f"[3/4] WhisperX ile altyazı (stil: {style}, dil: {language}, model: {WHISPER_MODEL})")
        sub, _, n_cues = transcribe(vocals, work, language, style=style, initial_prompt=initial_prompt)
        print(f"   -> {n_cues} altyazı kuyruğu ({sub.name})")

    cover = find_cover(audio)
    mp4 = work / f"{slug}_karaoke.mp4"
    tone = f", ton: {transpose:+d}" if transpose else ""
    print(f"[4/4] FFmpeg ile video oluşturma (kapak: {cover.name if cover else 'siyah'}{tone})")
    make_video(instr, sub, cover, mp4, n_cues, style, transpose)

    print(f"Tamamlandı: {work}")
    return work


def main() -> int:
    p = argparse.ArgumentParser(description="Şarkıyı karaoke videosuna dönüştürür.")
    p.add_argument("song", nargs="?", help="Şarkı dosyası (input/ içinde değilse tam yol). Boşsa input/ işlenir.")
    p.add_argument("--language", "-l", default="tr", help="WhisperX dili (varsayılan: tr)")
    p.add_argument("--style", choices=["karaoke", "word", "sentence"], default="karaoke",
                   help="Altyazı stili: karaoke (cümle + sıralı kelime vurgusu, ASS), word (her kelime ayrı cue, SRT), sentence (cümle düzeyinde, SRT)")
    p.add_argument("--initial-prompt", default=None, help="Whisper'a başlangıç ipucu: şarkı adı, sanatçı, beklenen anahtar kelimeler (doğruluğu artırır).")
    p.add_argument("--srt-only", action="store_true", help="Sadece altyazıyı yeniden üret (mevcut vocals.wav kullanılır, Demucs atlanır). Hızlı iterasyon için.")
    p.add_argument("--no-lrclib", action="store_true", help="Çevrimiçi söz kaynaklarını (LDDC/LRCLIB/force-align) atla, doğrudan WhisperX transkripsiyonu kullan.")
    p.add_argument("--transpose", type=int, default=0, metavar="N",
                   help="Perde kaydırma, yarım ses aralığıyla (-12..+12, örn. --transpose -2 = 1 tam ses aşağı). Süre değişmez, altyazı zamanlaması geçerli kalır.")
    args = p.parse_args()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    else:
        print("UYARI: GPU bulunamadı, CPU kullanılacak (çok yavaş).")

    if args.song:
        audio = Path(args.song)
        if not audio.exists():
            audio = INPUT_DIR / args.song
        if not audio.exists():
            print(f"Dosya bulunamadı: {args.song}", file=sys.stderr)
            return 1
        process(audio.resolve(), args.language, args.style, args.initial_prompt, args.srt_only, not args.no_lrclib, args.transpose)
    else:
        songs = find_inputs()
        if not songs:
            print(f"input/ klasöründe şarkı yok. Şarkıyı {INPUT_DIR}/ içine koy.")
            return 1
        for s in songs:
            process(s, args.language, args.style, args.initial_prompt, args.srt_only, not args.no_lrclib, args.transpose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
