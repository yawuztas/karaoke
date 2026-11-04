#!/usr/bin/env bash
# Karaoke başlatıcı — çift tıkla veya terminalden çalıştır.
# Akış: şarkı adı sor -> spotdl ile indir -> "Hazırla?" sor -> pipeline çalıştır.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/venv/bin/python"
SPOTDL="$ROOT/venv/bin/spotdl"
INPUT_DIR="$ROOT/input"
OUTPUT_DIR="$ROOT/output"

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

has_terminal() {
  [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]
}

ask_query() {
  if has_terminal && command -v zenity >/dev/null 2>&1; then
    zenity --entry --title="🎤 Karaoke" \
      --text="Spotify URL veya şarkı adı gir\n(örn: 'https://open.spotify.com/track/...' veya 'Tarkan Şımarık')\nİptal = çık" \
      --width=520
  else
    read -rp "Şarkı adı / Spotify URL (boş = çık): " q
    echo "$q"
  fi
}

ask_style() {
  local current="$1"
  if has_terminal && command -v zenity >/dev/null 2>&1; then
    zenity --list --radiolist --title="🎤 Altyazı Stili" \
      --text="Hangi altyazı stili kullanılsın?\n\nkaraoke: gerçek karaoke — cümle ekranda, kelimeler sırayla vurgulanır (ASS)\nword: her kelime ayrı ekranda (SRT)\nsentence: her cümle ayrı (SRT)" \
      --column="Seç" --column="Stil" \
      TRUE "karaoke" \
      FALSE "word" \
      FALSE "sentence" \
      --width=520 --height=240 2>/dev/null || echo "$current"
  else
    echo "$current"
  fi
}

ask_yesno() {
  local text="$1" default="${2:-false}"
  if has_terminal && command -v zenity >/dev/null 2>&1; then
    if [ "$default" = "true" ]; then
      zenity --question --title="🎤 Karaoke" --text="$text" --width=420
    else
      zenity --question --title="🎤 Karaoke" --text="$text" --default-cancel --width=420
    fi
  else
    local yn
    read -rp "$text [e/h]: " yn
    [[ "$yn" =~ ^[eEyY]$ ]]
  fi
}

info() {
  local text="$1"
  if has_terminal && command -v zenity >/dev/null 2>&1; then
    zenity --info --title="🎤 Karaoke" --text="$text" --width=500
  else
    echo "$text"
  fi
}

error() {
  local text="$1"
  if has_terminal && command -v zenity >/dev/null 2>&1; then
    zenity --error --title="🎤 Karaoke - Hata" --text="$text" --width=500
  else
    echo "HATA: $text" >&2
  fi
}

_norm() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9ığüöşçĞÜÖŞİÇ]/ /g' | tr -s ' '
}

pick_downloaded() {
  # İndirme sonrası doğru dosyayı seç:
  #  1) marker'dan yeni indirilen dosya(lar) — varsa en yenisi
  #  2) yoksa (spotdl 'zaten var' diyip atladıysa) sorguyla dosya adı eşleşmesi
  local marker="$1" query="$2"

  local newest
  newest=$(find "$INPUT_DIR" -maxdepth 1 -type f \
    \( -name '*.mp3' -o -name '*.flac' -o -name '*.m4a' -o -name '*.ogg' -o -name '*.wav' \) \
    -newer "$marker" -print 2>/dev/null | while IFS= read -r f; do
      stat -c '%Y %n' "$f"
    done | sort -rn | head -1 | cut -d' ' -f2-)
  if [ -n "$newest" ] && [ -f "$newest" ]; then
    echo "$newest"
    return 0
  fi

  # Yeni dosya yok → sorgu token'larıyla mevcut dosyaları puanla
  local norm best="" bestscore=0
  norm=$(_norm "$query")
  [ -z "$norm" ] && return 1

  local f fname score tok
  for f in "$INPUT_DIR"/*.mp3 "$INPUT_DIR"/*.flac "$INPUT_DIR"/*.m4a "$INPUT_DIR"/*.ogg "$INPUT_DIR"/*.wav; do
    [ -f "$f" ] || continue
    fname=$(_norm "$(basename "${f%.*}")")
    score=0
    for tok in $norm; do
      case "$fname" in *"$tok"*) score=$((score + 1));; esac
    done
    if [ "$score" -gt "$bestscore" ]; then
      bestscore=$score; best="$f"
    fi
  done
  [ -n "$best" ] && echo "$best"
}

download_song() {
  local query="$1"

  # Tüm kullanıcı mesajları stderr'e (fonksiyon stdout'unu kirletmesin — sadece path dönmeli)
  echo "" >&2
  echo "==> İndiriliyor: $query" >&2

  # İndirme öncesi işaret dosyası: sonrasında -newer ile gerçekten yeni inenleri buluruz
  local marker
  marker=$(mktemp "$INPUT_DIR/.dlmarker.XXXXXX")
  touch "$marker"

  local spotdl_out
  spotdl_out=$("$SPOTDL" download "$query" --output "$INPUT_DIR" --format mp3 --bitrate 320k --overwrite skip 2>&1 || true)
  if [ -n "$spotdl_out" ]; then
    echo "$spotdl_out" | tail -8 >&2
  fi

  local picked
  picked=$(pick_downloaded "$marker" "$query")
  rm -f "$marker"
  if [ -z "$picked" ] || [ ! -f "$picked" ]; then
    error "İndirilen ses dosyası bulunamadı.

İpucu: Aradığın şarkı zaten input/ klasöründe olabilir.
Adını net yazmayı dene: 'Sanatçı - Şarkı' (örn: Tarkan - Şımarık)"
    return 1
  fi
  echo "$picked"  # Fonksiyonun tek stdout satırı: dosya yolu
}

run_pipeline() {
  local audio="$1" style="${2:-karaoke}"
  echo ""
  echo "=================================================="
  echo "  KARAOKE PIPELINE"
  echo "  Dosya: $audio"
  echo "  Stil:  $style"
  echo "=================================================="

  local log="$ROOT/models/last-pipeline.log"
  # PYTHONUNBUFFERED=1 + stdbuf: log anlık yazılsın (pencere kapansa bile)
  if ! PYTHONUNBUFFERED=1 stdbuf -oL -eL "$PY" "$ROOT/karaoke.py" "$audio" --language tr --style "$style" 2>&1 | stdbuf -oL tee "$log"; then
    local tail
    tail=$(tail -25 "$log" 2>/dev/null | tr '\n' '|' | head -c 500)
    error "Pipeline başarısız oldu.\n\nLog: $log\n\nSon çıktı:\n$tail"
    return 1
  fi

  local slug
  slug=$(basename "$audio" | sed 's/\.[^.]*$//' | tr ' ' '_')
  local out="$OUTPUT_DIR/$slug/${slug}_karaoke.mp4"
  if [ -f "$out" ] && [ -s "$out" ]; then
    info "✅ Bitti!\n\nOrijinal: $OUTPUT_DIR/$slug/\nKaraoke video: $out"
  else
    local tail
    tail=$(tail -25 "$log" 2>/dev/null | tr '\n' '|' | head -c 500)
    error "Pipeline bitti ama video oluşmadı.\n\nLog: $log\n\nSon çıktı:\n$tail"
    return 1
  fi
}

main() {
  if [ ! -x "$PY" ]; then
    error "Python venv bulunamadı: $PY"
    exit 1
  fi
  if [ ! -x "$SPOTDL" ]; then
    error "spotdl bulunamadı: $SPOTDL"
    exit 1
  fi

  echo "🎤 Karaoke Yapıcı — $ROOT"
  echo "   GPU: $(command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'yok')"

  STYLE="karaoke"
  while true; do
    QUERY=$(ask_query)
    if [ -z "$QUERY" ]; then
      echo "Çıkılıyor."
      exit 0
    fi

    NEWEST=$(download_song "$QUERY" || true)
    if [ -z "$NEWEST" ]; then
      continue
    fi
    echo "İndirildi: $NEWEST"

    if ask_yesno "İndirildi: $(basename "$NEWEST")\n\nŞimdi karaoke videosu hazırlansın mı?\n(Demucs + WhisperX + FFmpeg — uzun sürer)"; then
      NEW_STYLE=$(ask_style "$STYLE")
      [ -n "$NEW_STYLE" ] && STYLE="$NEW_STYLE"
      run_pipeline "$NEWEST" "$STYLE"
    fi

    if ! ask_yesno "Başka şarkı işlenecek mi?"; then
      exit 0
    fi
  done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
