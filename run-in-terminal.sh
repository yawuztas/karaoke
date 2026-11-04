#!/usr/bin/env bash
# Terminal açıp launch.sh çalıştırır (.desktop Exec için).
cd "$(dirname "$0")"
exec gnome-terminal --wait --title="Karaoke" -- bash -c "cd '$(pwd)' && ./launch.sh; echo; echo 'Kapatmak icin Enter bas...'; read"
