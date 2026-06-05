#!/usr/bin/env bash
echo "=== whoami ==="; whoami
echo "=== tools ==="
for t in cmake g++ gcc make pkg-config ffmpeg; do printf '%s: ' "$t"; command -v "$t" || echo MISSING; done
echo "=== opencv ==="; pkg-config --modversion opencv4 2>/dev/null || echo "opencv4 MISSING"
echo "=== sudo nopasswd? ==="; sudo -n true 2>/dev/null && echo "SUDO OK (nopasswd)" || echo "sudo needs password"
echo "=== win host ip ==="; ip route show default | awk '{print $3}'
echo "=== display ==="; echo "DISPLAY=$DISPLAY"
