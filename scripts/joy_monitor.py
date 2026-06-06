# -*- coding: utf-8 -*-
"""
joy_monitor.py — Kumanda eksen/buton izleyici.

Kullanim: python scripts/joy_monitor.py
Switch'i cevir -> hangi eksen/buton degistigini gorursun.
CTRL+C ile cik.
"""
import ctypes
from ctypes import wintypes
import time

winmm = ctypes.windll.winmm

class JOYINFOEX(ctypes.Structure):
    _fields_ = [
        ("dwSize",         wintypes.DWORD),
        ("dwFlags",        wintypes.DWORD),
        ("dwXpos",         wintypes.DWORD),
        ("dwYpos",         wintypes.DWORD),
        ("dwZpos",         wintypes.DWORD),
        ("dwRpos",         wintypes.DWORD),
        ("dwUpos",         wintypes.DWORD),
        ("dwVpos",         wintypes.DWORD),
        ("dwButtons",      wintypes.DWORD),
        ("dwButtonNumber", wintypes.DWORD),
        ("dwPOV",          wintypes.DWORD),
        ("dwReserved1",    wintypes.DWORD),
        ("dwReserved2",    wintypes.DWORD),
    ]

JOY_RETURNALL = 0x00FF

def read_joy(joy_id):
    info = JOYINFOEX()
    info.dwSize  = ctypes.sizeof(JOYINFOEX)
    info.dwFlags = JOY_RETURNALL
    ret = winmm.joyGetPosEx(joy_id, ctypes.byref(info))
    if ret != 0:
        return None
    return {
        "X": info.dwXpos, "Y": info.dwYpos,
        "Z": info.dwZpos, "R": info.dwRpos,
        "U": info.dwUpos, "V": info.dwVpos,
        "btn": info.dwButtons,
    }

# Hangi slot'ta kumanda var?
JOY_ID = None
for i in range(8):
    if read_joy(i) is not None:
        JOY_ID = i
        print(f"Kumanda slot #{i}'de bulundu.")
        break

if JOY_ID is None:
    print("Hic kumanda bulunamadi! USB bagli mi?")
    exit(1)

print("Eksen/buton izleniyor... Switch'i cevir -> degisen eksen gorulecek.")
print("Cikis icin CTRL+C\n")

prev = read_joy(JOY_ID)
DEGISIM_ESIGI = 3000   # kucuk gurultu/titreme filtresi

try:
    while True:
        cur = read_joy(JOY_ID)
        if cur is None:
            print("Kumanda baglantiyi kesti!")
            break

        for eksen in ["X", "Y", "Z", "R", "U", "V"]:
            delta = abs(cur[eksen] - prev[eksen])
            if delta > DEGISIM_ESIGI:
                pct = (cur[eksen] / 65535.0) * 100.0
                print(f"  EKSEN {eksen}: {cur[eksen]:6d}  ({pct:5.1f}%)   delta={delta}")

        if cur["btn"] != prev["btn"]:
            print(f"  BUTON degisti: {bin(prev['btn'])} -> {bin(cur['btn'])}")

        prev = cur
        time.sleep(0.05)   # 20 Hz tarama

except KeyboardInterrupt:
    print("\nCikiyor.")
