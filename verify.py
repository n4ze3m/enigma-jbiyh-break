from enigma import Enigma

PLUGS = "AU BI CY ET FL HK MP NR OX QV"
ct = open("data/JBIYH.txt").read().strip()
key = Enigma("541", "EXT", "BSB", PLUGS).crypt("NTK")
print("message key:", key)
print(Enigma("541", "EXT", key, PLUGS).crypt(ct))
