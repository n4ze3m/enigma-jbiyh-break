from dataclasses import dataclass

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
WIRINGS = ["", "EKMFLGDQVZNTOWYHXUSPAIBRCJ", "AJDKSIRUXBLHWTMCQGZNPYFVOE",
           "BDFHJLCPRTXVZNYEIWGAKMUSQO", "ESOVPZJAYQUIRHXLNFTGKDCMWB",
           "VZBRGITYUPSDNHLXAWMJQOFECK"]
NOTCHES = ["", "Q", "E", "V", "J", "Z"]
REFLECTORS = {"B": "YRUHQSLDPXNGOKMIEBFZCWVJAT", "C": "FVPJIAOYEDRZXWGCTKUQSBNMHL"}

def nums(text):
    return [ALPHABET.index(c) for c in text.upper()]

def letters(values):
    return "".join(ALPHABET[x] for x in values)

class Enigma:
    def __init__(self, rotors, rings="AAA", position="AAA", plugs="", reflector="B"):
        self.rotors = tuple(map(int, rotors))
        assert len(self.rotors) == 3 and len(set(self.rotors)) == 3
        self.rings = nums(rings)
        self.pos = nums(position)
        self.forward = [nums(WIRINGS[r]) for r in self.rotors]
        self.reverse = [[w.index(i) for i in range(26)] for w in self.forward]
        self.notches = [nums(NOTCHES[r]) for r in self.rotors]
        self.reflector = nums(REFLECTORS[reflector])
        p = plugs.replace(" ", "").upper()
        assert len(p) % 2 == 0 and len(set(p)) == len(p)
        self.plug = list(range(26))
        for a, b in zip(p[::2], p[1::2]):
            a, b = nums(a+b)
            self.plug[a], self.plug[b] = b, a

    def step(self):
        middle = self.pos[1] in self.notches[1]
        right = self.pos[2] in self.notches[2]
        if middle:
            self.pos[0] = (self.pos[0] + 1) % 26
        if middle or right:
            self.pos[1] = (self.pos[1] + 1) % 26
        self.pos[2] = (self.pos[2] + 1) % 26

    def crypt(self, text):
        out = []
        for c in text.upper():
            if c not in ALPHABET and c != "-":
                continue
            self.step()
            if c == "-":
                out.append(c)
                continue
            x = self.plug[ord(c)-65]
            for i in (2, 1, 0):
                off = self.pos[i] - self.rings[i]
                x = (self.forward[i][(x+off) % 26]-off) % 26
            x = self.reflector[x]
            for i in (0, 1, 2):
                off = self.pos[i] - self.rings[i]
                x = (self.reverse[i][(x+off) % 26]-off) % 26
            out.append(ALPHABET[self.plug[x]])
        return "".join(out)

def from_key(key, plugs=""):
    reflector, rotors, rings, position = key.split(":")
    return Enigma(rotors, rings.rjust(3, "A"), position, plugs, reflector)

def selftest():
    assert Enigma("123").crypt("AAAAA") == "BDZGO"
    assert Enigma("123").crypt("HELLOWORLD") == "ILBDAAMTAZ"
    m = Enigma("123", position="ADU")
    trace = []
    for _ in range(3):
        m.step(); trace.append(letters(m.pos))
    assert trace == ["ADV", "AEW", "BFX"], trace
    p = "CY EL FH GS IJ KQ MW PV RZ TU"
    assert Enigma("432", "PKF", "OKF", p).crypt("QLV") == "PIK"
    cipher = "PSQDBCSFKHFJOMVCJAUXTOTQBBPBWACHZYXH"
    plain = "ABENDMELDUNGENENTFALLENXHARTJENSTEIN"
    assert from_key("B:432:RIT:VOR", "AHBODPEXFNJQKSLRMUTZ").crypt(cipher) == plain
    assert from_key("B:432:RIT:VOR", "AHBODPEXFNJQKSLRMUTZ").crypt(plain) == cipher
    print("PASS: standard vectors, double stepping, historical indicator, historical ciphertext and re-encryption")

if __name__ == "__main__":
    selftest()
