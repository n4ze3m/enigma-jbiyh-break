#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
using namespace std;

static const char *WIRES[6] = {"", "EKMFLGDQVZNTOWYHXUSPAIBRCJ", "AJDKSIRUXBLHWTMCQGZNPYFVOE",
                               "BDFHJLCPRTXVZNYEIWGAKMUSQO", "ESOVPZJAYQUIRHXLNFTGKDCMWB",
                               "VZBRGITYUPSDNHLXAWMJQOFECK"};
static const int NOTCH[6] = {0, 16, 4, 21, 9, 25};
static const char *REFB = "YRUHQSLDPXNGOKMIEBFZCWVJAT";
static int FW[6][26], BW[6][26];

struct Machine {
  int rot[3], ring[3], pos[3];
  int plug[26];
  void step() {
    bool mid = pos[1] == NOTCH[rot[1]], right = pos[2] == NOTCH[rot[2]];
    if (mid) pos[0] = (pos[0] + 1) % 26;
    if (mid || right) pos[1] = (pos[1] + 1) % 26;
    pos[2] = (pos[2] + 1) % 26;
  }
  int crypt(int c) {
    step();
    int x = plug[c];
    for (int i = 2; i >= 0; --i) { int off = (pos[i] - ring[i] + 26) % 26; x = (FW[rot[i]][(x + off) % 26] - off + 26) % 26; }
    x = REFB[x] - 'A';
    for (int i = 0; i < 3; ++i) { int off = (pos[i] - ring[i] + 26) % 26; x = (BW[rot[i]][(x + off) % 26] - off + 26) % 26; }
    return plug[x];
  }
};

static unordered_set<string> VOC;

static int words(const string &s) {
  int n = s.size();
  vector<int> best(n + 1, 0);
  for (int i = 1; i <= n; ++i) {
    int b = best[i - 1];
    for (int L = 4; L <= min(18, i); ++L)
      if (VOC.count(s.substr(i - L, L))) b = max(b, best[i - L] + (L - 3));
    best[i] = b;
  }
  return best[n];
}

static int dbl(const string &s) {
  int n = s.size(), best = 0;
  for (int L = 4; L <= n / 2; ++L) {
    bool hit = false;
    for (int i = 0; i + L <= n && !hit; ++i)
      for (int j = i + L; j + L <= n; ++j)
        if (s.compare(i, L, s, j, L) == 0) { hit = true; break; }
    if (hit) best = L; else break;
  }
  return best;
}

static string norm(string w) {
  string o;
  for (size_t i = 0; i < w.size(); ++i) {
    unsigned char c = w[i];
    if (c >= 'a' && c <= 'z') o += char(c - 32);
    else if (c >= 'A' && c <= 'Z') o += char(c);
    else if (c == 0xC3 && i + 1 < w.size()) {  // UTF-8 umlauts and sharp s
      unsigned char d = w[++i];
      if (d == 0xA4 || d == 0x84) o += "AE"; else if (d == 0xB6 || d == 0x96) o += "OE";
      else if (d == 0xBC || d == 0x9C) o += "UE"; else if (d == 0x9F) o += "SZ";
    }
  }
  return o;
}

static vector<string> split(const string &s, char d) { vector<string> v; stringstream ss(s); string t; while (getline(ss, t, d)) v.push_back(t); return v; }

int main(int argc, char **argv) {
  if (argc < 6) { fprintf(stderr, "usage: wordcheck VOCAB LAZYKEYS GS1,GS2 EK1,EK2 N < hits\n"); return 1; }
  for (int r = 1; r <= 5; ++r) for (int i = 0; i < 26; ++i) { FW[r][i] = WIRES[r][i] - 'A'; BW[r][FW[r][i]] = i; }
  { ifstream f(argv[1]); string w; long long fq;
    while (f >> w) {
      if (!(f >> fq)) break;
      string u = norm(w);
      if (u.size() < 4) continue;
      VOC.insert(u);
      string q = u; size_t p; while ((p = q.find("CH")) != string::npos) q.replace(p, 2, "Q"); VOC.insert(q);
    } }
  set<string> lazy; { ifstream f(argv[2]); string k; while (f >> k) lazy.insert(k); }
  vector<string> gss = split(argv[3], ','), eks = split(argv[4], ',');
  int n = atoi(argv[5]);
  string line;
  while (getline(cin, line)) {
    auto v = split(line, '\t');
    if (v.size() < 5) continue;
    string pt = v[4];
    int l = v[1][0] - '0', m = v[1][1] - '0', r = v[1][2] - '0';
    unsigned idx = stoul(v[2]);
    int wr = idx % 26, cr = idx / 26 % 26, cm = idx / 676 % 26, cl = idx / 17576 % 26, cs = idx / 456976;
    int plug[26]; for (int i = 0; i < 26; ++i) plug[i] = i;
    for (size_t i = 0; i + 1 < v[3].size(); i += 2) { int a = v[3][i] - 'A', b = v[3][i + 1] - 'A'; plug[a] = b; plug[b] = a; }
    // indicator: ringR fixed by (wr, cr); try every ringL, ringM giving the same body stepping
    int ringR = (wr - cr + 26) % 26;
    string found = "-", foundRings = "-";
    for (int ringM = 0; ringM < 26 && found == "-"; ++ringM) {
      int wm = (cm + ringM) % 26;
      // body stepping must match the location's turnover case
      { int x = wm, y = wr; bool left = false;
        for (int i = 0; i < n; ++i) { bool mid = x == NOTCH[m], rt = y == NOTCH[r]; if (mid) left = true; if (mid || rt) x = (x + 1) % 26; y = (y + 1) % 26; }
        int want = left ? ((wm - (NOTCH[m] - 4)) % 26 + 26) % 26 : 5;
        if (want != cs) continue; }
      for (int ringL = 0; ringL < 26 && found == "-"; ++ringL) {
        int key[3] = {(cl + ringL) % 26, wm, wr};
        for (auto &gs : gss) for (auto &ek : eks) {
          Machine M; M.rot[0] = l; M.rot[1] = m; M.rot[2] = r; M.ring[0] = ringL; M.ring[1] = ringM; M.ring[2] = ringR;
          for (int i = 0; i < 3; ++i) M.pos[i] = gs[i] - 'A';
          memcpy(M.plug, plug, sizeof plug);
          bool ok = true;
          for (int i = 0; i < 3 && ok; ++i) ok = M.crypt(ek[i] - 'A') == key[i];
          if (ok) { found = string(1, 'A' + key[0]) + char('A' + key[1]) + char('A' + key[2]);
            foundRings = string(1, 'A' + ringL) + char('A' + ringM) + char('A' + ringR) + "/" + gs + "/" + ek; break; }
        }
      }
    }
    int isl = found != "-" && lazy.count(found);
    printf("%d\t%d\t%d\t%d\t%s\t%s\t%s\n", words(pt), dbl(pt), found != "-", isl, found.c_str(), foundRings.c_str(), line.c_str());
  }
}
