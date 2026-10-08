#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

constexpr int NMAX = 100;   // longest message handled
constexpr int EMAX = 64;    // most menu edges
constexpr int NCASE = 6;
constexpr int A2 = 676, A3 = 17576, A4 = 456976;
constexpr int NSET = NCASE * A4;  // settings per wheel order
constexpr int MAXPAIRS = 10;

static const char *H_WIRES[6] = {"", "EKMFLGDQVZNTOWYHXUSPAIBRCJ", "AJDKSIRUXBLHWTMCQGZNPYFVOE",
                                 "BDFHJLCPRTXVZNYEIWGAKMUSQO", "ESOVPZJAYQUIRHXLNFTGKDCMWB",
                                 "VZBRGITYUPSDNHLXAWMJQOFECK"};
static const int H_NOTCH[6] = {0, 16, 4, 21, 9, 25};
static const char *H_REFB = "YRUHQSLDPXNGOKMIEBFZCWVJAT";

struct Menu {
  int n;
  int8_t a[EMAX], b[EMAX], pos[EMAX];
  int8_t order[27];          // DFS order of menu letters, -1 terminated
  int8_t inc_n[26];
  int8_t inc[26][EMAX];
};

struct Stop {
  uint32_t setting;
  int8_t plug[26];           // -1 = not decided by the crib
  int16_t pad;
};

struct Done {
  int32_t score, outside;
  int8_t plug[26];
  int16_t pad;
};

// table[((cl*26+cm)*26+cr)*26 + x] = scrambler output for input x
inline void build_scrambler(int l, int m, int r, uint8_t *table) {
  static int F[3][26][26], R[3][26][26];
  const int w[3] = {l, m, r};
  for (int k = 0; k < 3; ++k)
    for (int off = 0; off < 26; ++off)
      for (int c = 0; c < 26; ++c) {
        int out = ((H_WIRES[w[k]][(c + off) % 26] - 'A') - off + 26) % 26;
        F[k][off][c] = out;
        R[k][off][out] = c;
      }
  for (int cl = 0; cl < 26; ++cl)
    for (int cm = 0; cm < 26; ++cm)
      for (int cr = 0; cr < 26; ++cr) {
        uint8_t *row = table + ((cl * 26 + cm) * 26 + cr) * 26;
        for (int x = 0; x < 26; ++x) {
          int c = F[2][cr][x];
          c = F[1][cm][c];
          c = F[0][cl][c];
          c = H_REFB[c] - 'A';
          c = R[0][cl][c];
          c = R[1][cm][c];
          c = R[2][cr][c];
          row[x] = (uint8_t)c;
        }
      }
}

inline int middle_window(int cs, int m) {
  return cs < 5 ? (H_NOTCH[m] - 4 + cs + 26) % 26 : (H_NOTCH[m] + 1) % 26;
}

// steps[(cs*26+wr)*NMAX*2 + 2*i] = left steps, +1 = middle steps taken by
// the time letter i is enciphered. active[cs*26+wr] = setting is distinct.
inline void build_steps(int m, int r, int n, uint8_t *steps, uint8_t *active) {
  for (int cs = 0; cs < NCASE; ++cs)
    for (int wr0 = 0; wr0 < 26; ++wr0) {
      int wm = middle_window(cs, m), wr = wr0, dl = 0, dm = 0;
      uint8_t *s = steps + (cs * 26 + wr0) * NMAX * 2;
      for (int i = 0; i < NMAX; ++i) {
        bool mid = wm == H_NOTCH[m], right = wr == H_NOTCH[r];
        if (mid) ++dl;
        if (mid || right) { ++dm; wm = (wm + 1) % 26; }
        wr = (wr + 1) % 26;
        s[2 * i] = (uint8_t)dl;
        s[2 * i + 1] = (uint8_t)dm;
      }
      active[cs * 26 + wr0] = (cs == 5) || (s[2 * (n - 1)] > 0);
    }
}

HD void decode(uint32_t idx, int &cs, int &cl, int &cm, int &cr, int &wr) {
  wr = idx % 26; idx /= 26;
  cr = idx % 26; idx /= 26;
  cm = idx % 26; idx /= 26;
  cl = idx % 26;
  cs = idx / 26;
}

HD int row_of(int cl, int cm, int cr, int i, const uint8_t *st) {
  return ((((cl + st[2 * i]) % 26) * 26 + (cm + st[2 * i + 1]) % 26) * 26 + (cr + i + 1) % 26) * 26;
}

struct Trail {
  int8_t p[26];
  int8_t t[26];
  int n, pairs;
};

HD bool assign(Trail &T, int a, int v) {
  if (T.p[a] >= 0) return T.p[a] == v;
  if (T.p[v] >= 0) return false;
  if (a != v) {
    if (T.pairs >= MAXPAIRS) return false;
    T.p[a] = (int8_t)v; T.p[v] = (int8_t)a;
    T.t[T.n++] = (int8_t)a; T.t[T.n++] = (int8_t)v;
    ++T.pairs;
  } else {
    T.p[a] = (int8_t)a;
    T.t[T.n++] = (int8_t)a;
  }
  return true;
}

// Collects stops; works on host and device.
struct StopSink {
  Stop *buf;
  unsigned int *count;
  unsigned int cap;
  uint32_t setting;
  HD void operator()(const int8_t *p) const {
#ifdef __CUDA_ARCH__
    unsigned int k = atomicAdd(count, 1u);
#else
    unsigned int k = (*count)++;
#endif
    if (k < cap) {
      buf[k].setting = setting;
      for (int i = 0; i < 26; ++i) buf[k].plug[i] = p[i];
    }
  }
};

// Depth-first plugboard deduction from the menu at one setting. Calls
// emit(plug) for every complete consistent assignment. Returns false if
// the node budget ran out.
template <class Emit>
HD bool bombe_setting(const Menu &M, const uint8_t *scr, const int *rows, Emit emit, int budget) {
  Trail T;
  for (int i = 0; i < 26; ++i) T.p[i] = -1;
  T.n = 0; T.pairs = 0;
  int8_t var[27], next[27], mark[27], pmark[27];
  int depth = 0, nodes = 0;
  var[0] = M.order[0]; next[0] = 0; mark[0] = 0; pmark[0] = 0;
  if (var[0] < 0) return true;
  while (depth >= 0) {
    while (T.n > mark[depth]) T.p[T.t[--T.n]] = -1;
    T.pairs = pmark[depth];
    if (next[depth] == 26) { --depth; continue; }
    if (++nodes > budget) return false;
    int value = next[depth]++;
    int begin = T.n;
    bool ok = assign(T, var[depth], value);
    for (int q = begin; ok && q < T.n; ++q) {
      int a = T.t[q];
      int pa = T.p[a];
      for (int k = 0; ok && k < M.inc_n[a]; ++k) {
        int e = M.inc[a][k];
        int other = M.a[e] == a ? M.b[e] : M.a[e];
        ok = assign(T, other, scr[rows[e] + pa]);
      }
    }
    if (!ok) continue;
    int u = -1;
    for (int k = 0; k < 26 && M.order[k] >= 0; ++k)
      if (T.p[M.order[k]] < 0) { u = M.order[k]; break; }
    if (u < 0) { emit(T.p); continue; }
    ++depth;
    var[depth] = (int8_t)u; next[depth] = 0; mark[depth] = (int8_t)T.n; pmark[depth] = (int8_t)T.pairs;
  }
  return true;
}

// Trigram score of the decrypt. ct[i] < 0 marks an unreadable cipher letter.
// lo..hi is the crib window; 'outside' counts trigrams not wholly inside it.
HD int score_plug(const int8_t *ct, int n, const int *rows, const uint8_t *scr, const int8_t *plug,
                  const int16_t *tri, int lo, int hi, int *outside) {
  int s = 0, o = 0, p1 = -1, p2 = -1;
  for (int i = 0; i < n; ++i) {
    int c = ct[i];
    if (c < 0) { p1 = p2 = -1; continue; }
    int m = plug[scr[rows[i] + plug[c]]];
    if (p2 >= 0) {
      int v = tri[(p2 * 26 + p1) * 26 + m];
      s += v;
      if (!(i - 2 >= lo && i < hi)) o += v;
    }
    p2 = p1; p1 = m;
  }
  if (outside) *outside = o;
  return s;
}

// Complete the plugboard over letters the crib left open, by hill climbing.
HD int complete(int8_t *plug, const int8_t *ct, int n, const int *rows, const uint8_t *scr,
                const int16_t *tri, int lo, int hi, int *outside) {
  int8_t freel[26];
  int nf = 0, pairs = 0;
  for (int i = 0; i < 26; ++i) {
    if (plug[i] < 0) { plug[i] = (int8_t)i; freel[nf++] = (int8_t)i; }
  }
  for (int i = 0; i < 26; ++i) if (plug[i] > i) ++pairs;
  int best = score_plug(ct, n, rows, scr, plug, tri, lo, hi, nullptr);
  int8_t save[26], cand[26];
  for (int round = 0; round < 6; ++round) {
    bool improved = false;
    for (int i = 0; i < nf; ++i)
      for (int j = i + 1; j < nf; ++j) {
        int a = freel[i], b = freel[j];
        for (int k = 0; k < 26; ++k) save[k] = plug[k];
        int bestmove = best, bestpairs = pairs;
        bool found = false;
        if (plug[a] == b) {
          plug[a] = (int8_t)a; plug[b] = (int8_t)b;
          int sc = score_plug(ct, n, rows, scr, plug, tri, lo, hi, nullptr);
          if (sc > bestmove) { bestmove = sc; bestpairs = pairs - 1; found = true; for (int k = 0; k < 26; ++k) cand[k] = plug[k]; }
        } else {
          int x = plug[a], y = plug[b];
          int np = pairs;
          if (x != a) { plug[x] = (int8_t)x; plug[a] = (int8_t)a; --np; }
          if (y != b) { plug[y] = (int8_t)y; plug[b] = (int8_t)b; --np; }
          plug[a] = (int8_t)b; plug[b] = (int8_t)a; ++np;
          if (np <= MAXPAIRS) {
            int sc = score_plug(ct, n, rows, scr, plug, tri, lo, hi, nullptr);
            if (sc > bestmove) { bestmove = sc; bestpairs = np; found = true; for (int k = 0; k < 26; ++k) cand[k] = plug[k]; }
          }
          if (x != a && y != b && np + 1 <= MAXPAIRS) {
            plug[x] = (int8_t)y; plug[y] = (int8_t)x;
            int sc = score_plug(ct, n, rows, scr, plug, tri, lo, hi, nullptr);
            if (sc > bestmove) { bestmove = sc; bestpairs = np + 1; found = true; for (int k = 0; k < 26; ++k) cand[k] = plug[k]; }
          }
        }
        if (found) {
          for (int k = 0; k < 26; ++k) plug[k] = cand[k];
          best = bestmove; pairs = bestpairs; improved = true;
        } else {
          for (int k = 0; k < 26; ++k) plug[k] = save[k];
        }
      }
    if (!improved) break;
  }
  return score_plug(ct, n, rows, scr, plug, tri, lo, hi, outside);
}
