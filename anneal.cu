#include "hunt_core.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(3); } } while (0)

#ifndef WARPS
#define WARPS 16                 
#endif
constexpr int NS = 64;            
constexpr int NBINS = 4096;       

struct Hit { uint32_t idx; int32_t score; int8_t plug[26]; int16_t pad; };

__constant__ int8_t c_ct[NMAX];
__constant__ int c_n;
__constant__ int c_R, c_K;
__constant__ float c_T0, c_T1;
__constant__ uint8_t c_active[NCASE * 26];

__device__ __forceinline__ uint32_t xs(uint32_t &s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }


__device__ __forceinline__ int dscore(const uint8_t *S, const uint8_t *P, int lane, const uint8_t *T3, const uint8_t *T2) {
  int n = c_n;
  int a = P[S[P[c_ct[0] * 32 + lane]] * 32 + lane];
  int b = P[S[26 + P[c_ct[1] * 32 + lane]] * 32 + lane];
  int s = T2[a * 26 + b];
#pragma unroll 4
  for (int i = 2; i < n; ++i) {
    int c = P[S[i * 26 + P[c_ct[i] * 32 + lane]] * 32 + lane];
    s += T3[a * 676 + b * 26 + c];
    a = b; b = c;
  }
  return s;
}

struct Move { int8_t l[8], v[8]; int k; };

__device__ __forceinline__ void setp(uint8_t *P, int lane, int x, int y, Move &m) {
  m.l[m.k] = (int8_t)x; m.v[m.k] = (int8_t)P[x * 32 + lane]; ++m.k;
  P[x * 32 + lane] = (uint8_t)y;
}
__device__ __forceinline__ void undo(uint8_t *P, int lane, const Move &m) {
  for (int j = m.k - 1; j >= 0; --j) P[m.l[j] * 32 + lane] = (uint8_t)m.v[j];
}

__device__ int dclimb(const uint8_t *S, uint8_t *P, int lane, const uint8_t *T3, const uint8_t *T2, int pairs, int best) {
  for (int round = 0; round < 10; ++round) {
    bool improved = false;
    for (int a = 0; a < 26; ++a) for (int b = a + 1; b < 26; ++b) {
      int pa = P[a * 32 + lane], pb = P[b * 32 + lane];
      if (pa == b) {
        Move m; m.k = 0; setp(P, lane, a, a, m); setp(P, lane, b, b, m);
        int sc = dscore(S, P, lane, T3, T2);
        if (sc > best) { best = sc; --pairs; improved = true; } else undo(P, lane, m);
        continue;
      }
      // variant 1: (a,b), orphans unplugged; variant 2: also (x,y)
      Move m; m.k = 0; int np = pairs;
      int x = pa, y = pb;
      if (x != a) { setp(P, lane, x, x, m); setp(P, lane, a, a, m); --np; }
      if (y != b) { setp(P, lane, y, y, m); setp(P, lane, b, b, m); --np; }
      setp(P, lane, a, b, m); setp(P, lane, b, a, m); ++np;
      int s1 = np <= MAXPAIRS ? dscore(S, P, lane, T3, T2) : -1 << 30;
      int s2 = -1 << 30;
      if (x != a && y != b && np + 1 <= MAXPAIRS) {
        Move m2; m2.k = 0; setp(P, lane, x, y, m2); setp(P, lane, y, x, m2);
        s2 = dscore(S, P, lane, T3, T2);
        if (s2 > best && s2 >= s1) { best = s2; pairs = np + 1; improved = true; continue; }
        undo(P, lane, m2);
      }
      if (s1 > best) { best = s1; pairs = np; improved = true; } else undo(P, lane, m);
    }
    if (!improved) break;
  }
  return best;
}

constexpr int WARP_SMEM = NS * 26 + 3 * 26 * 32;
constexpr int SMEM = A3 + A2 + WARPS * WARP_SMEM;

__global__ void __launch_bounds__(WARPS * 32, 1) k_anneal(const uint8_t *__restrict__ scr, const uint8_t *__restrict__ steps,
    const uint8_t *__restrict__ t3g, uint32_t base, uint32_t count, uint32_t seed, int thr, Hit *hits, unsigned *nhit,
    unsigned cap, unsigned *hist, int verbose) {
  extern __shared__ uint8_t smem[];
  uint8_t *T3 = smem, *T2 = smem + A3;
  for (int i = threadIdx.x; i < (A3 + A2) / 4; i += blockDim.x) ((uint32_t *)smem)[i] = ((const uint32_t *)t3g)[i];
  __syncthreads();
  int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint8_t *wsm = smem + A3 + A2 + w * WARP_SMEM;
  uint32_t loc = blockIdx.x * WARPS + w;
  if (loc >= count) return;
  uint32_t idx = base + loc;
  int cs, cl, cm, cr, wr;
  decode(idx, cs, cl, cm, cr, wr);
  if (!c_active[cs * 26 + wr]) return;
  const uint8_t *st = steps + (cs * 26 + wr) * NMAX * 2;
  int n = c_n;
  uint8_t *S = wsm, *P = wsm + NS * 26, *B = P + 26 * 32, *A = B + 26 * 32;
  for (int i = lane; i < n; i += 32) {
    int row = row_of(cl, cm, cr, i, st);
    for (int x = 0; x < 26; ++x) S[i * 26 + x] = scr[row + x];
  }
  __syncwarp();
  uint32_t rng = (seed ^ (idx * 0x9E3779B1u) ^ (lane * 0x85EBCA77u)) | 1u;
  for (int k = 0; k < 8; ++k) xs(rng);
  int bestAll = -1 << 30;
  int runs = (c_R + 31) / 32;
  for (int run = 0; run < runs; ++run) {
    if (run * 32 + lane >= c_R) break;
    for (int x = 0; x < 26; ++x) P[x * 32 + lane] = (uint8_t)x;
    int pairs = 0, cur = dscore(S, P, lane, T3, T2);
    int bs = cur;
    for (int x = 0; x < 26; ++x) B[x * 32 + lane] = (uint8_t)x;
    int K = c_K;
    float T0 = c_T0, T1 = c_T1, dT = (T0 - T1) / K;
    float T = T0;
    for (int k = 0; k < K; ++k, T -= dT) {
      uint32_t r = xs(rng);
      int a = (r & 0xffff) % 26, b = ((r >> 16) & 0xffff) % 26;
      if (a == b) continue;
      Move m; m.k = 0; int np = pairs;
      int pa = P[a * 32 + lane];
      if (pa == b) { setp(P, lane, a, a, m); setp(P, lane, b, b, m); --np; }
      else {
        int x = pa, y = P[b * 32 + lane];
        if (x != a) { setp(P, lane, x, x, m); setp(P, lane, a, a, m); --np; }
        if (y != b) { setp(P, lane, y, y, m); setp(P, lane, b, b, m); --np; }
        setp(P, lane, a, b, m); setp(P, lane, b, a, m); ++np;
        if (x != a && y != b && (xs(rng) & 1) && np < MAXPAIRS) {
          // reuse the slots of the unplug entries: x->y, y->x
          P[x * 32 + lane] = (uint8_t)y; P[y * 32 + lane] = (uint8_t)x; ++np;
        }
      }
      if (np > MAXPAIRS) { undo(P, lane, m); continue; }
      int s = dscore(S, P, lane, T3, T2);
      int d = s - cur;
      if (d >= 0 || (xs(rng) & 0xffffff) * (1.0f / 16777216.0f) < __expf(d / T)) {
        cur = s; pairs = np;
        if (s > bs) { bs = s; for (int x = 0; x < 26; ++x) B[x * 32 + lane] = P[x * 32 + lane]; }
      } else undo(P, lane, m);
    }
    // polish the best state of this run
    for (int x = 0; x < 26; ++x) P[x * 32 + lane] = B[x * 32 + lane];
    int bp = 0; for (int x = 0; x < 26; ++x) bp += P[x * 32 + lane] > x;
    int fs = dclimb(S, P, lane, T3, T2, bp, bs);
    if (verbose) {
      char pl[21]; int q = 0; for (int x = 0; x < 26 && q < 20; ++x) { int y = P[x * 32 + lane]; if (y > x) { pl[q++] = 'A' + x; pl[q++] = 'A' + y; } }
      pl[q] = 0;
      printf("run %d lane %d: %d %s\n", run, lane, fs, pl);
    }
    if (fs > bestAll) { bestAll = fs; for (int x = 0; x < 26; ++x) A[x * 32 + lane] = P[x * 32 + lane]; }
  }
  // warp argmax
  int v = bestAll, who = lane;
  for (int o = 16; o > 0; o >>= 1) {
    int v2 = __shfl_xor_sync(0xffffffff, v, o), w2 = __shfl_xor_sync(0xffffffff, who, o);
    if (v2 > v || (v2 == v && w2 < who)) { v = v2; who = w2; }
  }
  if (lane == who) {
    int bin = max(0, min(NBINS - 1, v * 8 / 32));
    atomicAdd(hist + bin, 1u);
    if (v >= thr) {
      unsigned k = atomicAdd(nhit, 1u);
      if (k < cap) {
        hits[k].idx = idx; hits[k].score = v * 8;
        for (int x = 0; x < 26; ++x) hits[k].plug[x] = (int8_t)A[x * 32 + lane];
      }
    }
  }
}

static const int ORDERS[60][3] = {
#define O(a, b, c) {a, b, c},
    O(1,2,3) O(1,2,4) O(1,2,5) O(1,3,2) O(1,3,4) O(1,3,5) O(1,4,2) O(1,4,3) O(1,4,5) O(1,5,2) O(1,5,3) O(1,5,4)
    O(2,1,3) O(2,1,4) O(2,1,5) O(2,3,1) O(2,3,4) O(2,3,5) O(2,4,1) O(2,4,3) O(2,4,5) O(2,5,1) O(2,5,3) O(2,5,4)
    O(3,1,2) O(3,1,4) O(3,1,5) O(3,2,1) O(3,2,4) O(3,2,5) O(3,4,1) O(3,4,2) O(3,4,5) O(3,5,1) O(3,5,2) O(3,5,4)
    O(4,1,2) O(4,1,3) O(4,1,5) O(4,2,1) O(4,2,3) O(4,2,5) O(4,3,1) O(4,3,2) O(4,3,5) O(4,5,1) O(4,5,2) O(4,5,3)
    O(5,1,2) O(5,1,3) O(5,1,4) O(5,2,1) O(5,2,3) O(5,2,4) O(5,3,1) O(5,3,2) O(5,3,4) O(5,4,1) O(5,4,2) O(5,4,3)
#undef O
};

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: anneal CIPHER_FILE OUT_DIR [key=value ...]\n"); return 1; }
  map<string, string> opt = {{"orders", "ALL"}, {"R", "64"}, {"K", "12500"}, {"T0", "900"}, {"T1", "1"},
                             {"WU", "8"}, {"WB", "8"}, {"thr", "19000"}, {"cases", "5"}, {"seed", "1"},
                             {"only", ""}, {"tri", "data/Tri_1941_ln.txt"}, {"bi", "data/Bi_1941_ln.txt"}, {"uni", "data/Uni_1941_ln.txt"}};
  for (int i = 3; i < argc; ++i) { string a = argv[i]; size_t e = a.find('='); if (e == string::npos) { fprintf(stderr, "bad option %s\n", argv[i]); return 1; } opt[a.substr(0, e)] = a.substr(e + 1); }
  string raw; { ifstream f(argv[1]); string line; while (getline(f, line)) raw += line; }
  vector<int8_t> ct; for (char c : raw) if (c >= 'A' && c <= 'Z') ct.push_back(c - 'A');
  int n = ct.size();
  if (n < 3 || n > NS) { fprintf(stderr, "bad ciphertext length %d\n", n); return 1; }
  string outdir = argv[2];
  // score tables
  vector<int> TRI(A3, 0), BI(A2, 0), UNI(26, 0);
  { ifstream f(opt["tri"]); string t; int v; while (f >> t >> v) TRI[((t[0]-'A')*26+(t[1]-'A'))*26+(t[2]-'A')] = v; }
  { ifstream f(opt["bi"]); string t; int v; while (f >> t >> v) BI[(t[0]-'A')*26+(t[1]-'A')] = v; }
  { ifstream f(opt["uni"]); string t; int v; while (f >> t >> v) UNI[t[0]-'A'] = v; }
  int WU = stoi(opt["WU"]), WB = stoi(opt["WB"]);
  vector<uint8_t> c3(A3 + A2 + 4, 0);
  auto q8 = [](int v) { return (uint8_t)min(255, (v + 4) / 8); };
  for (int a = 0; a < 26; ++a) for (int b = 0; b < 26; ++b) {
    c3[A3 + a*26+b] = q8((WU * UNI[a] + WB * BI[a*26+b] + WU * UNI[b]) / 16);
    for (int c = 0; c < 26; ++c) c3[(a*26+b)*26+c] = q8(TRI[(a*26+b)*26+c] + (WB * BI[b*26+c] + WU * UNI[c]) / 16); }
  int R = stoi(opt["R"]), K = stoi(opt["K"]), thr = stoi(opt["thr"]);
  float T0 = stof(opt["T0"]) / 8, T1 = stof(opt["T1"]) / 8;  // score units of 8 points
  thr = thr / 8;
  uint32_t seed = (uint32_t)stoul(opt["seed"]);
  CK(cudaMemcpyToSymbol(c_ct, ct.data(), n)); CK(cudaMemcpyToSymbol(c_n, &n, 4));
  CK(cudaMemcpyToSymbol(c_R, &R, 4)); CK(cudaMemcpyToSymbol(c_K, &K, 4));
  CK(cudaMemcpyToSymbol(c_T0, &T0, 4)); CK(cudaMemcpyToSymbol(c_T1, &T1, 4));
  vector<int> orders;
  if (opt["orders"] == "ALL") for (int o = 0; o < 60; ++o) orders.push_back(o);
  else { stringstream ss(opt["orders"]); string t; while (getline(ss, t, ',')) for (int o = 0; o < 60; ++o)
    if (ORDERS[o][0] == t[0]-'0' && ORDERS[o][1] == t[1]-'0' && ORDERS[o][2] == t[2]-'0') orders.push_back(o); }
  vector<int> cases; for (char c : opt["cases"]) cases.push_back(c - '0');
  uint8_t *d_scr, *d_steps, *d_c3; Hit *d_hits; unsigned *d_nhit, *d_hist;
  const unsigned cap = 4000000;
  CK(cudaMalloc(&d_scr, (size_t)A4 * 26)); CK(cudaMalloc(&d_steps, NCASE * 26 * NMAX * 2));
  CK(cudaMalloc(&d_c3, c3.size())); CK(cudaMemcpy(d_c3, c3.data(), c3.size(), cudaMemcpyHostToDevice));
  CK(cudaFuncSetAttribute(k_anneal, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  CK(cudaMalloc(&d_hits, sizeof(Hit) * (size_t)cap)); CK(cudaMalloc(&d_nhit, 4)); CK(cudaMalloc(&d_hist, NBINS * 4));
  vector<uint8_t> scr((size_t)A4 * 26), steps(NCASE * 26 * NMAX * 2), active(NCASE * 26);
  auto t0 = chrono::steady_clock::now();
  FILE *log = fopen((outdir + "/progress.log").c_str(), "a");
  fprintf(log, "start n=%d R=%d K=%d T0=%.0f T1=%.0f WU=%d WB=%d thr=%d cases=%s seed=%u orders=%zu\n", n, R, K, T0, T1, WU, WB, thr, opt["cases"].c_str(), seed, orders.size());
  fflush(log);
  for (int o : orders) {
    int l = ORDERS[o][0], m = ORDERS[o][1], r = ORDERS[o][2];
    char name[64]; snprintf(name, sizeof name, "%s/order-%d%d%d", outdir.c_str(), l, m, r);
    if (opt["only"].empty()) { FILE *done = fopen((string(name) + ".done").c_str(), "r"); if (done) { fclose(done); continue; } }
    build_scrambler(l, m, r, scr.data()); build_steps(m, r, n, steps.data(), active.data());
    CK(cudaMemcpy(d_scr, scr.data(), scr.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_steps, steps.data(), steps.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpyToSymbol(c_active, active.data(), NCASE * 26));
    CK(cudaMemset(d_nhit, 0, 4)); CK(cudaMemset(d_hist, 0, NBINS * 4));
    auto to = chrono::steady_clock::now();
    unsigned long long locs = 0;
    if (!opt["only"].empty()) {
      uint32_t idx = (uint32_t)stoul(opt["only"]);
      k_anneal<<<1, WARPS * 32, SMEM>>>(d_scr, d_steps, d_c3, idx, 1, seed, -1 << 27, d_hits, d_nhit, cap, d_hist, 1);
      CK(cudaDeviceSynchronize());
      Hit h; CK(cudaMemcpy(&h, d_hits, sizeof h, cudaMemcpyDeviceToHost));
      string pt; for (int i = 0; i < n; ++i) { int cs, cl, cm, cr, wr; decode(idx, cs, cl, cm, cr, wr);
        const uint8_t *st = steps.data() + (cs * 26 + wr) * NMAX * 2; int row = row_of(cl, cm, cr, i, st);
        pt += char('A' + h.plug[scr[row + h.plug[ct[i]]]]); }
      printf("best %d %s\n", h.score, pt.c_str());
      return 0;
    }
    for (int cs : cases) {
      uint32_t base = (uint32_t)cs * A4;
      for (uint32_t off = 0; off < (uint32_t)A4; off += 1u << 18) {
        uint32_t cnt = min((uint32_t)A4 - off, 1u << 18);
        k_anneal<<<(cnt + WARPS - 1) / WARPS, WARPS * 32, SMEM>>>(d_scr, d_steps, d_c3, base + off, cnt, seed, thr, d_hits, d_nhit, cap, d_hist, 0);
        CK(cudaGetLastError());
      }
      for (int wr = 0; wr < 26; ++wr) if (active[cs * 26 + wr]) locs += A3;
    }
    CK(cudaDeviceSynchronize());
    unsigned nh; CK(cudaMemcpy(&nh, d_nhit, 4, cudaMemcpyDeviceToHost));
    vector<Hit> hits(min(nh, cap)); if (!hits.empty()) CK(cudaMemcpy(hits.data(), d_hits, sizeof(Hit) * hits.size(), cudaMemcpyDeviceToHost));
    vector<unsigned> hist(NBINS); CK(cudaMemcpy(hist.data(), d_hist, NBINS * 4, cudaMemcpyDeviceToHost));
    sort(hits.begin(), hits.end(), [](const Hit &a, const Hit &b) { return a.score > b.score; });
    FILE *f = fopen((string(name) + ".tsv").c_str(), "w");
    for (auto &h : hits) {
      int cs, cl, cm, cr, wr; decode(h.idx, cs, cl, cm, cr, wr);
      const uint8_t *st = steps.data() + (cs * 26 + wr) * NMAX * 2;
      string pt, pl;
      for (int i = 0; i < n; ++i) { int row = row_of(cl, cm, cr, i, st); pt += char('A' + h.plug[scr[row + h.plug[ct[i]]]]); }
      for (int x = 0; x < 26; ++x) if (h.plug[x] > x) { pl += char('A' + x); pl += char('A' + h.plug[x]); }
      fprintf(f, "%d\t%d%d%d\t%u\t%s\t%s\n", h.score, l, m, r, h.idx, pl.c_str(), pt.c_str());
    }
    fclose(f);
    f = fopen((string(name) + ".hist").c_str(), "w");
    for (int b = 0; b < NBINS; ++b) if (hist[b]) fprintf(f, "%d\t%u\n", b * 32, hist[b]);
    fclose(f);
    f = fopen((string(name) + ".done").c_str(), "w"); fprintf(f, "%u hits%s\n", nh, nh > cap ? " (TRUNCATED)" : ""); fclose(f);
    double so = chrono::duration<double>(chrono::steady_clock::now() - to).count();
    double sa = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    fprintf(log, "order %d%d%d: %llu locations, %u hits, best %d, %.1fs (total %.1fs)\n", l, m, r, locs, nh, hits.empty() ? 0 : hits[0].score, so, sa);
    fflush(log);
  }
  fprintf(log, "finished\n"); fclose(log);
  return 0;
}
