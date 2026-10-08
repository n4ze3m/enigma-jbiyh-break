<p align="center">
    <h1 align="center">enigma-jbiyh-break</h1>
</p>

enigma-jbiyh-break is an open-source GPU Enigma breaker. It recovered the key of the real 1941 message JBIYH (20 July 1941, message No. 242) from its 55 cipher letters alone, with no crib.

## Result

```
NVYMI VLOGG KTDKK OYXWR DLBHR RZYPI LVVXO GBEFX AXCWB NGILR WARXO
```

```
WEGEN FXRNVJ REQ STOERUNG TABESMELDUNG SOFORT DURQ FUNK HQ ER HER
```

- **Reflector**: B
- **Rotors**: V IV I
- **Rings**: EXT
- **Plugboard**: AU BI CY ET FL HK MP NR OX QV
- **Indicator**: BSB NTK → TZU

Read with the garbles corrected:

```
WEGEN FERNSPRECHSTOERUNG TAGESMELDUNG SOFORT DURCH FUNK HIERHER
```

*"Because of a telephone fault, send the daily report here by radio immediately."* Q for CH (DURQ = DURCH) is a standard abbreviation in wartime German signals.

### Garbles

Five letters come out wrong, each on its own, which is what a misread cipher letter does. For each one, this table gives the cipher letter that would have produced the expected plaintext. Three of them differ from the letter received by one dropped Morse element.

| Pos | Got → Expected | Received | Needed | Morse |
| --- | -------------- | -------- | ------ | ----- |
| 6   | X → E          | L        | T      | `.-..` vs `-` |
| 9   | V → S          | G        | Q      | `--.` vs `--.-` |
| 10  | J → P          | K        | Q      | `-.-` vs `--.-` |
| 24  | B → G          | R        | E      | `.-.` vs `.` |
| 49  | Q → I          | R        | L      | `.-.` vs `.-..` |

The plugboard is a local optimum: no change to a single plug fixes any of these letters.

You can verify the result yourself with nothing but Python:

```bash
python3 enigma.py   # Enigma self-test against known historical vectors
python3 verify.py   # decrypts JBIYH with the recovered key
```

## Features

- **Anneal**: A CUDA kernel that runs simulated annealing on the plugboard for every rotor order and start position, scored with 1941 German n-grams.

- **Word Check**: Keeps only the candidates that contain German words and match the indicator.

- **Review**: Ranks the remaining candidates for manual reading.

I ran this on a rented RTX 4090.

### Manual Installation

#### Prerequisites

- NVIDIA GPU with CUDA toolkit (`nvcc`)
- g++ and Python 3

1. Clone the repository

```bash
git clone https://github.com/n4ze3m/enigma-jbiyh-break.git
cd enigma-jbiyh-break
```

2. Build (by default it will build for RTX 4090)

```bash
make
```

_Note: For other GPUs, set the arch, e.g. `make ARCH=sm_86`._

3. Run the search

```bash
./run.sh jbiyh data/JBIYH.txt BSB NTK 19000 55
python3 review.py 'out/jbiyh/*.s2top'
```

You can pass annealer options to `run.sh` as `key=value` pairs, for example `orders=541`, `K=12500`, `T0=900`, `T1=1` or `seed=1`. All results go to `out/jbiyh/`, and the run writes `ALL_DONE` there when it finishes.

## Data

| File                   | Contents                                     |
| ---------------------- | -------------------------------------------- |
| `data/JBIYH.txt`       | The ciphertext                               |
| `data/*_1941_ln.txt`   | German n-gram log-frequency tables (1941)    |
| `data/vocab_all.txt`   | German word frequency list                   |
| `data/keys-strict.txt` | Operator "lazy" message keys (AAA, ABC, ...) |

## Acknowledgements

- The CUDA kernel (`anneal.cu`) was written with help from an AI coding assistant.

## License

[MIT](LICENSE)
