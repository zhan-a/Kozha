# Language coverage — existing corpus

Snapshot: 2026-04-20. All counts measured directly against files in
`data/`. Two distinct naming conventions exist in-repo — the authoring
pipeline writes a third.

---

## 1. SiGML corpora (`<hns_sign>` format)

| Language | Filename | Entries (`<hns_sign>` tags) | Language code (implied) |
|---|---|---:|---|
| Vietnamese SL | `data/Vietnamese_SL.sigml` | 3564 | — |
| Polish SL (PJM) | `data/Polish_SL_PJM.sigml` | 1932 | `pjm` |
| German SL (DGS) | `data/German_SL_DGS.sigml` | 1914 | `dgs` |
| Greek SL (GSL) | `data/Greek_SL_GSL.sigml` | 889 | `gsl` |
| BSL v1 | `data/hamnosys_bsl_version1.sigml` | 881 | `bsl` |
| Indian SL | `data/Indian_SL.sigml` | 763 | — |
| Kurdish SL | `data/Kurdish_SL.sigml` | 558 | — |
| Algerian SL | `data/Algerian_SL.sigml` | 418 | — |
| French SL (LSF) | `data/French_SL_LSF.sigml` | 381 | `lsf` |
| Bengali | `data/Bangla_SL.sigml` | 81 | — |
| Dutch SL (NGT) | `data/Dutch_SL_NGT.sigml` | 39 | `ngt` |

**Corpus subtotal: 11,420 signs across 11 languages.**

## 2. SiGML corpora (`<sign_manual>` older format)

| Language | Filename | Entries (`<sign_manual>` tags) | Notes |
|---|---|---:|---|
| Filipino SL | `data/Filipino_SL.sigml` | 14 | Older tag format — 0 `<hns_sign>` tags. |

## 3. SiGML fingerspelling alphabets

Filename convention: `<code>_alphabet_sigml.sigml`. Each file contains
26 entries (A–Z).

| Language | Filename | Entries |
|---|---|---:|
| ASL | `data/asl_alphabet_sigml.sigml` | 26 |
| BSL | `data/bsl_alphabet_sigml.sigml` | 26 |
| DGS | `data/dgs_alphabet_sigml.sigml` | 26 |
| LSF | `data/lsf_alphabet_sigml.sigml` | 26 |
| NGT | `data/ngt_alphabet_sigml.sigml` | 26 |
| PJM | `data/pjm_alphabet_sigml.sigml` | 26 |

**Alphabet subtotal: 156 fingerspelling entries across 6 languages.**

## 4. HamNoSys CSVs

Filename convention: `hamnosys_<lang>.csv`.

| Language | Filename | Rows (minus header) |
|---|---|---:|
| PJM | `data/hamnosys_pjm.csv` | 1914 |
| DGS | `data/hamnosys_dgs.csv` | 1123 |
| BSL | `data/hamnosys_bsl.csv` | 1046 |
| GSL | `data/hamnosys_gsl.csv` | 860 |
| LSF | `data/hamnosys_lsf.csv` | 380 |
| NGT | `data/hamnosys_ngt.csv` | 55 |

**CSV subtotal: 5378 rows across 6 languages.**

---

## 5. Filename conventions in play

Three distinct patterns exist:

### (a) Upstream corpora — `<Language>_SL[_code].sigml`

Capitalised English language name, optional ISO-ish suffix.
Examples: `French_SL_LSF.sigml`, `Dutch_SL_NGT.sigml`,
`Vietnamese_SL.sigml` (no suffix).

### (b) Alphabets — `<code>_alphabet_sigml.sigml`

Lowercase code, redundant `_sigml` in the stem.
Examples: `asl_alphabet_sigml.sigml`, `bsl_alphabet_sigml.sigml`.

### (c) Legacy BSL corpus — `hamnosys_<lang>_version<N>.sigml`

One occurrence today: `hamnosys_bsl_version1.sigml`. This is the
only existing file that matches the `hamnosys_<lang>_*.sigml`
pattern referenced in the Prompt 1 brief.

### (d) Authoring export target — `hamnosys_<lang>_authored.sigml`

Defined in
`backend/chat2hamnosys/storage.py:SQLiteSignStore.export_to_kozha_library()`.
The chat2hamnosys pipeline writes to this path once the two-reviewer
approval gate is passed. **No such file exists at audit time** —
`data/hamnosys_bsl_authored.sigml`, `_asl_authored.sigml`, and
`_dgs_authored.sigml` are all absent.

---

## 6. Authorable languages vs promised languages

The authoring API literal `Literal["bsl", "asl", "dgs"]`
(`api/models.py:37`, `review/models.py:32`, `eval/models.py:68`)
restricts contributions to three sign languages.

| Language | Promised on `contribute.html:489`? | Authorable today? | Has corpus? |
|---|:-:|:-:|:-:|
| BSL | yes | yes | yes (881 entries + 26 alphabet) |
| ASL | yes | yes | alphabet only (26); no lexicon corpus |
| DGS | yes | yes | yes (1914 + 26 alphabet + 1123 CSV) |
| LSF | yes | no | yes (381 + 26 + 380) |
| LSE | yes | no | **no corpus at all** |
| PJM | yes | no | yes (1932 + 26 + 1914) |
| NGT | yes | no | yes (39 + 26 + 55) |
| GSL | yes | no | yes (889 + 860) |
| Vietnamese | no | no | yes (3564) |
| Indian | no | no | yes (763) |
| Kurdish | no | no | yes (558) |
| Algerian | no | no | yes (418) |
| Bengali | no | no | yes (81) |
| Filipino | no | no | yes (14, older format) |

Five languages are advertised but not writable (LSF, LSE, PJM, NGT,
GSL). Six languages have corpora but neither appear in the FAQ nor
in the API literal.

---

## 7. Total corpus size

- **16,972** signed entries across all `.sigml` files today
  (11,420 hns_sign + 14 sign_manual + 156 alphabet + 5382
  counted via CSVs for BSL/DGS/PJM/GSL/LSF/NGT, though some CSVs
  overlap with sigml corpora).
- **Zero** chat2hamnosys-authored entries so far.
