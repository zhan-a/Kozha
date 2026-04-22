# Alphabet coverage per language

Fingerspelling alphabets must cover the full writing system of the
target language or the translator cannot fall through to letter-by-
letter spelling for unknown words.

| language | expected letters | present | missing | notes |
|---|---|---|---|---|
| American Sign Language (asl) | 26 | 25 | W | `asl_alphabet_sigml.sigml` (1 letter(s) quarantined) |
| British Sign Language (bsl) | 26 | 26 | none | `bsl_alphabet_sigml.sigml` |
| German Sign Language (dgs) | 30 | 26 | Ä, Ö, Ü, ß | `dgs_alphabet_sigml.sigml` |
| French Sign Language (lsf) | 26 | 26 | none | `lsf_alphabet_sigml.sigml` |
| Polish Sign Language (pjm) | 35 | 26 | Ó, Ą, Ć, Ę, Ł, Ń, Ś, Ź, Ż | `pjm_alphabet_sigml.sigml` |
| Dutch Sign Language (NGT) (ngt) | 26 | 26 | none | `ngt_alphabet_sigml.sigml` |

## Languages without a fingerspelling alphabet

These languages have a sign database but no alphabet file.
Words that fall through the translator's vocabulary lookup
cannot be fingerspelled and will be dropped.

| language | note |
|---|---|
| algerian | no `algerian_alphabet_sigml.sigml` in `data/` |
| bangla | no `bangla_alphabet_sigml.sigml` in `data/` |
| fsl | no `fsl_alphabet_sigml.sigml` in `data/` |
| gsl | no `gsl_alphabet_sigml.sigml` in `data/` |
| isl | no `isl_alphabet_sigml.sigml` in `data/` |
| kurdish | no `kurdish_alphabet_sigml.sigml` in `data/` |
| vsl | no `vsl_alphabet_sigml.sigml` in `data/` |
