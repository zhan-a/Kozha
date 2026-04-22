# Polish 14 — Credits sanity check

Cross-check of every license citation on `/credits` against the
corresponding entry in `README.md`. The prompt-14 scope is the
**license** text; per-corpus sign counts are audited separately
below as a flag, not a gate item.

## License cross-check (README ↔ credits.html)

| Source | README | credits.html | Match |
| ------ | ------ | ------------ | ----- |
| CWASA | CC BY-ND | CC BY-ND — *CWASA Conditions of Use* | yes |
| HamNoSys | CC BY 4.0 | CC BY 4.0 | yes |
| bgHamNoSysUnicode font | (not in README summary) | IDGS distribution terms | README does not cover font separately |
| BSL (DictaSign) | CC BY-NC-SA 3.0 | CC BY-NC-SA 3.0 | yes |
| LSF (DictaSign via SignAvatars) | CC BY-NC-SA 3.0 inherited from DictaSign | CC BY-NC-SA 3.0 — inherited | yes |
| GSL (DictaSign via SignAvatars) | CC BY-NC-SA 3.0 | CC BY-NC-SA 3.0 | yes |
| DGS Lexicon (via SignAvatars) | per upstream DGS Lexicon terms | Per DGS Lexicon terms | yes |
| PJM (Warsaw via SignAvatars) | per upstream Warsaw PJM Dictionary terms | Per the Warsaw PJM Dictionary terms | yes |
| NGT (SignLanguageSynthesis) | No license declared upstream | License unclear — reaching out to confirm | yes |
| Algerian (algerianSignLanguage-avatar) | No license declared upstream | License unclear | yes |
| Bangla (bdsl-3d-animation) | No license declared upstream | License unclear | yes |
| Indian (Text-to-Sign-Language / text_to_isl) | No license declared upstream | License unclear | yes |
| Kurdish (KurdishSignLanguage) | No license declared upstream | License unclear | yes |
| Vietnamese (VSL) | No license declared upstream | License unclear | yes |
| Filipino (syntheticfsl / signtyper) | No license declared upstream | License unclear | yes |
| argostranslate | MIT (library) | MIT License (library) | yes |
| spaCy | MIT (library) | MIT License (library) | yes |

**All license citations match.** Zero discrepancies on the license axis.

## Sign-count drift (flagged for follow-up, not a gate item)

Comparing README's summary sign counts against the live snapshot at
`public/progress_snapshot.json` (generated from the currently-loaded
corpora after prompt-7 quarantine):

| Lang | README count | Snapshot total* | Source before quarantine | Notes |
| ---- | ------------ | --------------- | ------------------------ | ----- |
| BSL  | 881          | 907             | — | snapshot includes 26-letter fingerspell alphabet |
| LSF  | 381          | 250             | 224 active + 157 quarantined = 381 | README reports **pre-quarantine**; snapshot is post-quarantine + alphabet |
| GSL  | 889          | 534             | 534 + 355 = 889 | same pattern |
| DGS  | 1,914        | 1,940           | 1,914 + 0 | snapshot adds alphabet; README matches source |
| PJM  | 1,932        | 1,046           | 1,020 + 912 = 1,932 | README is pre-quarantine |
| NGT  | 39           | 65              | 39 + 0 | snapshot adds alphabet |
| Bangla | 81         | 60              | 60 + 21 = 81 | README pre-quarantine |
| Indian | 763        | 729             | 729 + 34 = 763 | README pre-quarantine |
| Kurdish | 558       | 558             | 558 + 0 | match |
| Algerian | 1        | 1               | 1 + 417 = 418 | README reports post-quarantine active (1) |
| Vietnamese | 3,564  | 3,564           | 3,564 + 0 | match |
| Filipino | 0         | 0               | 0 (loader does not parse) | README explicitly says "lexical signs do not load today" |

\* snapshot total = active_signs + alphabet_letters (typically +26)

### Recommended (follow-up, not a gate blocker)

The README currently mixes pre-quarantine (upstream source totals) and
post-quarantine (actually-loaded) counts without labelling which is
which. This is a documentation-clarity issue introduced by the
prompt-7 quarantine pass, not a licensing one. A one-line edit to the
README's §Sources paragraph to clarify the convention ("counts below
reflect upstream source totals; see /progress for what is actively
loaded after quarantine") would resolve it. Filed for a post-launch
doc pass — not a launch blocker.

## Source corpus license pages — out-of-scope verification

Verifying each citation against the *upstream* license page requires
a browser fetch to slownikpjm.uw.edu.pl, sign-lang.uni-hamburg.de,
github.com, etc. That was last done during the prompt-9 credits
expansion commit (813bfa7) and documented there. This gate does not
re-fetch those pages; the internal consistency between README and
credits.html is the scope.
