# Glossary

Terms used throughout the chat2hamnosys subsystem and the broader Kozha project. Each entry is one
sentence followed by a link to an authoritative source. Use this when reading the codebase, the
research log, or the docs.

---

**ASL** — American Sign Language; the dominant sign language of Deaf communities in the United
States and most of Canada, distinct from BSL despite both serving English-speaking hearing
populations. See [Lifeprint](https://www.lifeprint.com/asl101/topics/asl1.htm) for a community
introduction; [Stokoe (1960)](https://en.wikipedia.org/wiki/Sign_Language_Structure) for the
foundational linguistic description.

**ASL-LEX** — A peer-reviewed lexical database of ~2,700 ASL signs with phonological,
iconicity, and frequency annotations; one of the few large-scale resources for ASL phonology.
See [asl-lex.org](https://asl-lex.org/) and Caselli et al. (2017),
[Behavior Research Methods 49(2)](https://link.springer.com/article/10.3758/s13428-016-0742-0).

**BSL** — British Sign Language; the dominant sign language of the Deaf community in the United
Kingdom, distinct from ASL despite both serving English-speaking hearing populations. Officially
recognised in UK law (BSL Act 2022). See the [British Deaf Association](https://bda.org.uk/).

**Citation form** — The "dictionary form" of a sign, produced in isolation without
sentence-level prosody, classifier modifications, or role shift. This system authors *only*
citation forms. See [Johnston & Schembri (2007), *Australian Sign Language*](https://www.cambridge.org/core/books/australian-sign-language-auslan/D8E8AC53D9D45D1A36B3D87E3FD4B40A)
for the citation-form vs. connected-discourse distinction.

**Classifier (predicate)** — A sign-language construction in which a handshape *depicts* a
referent (e.g. a vehicle, a person, a flat surface) and the movement depicts that referent's
spatial behavior. Classifiers are grammar-bound to context and *not* citation-form lexicon entries;
this system explicitly does not generate them. See Liddell (2003),
*[Grammar, Gesture, and Meaning in American Sign Language](https://www.cambridge.org/core/books/grammar-gesture-and-meaning-in-american-sign-language/4DEC1C0C617E5A3F1F37A39DB0B8AB55)*.

**CODA** — Child Of Deaf Adults; a hearing person raised in a Deaf household whose first language
is often a sign language. CODAs occupy a particular place in Deaf community discourse and may
serve as cultural bridges; they are not interchangeable with native-Deaf signers in this
project's reviewer policy. See [coda-international.org](https://www.coda-international.org/).

**CWASA** — CWA Synthetic Animation, the 3D signing-avatar engine produced by the Virtual Humans
Group at the University of East Anglia; consumes SiGML and renders WebGL animation in-browser. CC
BY-ND licensed. See [vhg.cmp.uea.ac.uk/tech/jas/vhg2024/cwa/](https://vhg.cmp.uea.ac.uk/tech/jas/vhg2024/cwa/).

**Deaf** (capital-D) — Deaf with a capital D denotes cultural and linguistic identity (a member
of a Deaf community whose primary language is a sign language); deaf with a lowercase d denotes
the audiological condition of hearing loss alone. Use of the capital-D form throughout this
project is intentional. See Padden & Humphries (1988), *[Deaf in America: Voices from a
Culture](https://www.hup.harvard.edu/books/9780674194243)*.

**DGS** — Deutsche Gebärdensprache (German Sign Language); the dominant sign language of the Deaf
community in Germany, with a substantial corpus and lexicographic tradition through the
Universität Hamburg DGS-Korpus. See [dgs-korpus.de](https://www.sign-lang.uni-hamburg.de/dgs-korpus/).

**Fingerspelling** — Spelling out a word letter-by-letter using a sign-language manual alphabet;
typically used for proper nouns, acronyms, and out-of-vocabulary terms. The fallback strategy in
the core Kozha pipeline when no entry exists for a word; the OOV problem this subsystem addresses.

**Gloss** — A short uppercase English label used to refer to a sign in writing (e.g., HELLO,
ELECTRON, CAT). A gloss is not a translation; it is a reference token for a specific sign. In
this codebase, gloss is the lookup key (`SignEntry.gloss`).

**HamNoSys** — Hamburg Notation System; a phonetic transcription system for sign languages
developed at the Institute of German Sign Language, Universität Hamburg. Version 4 (2018) defines
~231 codepoints in Unicode Private Use Area (PUA) starting at U+E000. See the
[HamNoSys 2018 user guide](https://www.sign-lang.uni-hamburg.de/dgs-korpus/files/inhalt_pdf/HamNoSys_2018.pdf).

**JASigning** — A renderer for SiGML produced by the Virtual Humans Group at UEA; the engine
underlying CWASA. See [vhg.cmp.uea.ac.uk/tech/jas/](https://vhg.cmp.uea.ac.uk/tech/jas/).

**LSF** — Langue des Signes Française (French Sign Language); the dominant sign language in
France. See [academiedelangue.fr](https://academiedelangue.fr/).

**Mouth picture** — A non-manual feature in many sign languages where the mouth produces a
specific shape derived from (but not always synchronous with) the corresponding spoken-language
word; encoded in SiGML via `<hnm_mouthpicture picture="…"/>` using SAMPA-ish notation. See
Boyes Braem & Sutton-Spence (2001), *[The Hands are the Head of the Mouth](https://www.signum-verlag.com/)*.

**Native-Deaf signer** — A Deaf person whose acquisition of a sign language began in early
childhood (typically before age 5), commonly through Deaf parents or early immersion in a Deaf
community. The reviewer policy distinguishes native-Deaf approvals from non-native approvals; see
[20-ethics.md § 4](20-ethics.md#4-governance).

**Non-manual marker** — A sign-language linguistic element produced by something other than the
hands: facial expression, eye gaze, head movement, mouth pictures, body posture. These are
grammatically meaningful, not optional. This system encodes a limited set (mouth picture, eyebrow
flag, eye gaze) per sign; broader prosodic non-manuals are out of scope.

**Phonological parameter** — A primitive feature of a sign in the Stokoe / Liddell-Johnson
phonological tradition: handshape, palm orientation, finger orientation, location, contact,
movement, and (in modern accounts) non-manual features. The eight-slot decomposition in
`parser/models.py:PartialSignParameters` mirrors this set.

**PUA** — Private Use Area; a range of Unicode codepoints (U+E000–U+F8FF in the BMP) reserved
for application-specific characters with no standard meaning. HamNoSys uses U+E000 onwards for
its symbol inventory.

**Regional variant** — A sign that differs across geographic communities of the same sign
language (e.g., BSL signs differ between London and Manchester). The reviewer policy can
require regional-match approvals (`CHAT2HAMNOSYS_REVIEW_REQUIRE_REGION_MATCH=true`).

**Role shift** — A sign-language construction in which the signer adopts the perspective of a
referent (typically marked by body shift, eye gaze direction, and facial expression). Like
classifiers, role shift is grammar-bound to context and out of scope for this system.

**SAMPA** — Speech Assessment Methods Phonetic Alphabet; an ASCII-safe phonetic notation. The
SiGML mouth-picture attribute uses a SAMPA-derived notation. See
[Wells, *Computer-coding the IPA: a proposed extension of SAMPA*](http://www.phon.ucl.ac.uk/home/sampa/).

**SiGML** — Signing Gesture Markup Language; an XML wrapper around HamNoSys that adds non-manual
streams (mouth, eyebrows, etc.) and per-sign metadata. The format CWASA / JASigning consume. See
[vhg.cmp.uea.ac.uk/tech/jas/sigml/](https://vhg.cmp.uea.ac.uk/tech/jas/sigml/).

**Signing space** — The three-dimensional region in front of the signer's body where signs are
articulated; spatial relationships within this region are linguistically meaningful (especially
for pronouns, classifier predicates, and verb agreement). This system encodes a fixed-frame
location only; signing-space modelling is out of scope.

**Stokoe notation** — The first phonological notation system for sign languages, proposed by
William Stokoe in 1960. Used three primitive features (handshape, location, movement); HamNoSys
extends the inventory and adds explicit orientation, non-manual, and modifier features.

**WFD** — World Federation of the Deaf; the international NGO representing Deaf communities
globally. Position statements from WFD inform the ethics statement. See [wfdeaf.org](https://wfdeaf.org/).

---

## Sign-language abbreviations used in this codebase

| Code | Sign language | Country / community |
|---|---|---|
| `bsl` | British Sign Language | UK |
| `asl` | American Sign Language | USA, Anglophone Canada |
| `dgs` | Deutsche Gebärdensprache | Germany |
| (future) `lsf` | Langue des Signes Française | France |
| (future) `ngt` | Nederlandse Gebarentaal | Netherlands |

The current system supports `bsl`, `asl`, `dgs`. Adding a new sign language requires (a) a vocab
map for that language, (b) ≥3 native-Deaf reviewers registered for that language, (c) a fixture
set in the eval harness, and (d) board approval.

## Spoken-language ISO codes (for the parent Kozha project)

`en` (English), `de` (German), `fr` (French), `es` (Spanish), `pl` (Polish), `nl` (Dutch),
`el` (Greek). These are the spoken languages the core Kozha translator processes via spaCy; they
are not sign-language codes.
