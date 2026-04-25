# HamNoSys + SiGML Machine-Readable Reference

> Combined reference derived from the `hamnosys` TeX package v1.0.3 (Schulder & Hanke, 2022, LPPL 1.3c) and from research on UEA SiGML schemas (Elliott et al. 2004, Kennaway 2015, hamnosysml11.dtd, sigml.dtd). Designed to be parsed by code or ingested by an LLM.

## How this document is structured

1. **HamNoSys symbol catalog** — every symbol with its name, LaTeX command, Unicode codepoint, category, and semantic role. YAML-like blocks for clean parsing.
2. **HamNoSys phonological structure** — how symbols compose into a sign.
3. **SiGML element catalog** — every SiGML element used in `<hns_sign>`, `<hamgestural_sign>`, and `<hamnosys_sign>` forms.
4. **SiGML attribute enumerations** — the value sets for `handshape`, `extfidir`, `palmor`, `location`, etc.
5. **HamNoSys ↔ SiGML mapping** — how raw HamNoSys Unicode strings translate into SiGML named-tag form.

---

## 1. HamNoSys symbol catalog

The HamNoSys 4.0 alphabet contains 210 symbols spanning U+E000–U+E0F1 in the Unicode Private Use Area, plus 8 symbols at regular Unicode code points (U+0020, U+0021, U+002C, U+002E, U+003F, U+007B, U+007C, U+007D).

Each entry below uses the structure:

```yaml
- name: <official symbol name>
  hex: <Unicode hex codepoint>
  latex_cmd: <LaTeX command>
  category: <handshape | handshape_modifier | extended_finger_direction | palm_orientation | location | location_modifier | movement | movement_modifier | other | version | regular_unicode | obsolete>
  role: <short semantic description>
```

### 1.1 Handshapes (12 symbols)

The base handshape slot in a sign. One required per hand.

```yaml
- {name: hamfist,                hex: E000, latex_cmd: \hamfist,                category: handshape, role: closed fist}
- {name: hamflathand,            hex: E001, latex_cmd: \hamflathand,            category: handshape, role: flat hand, all fingers extended and adjacent}
- {name: hamfinger2,             hex: E002, latex_cmd: \hamfingertwo,           category: handshape, role: index finger extended only}
- {name: hamfinger23,            hex: E003, latex_cmd: \hamfingertwothree,      category: handshape, role: index and middle fingers extended, adjacent}
- {name: hamfinger23spread,      hex: E004, latex_cmd: \hamfingertwothreespread,category: handshape, role: index and middle fingers extended, spread (V-shape)}
- {name: hamfinger2345,          hex: E005, latex_cmd: \hamfingertwothreefourfive, category: handshape, role: four fingers extended (no thumb)}
- {name: hampinch12,             hex: E006, latex_cmd: \hampinchonetwo,         category: handshape, role: pinch between thumb and index}
- {name: hampinchall,            hex: E007, latex_cmd: \hampinchall,            category: handshape, role: all fingers pinched together with thumb}
- {name: hampinch12open,         hex: E008, latex_cmd: \hampinchonetwoopen,     category: handshape, role: pinch shape but with thumb-index gap (open pinch)}
- {name: hamcee12,               hex: E009, latex_cmd: \hamceeonetwo,           category: handshape, role: C-shape between thumb and index}
- {name: hamceeall,              hex: E00A, latex_cmd: \hamceeall,              category: handshape, role: C-shape using all fingers}
- {name: hamceeopen,             hex: E00B, latex_cmd: \hamceeopen,             category: handshape, role: open C-shape, wider aperture}
```

### 1.2 Handshape modifiers (8 symbols)

Diacritics applied to the preceding handshape symbol.

```yaml
- {name: hamthumboutmod,         hex: E00C, latex_cmd: \hamthumboutmod,         category: handshape_modifier, role: thumb extended outward}
- {name: hamthumbacrossmod,      hex: E00D, latex_cmd: \hamthumbacrossmod,      category: handshape_modifier, role: thumb crosses over palm}
- {name: hamthumbopenmod,        hex: E00E, latex_cmd: \hamthumbopenmod,        category: handshape_modifier, role: thumb away from fingers (open)}
- {name: hamfingerstraightmod,   hex: E010, latex_cmd: \hamfingerstraightmod,   category: handshape_modifier, role: fingers fully straight}
- {name: hamfingerbendmod,       hex: E011, latex_cmd: \hamfingerbendmod,       category: handshape_modifier, role: fingers bent at base joint}
- {name: hamfingerhookmod,       hex: E012, latex_cmd: \hamfingerhookmod,       category: handshape_modifier, role: fingers curled into hook}
- {name: hamdoublebent,          hex: E013, latex_cmd: \hamdoublebent,          category: handshape_modifier, role: fingers bent at multiple joints}
- {name: hamdoublehooked,        hex: E014, latex_cmd: \hamdoublehooked,        category: handshape_modifier, role: doubly hooked finger configuration}
```

### 1.3 Extended finger directions (18 symbols)

Direction the extended fingers point. Compass-style codes: u=up, d=down, l=left, r=right, o=out (away from signer), i=in (toward signer). Combinations indicate diagonals.

```yaml
- {name: hamextfingeru,          hex: E020, latex_cmd: \hamextfingeru,          category: extended_finger_direction, role: fingers point up}
- {name: hamextfingerur,         hex: E021, latex_cmd: \hamextfingerur,         category: extended_finger_direction, role: fingers point up-right}
- {name: hamextfingerr,          hex: E022, latex_cmd: \hamextfingerr,          category: extended_finger_direction, role: fingers point right}
- {name: hamextfingerdr,         hex: E023, latex_cmd: \hamextfingerdr,         category: extended_finger_direction, role: fingers point down-right}
- {name: hamextfingerd,          hex: E024, latex_cmd: \hamextfingerd,          category: extended_finger_direction, role: fingers point down}
- {name: hamextfingerdl,         hex: E025, latex_cmd: \hamextfingerdl,         category: extended_finger_direction, role: fingers point down-left}
- {name: hamextfingerl,          hex: E026, latex_cmd: \hamextfingerl,          category: extended_finger_direction, role: fingers point left}
- {name: hamextfingerul,         hex: E027, latex_cmd: \hamextfingerul,         category: extended_finger_direction, role: fingers point up-left}
- {name: hamextfingerol,         hex: E028, latex_cmd: \hamextfingerol,         category: extended_finger_direction, role: fingers point out-left (away and to left)}
- {name: hamextfingero,          hex: E029, latex_cmd: \hamextfingero,          category: extended_finger_direction, role: fingers point out (away from signer)}
- {name: hamextfingeror,         hex: E02A, latex_cmd: \hamextfingeror,         category: extended_finger_direction, role: fingers point out-right}
- {name: hamextfingeril,         hex: E02B, latex_cmd: \hamextfingeril,         category: extended_finger_direction, role: fingers point in-left (toward signer and to left)}
- {name: hamextfingeri,          hex: E02C, latex_cmd: \hamextfingeri,          category: extended_finger_direction, role: fingers point in (toward signer)}
- {name: hamextfingerir,         hex: E02D, latex_cmd: \hamextfingerir,         category: extended_finger_direction, role: fingers point in-right}
- {name: hamextfingerui,         hex: E02E, latex_cmd: \hamextfingerui,         category: extended_finger_direction, role: fingers point up-in (up and toward signer)}
- {name: hamextfingerdi,         hex: E02F, latex_cmd: \hamextfingerdi,         category: extended_finger_direction, role: fingers point down-in}
- {name: hamextfingerdo,         hex: E030, latex_cmd: \hamextfingerdo,         category: extended_finger_direction, role: fingers point down-out}
- {name: hamextfingeruo,         hex: E031, latex_cmd: \hamextfingeruo,         category: extended_finger_direction, role: fingers point up-out}
```

### 1.4 Palm orientations (8 symbols)

Direction the palm faces. Same compass codes as fingers.

```yaml
- {name: hampalmu,               hex: E038, latex_cmd: \hampalmu,               category: palm_orientation, role: palm faces up}
- {name: hampalmur,              hex: E039, latex_cmd: \hampalmur,              category: palm_orientation, role: palm faces up-right}
- {name: hampalmr,               hex: E03A, latex_cmd: \hampalmr,               category: palm_orientation, role: palm faces right}
- {name: hampalmdr,              hex: E03B, latex_cmd: \hampalmdr,              category: palm_orientation, role: palm faces down-right}
- {name: hampalmd,               hex: E03C, latex_cmd: \hampalmd,               category: palm_orientation, role: palm faces down}
- {name: hampalmdl,              hex: E03D, latex_cmd: \hampalmdl,              category: palm_orientation, role: palm faces down-left}
- {name: hampalml,               hex: E03E, latex_cmd: \hampalml,               category: palm_orientation, role: palm faces left}
- {name: hampalmul,              hex: E03F, latex_cmd: \hampalmul,              category: palm_orientation, role: palm faces up-left}
```

### 1.5 Locations (44 symbols)

Body or hand locations where a sign is articulated. Divided into head/face, torso, arm, and hand-zone subcategories.

```yaml
# Head and face
- {name: hamhead,                hex: E040, latex_cmd: \hamhead,                category: location, role: head (general)}
- {name: hamheadtop,             hex: E041, latex_cmd: \hamheadtop,             category: location, role: top of head}
- {name: hamforehead,            hex: E042, latex_cmd: \hamforehead,            category: location, role: forehead}
- {name: hameyebrows,            hex: E043, latex_cmd: \hameyebrows,            category: location, role: eyebrows}
- {name: hameyes,                hex: E044, latex_cmd: \hameyes,                category: location, role: eyes}
- {name: hamnose,                hex: E045, latex_cmd: \hamnose,                category: location, role: nose}
- {name: hamnostrils,            hex: E046, latex_cmd: \hamnostrils,            category: location, role: nostrils}
- {name: hamear,                 hex: E047, latex_cmd: \hamear,                 category: location, role: ear}
- {name: hamearlobe,             hex: E048, latex_cmd: \hamearlobe,             category: location, role: earlobe}
- {name: hamcheek,               hex: E049, latex_cmd: \hamcheek,               category: location, role: cheek}
- {name: hamlips,                hex: E04A, latex_cmd: \hamlips,                category: location, role: lips}
- {name: hamtongue,              hex: E04B, latex_cmd: \hamtongue,              category: location, role: tongue}
- {name: hamteeth,               hex: E04C, latex_cmd: \hamteeth,               category: location, role: teeth}
- {name: hamchin,                hex: E04D, latex_cmd: \hamchin,                category: location, role: chin}
- {name: hamunderchin,           hex: E04E, latex_cmd: \hamunderchin,           category: location, role: under chin}

# Torso
- {name: hamneck,                hex: E04F, latex_cmd: \hamneck,                category: location, role: neck}
- {name: hamshouldertop,         hex: E050, latex_cmd: \hamshouldertop,         category: location, role: top of shoulder}
- {name: hamshoulders,           hex: E051, latex_cmd: \hamshoulders,           category: location, role: shoulders (front)}
- {name: hamchest,               hex: E052, latex_cmd: \hamchest,               category: location, role: chest}
- {name: hamstomach,             hex: E053, latex_cmd: \hamstomach,             category: location, role: stomach}
- {name: hambelowstomach,        hex: E054, latex_cmd: \hambelowstomach,        category: location, role: below stomach}
- {name: hamneutralspace,        hex: E05F, latex_cmd: \hamneutralspace,        category: location, role: neutral signing space (in front of body)}

# Arm
- {name: hamupperarm,            hex: E060, latex_cmd: \hamupperarm,            category: location, role: upper arm}
- {name: hamelbow,               hex: E061, latex_cmd: \hamelbow,               category: location, role: elbow (outside)}
- {name: hamelbowinside,         hex: E062, latex_cmd: \hamelbowinside,         category: location, role: inside of elbow}
- {name: hamlowerarm,            hex: E063, latex_cmd: \hamlowerarm,            category: location, role: lower arm}
- {name: hamwristback,           hex: E064, latex_cmd: \hamwristback,           category: location, role: back of wrist}
- {name: hamwristpulse,          hex: E065, latex_cmd: \hamwristpulse,          category: location, role: wrist pulse side (palm side)}

# Hand zones (used as targets)
- {name: hamthumbball,           hex: E066, latex_cmd: \hamthumbball,           category: location, role: thumb ball (thenar eminence)}
- {name: hampalm,                hex: E067, latex_cmd: \hampalm,                category: location, role: palm}
- {name: hamhandback,            hex: E068, latex_cmd: \hamhandback,            category: location, role: back of hand}
- {name: hamthumbside,           hex: E069, latex_cmd: \hamthumbside,           category: location, role: thumb side of hand (radial)}
- {name: hampinkyside,           hex: E06A, latex_cmd: \hampinkyside,           category: location, role: pinky side of hand (ulnar)}

# Specific fingers and finger zones
- {name: hamthumb,               hex: E070, latex_cmd: \hamthumb,               category: location, role: thumb}
- {name: hamindexfinger,         hex: E071, latex_cmd: \hamindexfinger,         category: location, role: index finger}
- {name: hammiddlefinger,        hex: E072, latex_cmd: \hammiddlefinger,        category: location, role: middle finger}
- {name: hamringfinger,          hex: E073, latex_cmd: \hamringfinger,          category: location, role: ring finger}
- {name: hampinky,               hex: E074, latex_cmd: \hampinky,               category: location, role: pinky}
- {name: hamfingertip,           hex: E075, latex_cmd: \hamfingertip,           category: location, role: fingertip}
- {name: hamfingernail,          hex: E076, latex_cmd: \hamfingernail,          category: location, role: fingernail}
- {name: hamfingerpad,           hex: E077, latex_cmd: \hamfingerpad,           category: location, role: finger pad (volar tip)}
- {name: hamfingermidjoint,      hex: E078, latex_cmd: \hamfingermidjoint,      category: location, role: finger middle joint (PIP)}
- {name: hamfingerbase,          hex: E079, latex_cmd: \hamfingerbase,          category: location, role: finger base joint (MCP)}
- {name: hamfingerside,          hex: E07A, latex_cmd: \hamfingerside,          category: location, role: side of finger}
```

### 1.6 Location modifiers (4 symbols)

```yaml
- {name: hamlrbeside,            hex: E058, latex_cmd: \hamlrbeside,            category: location_modifier, role: location is beside the body part (offset)}
- {name: hamlrat,                hex: E059, latex_cmd: \hamlrat,                category: location_modifier, role: location is at/touching the body part}
- {name: hamcoreftag,            hex: E05A, latex_cmd: \hamcoreftag,            category: location_modifier, role: coreference tag (establishes a referent location)}
- {name: hamcorefref,            hex: E05B, latex_cmd: \hamcorefref,            category: location_modifier, role: coreference reference (refers to a previously tagged location)}
```

### 1.7 Movements (76 symbols)

Largest category. Includes straight directional movements, circular movements (full and clock-position partial circles), arcs, wavy/zigzag paths, dynamic and contact actions.

```yaml
# Straight directional movement (compass + diagonals through 3D space)
- {name: hammoveu,               hex: E080, latex_cmd: \hammoveu,               category: movement, role: move up}
- {name: hammoveur,              hex: E081, latex_cmd: \hammoveur,              category: movement, role: move up-right}
- {name: hammover,               hex: E082, latex_cmd: \hammover,               category: movement, role: move right}
- {name: hammovedr,              hex: E083, latex_cmd: \hammovedr,              category: movement, role: move down-right}
- {name: hammoved,               hex: E084, latex_cmd: \hammoved,               category: movement, role: move down}
- {name: hammovedl,              hex: E085, latex_cmd: \hammovedl,              category: movement, role: move down-left}
- {name: hammovel,               hex: E086, latex_cmd: \hammovel,               category: movement, role: move left}
- {name: hammoveul,              hex: E087, latex_cmd: \hammoveul,              category: movement, role: move up-left}
- {name: hammoveol,              hex: E088, latex_cmd: \hammoveol,              category: movement, role: move out-left (away from signer to left)}
- {name: hammoveo,               hex: E089, latex_cmd: \hammoveo,               category: movement, role: move out (away from signer)}
- {name: hammoveor,              hex: E08A, latex_cmd: \hammoveor,              category: movement, role: move out-right}
- {name: hammoveil,              hex: E08B, latex_cmd: \hammoveil,              category: movement, role: move in-left (toward signer to left)}
- {name: hammovei,               hex: E08C, latex_cmd: \hammovei,               category: movement, role: move in (toward signer)}
- {name: hammoveir,              hex: E08D, latex_cmd: \hammoveir,              category: movement, role: move in-right}
- {name: hammoveui,              hex: E08E, latex_cmd: \hammoveui,              category: movement, role: move up-in}
- {name: hammovedi,              hex: E08F, latex_cmd: \hammovedi,              category: movement, role: move down-in}
- {name: hammovedo,              hex: E090, latex_cmd: \hammovedo,              category: movement, role: move down-out}
- {name: hammoveuo,              hex: E091, latex_cmd: \hammoveuo,              category: movement, role: move up-out}

# Full circles in cardinal planes
- {name: hamcircleo,             hex: E092, latex_cmd: \hamcircleo,             category: movement, role: full circle starting outward}
- {name: hamcirclei,             hex: E093, latex_cmd: \hamcirclei,             category: movement, role: full circle starting inward}
- {name: hamcircled,             hex: E094, latex_cmd: \hamcircled,             category: movement, role: full circle starting downward}
- {name: hamcircleu,             hex: E095, latex_cmd: \hamcircleu,             category: movement, role: full circle starting upward}
- {name: hamcirclel,             hex: E096, latex_cmd: \hamcirclel,             category: movement, role: full circle starting leftward}
- {name: hamcircler,             hex: E097, latex_cmd: \hamcircler,             category: movement, role: full circle starting rightward}

# Full circles in diagonal planes
- {name: hamcircleul,            hex: E098, latex_cmd: \hamcircleul,            category: movement, role: full circle in up-left plane}
- {name: hamcircledr,            hex: E099, latex_cmd: \hamcircledr,            category: movement, role: full circle in down-right plane}
- {name: hamcircleur,            hex: E09A, latex_cmd: \hamcircleur,            category: movement, role: full circle in up-right plane}
- {name: hamcircledl,            hex: E09B, latex_cmd: \hamcircledl,            category: movement, role: full circle in down-left plane}
- {name: hamcircleol,            hex: E09C, latex_cmd: \hamcircleol,            category: movement, role: full circle in out-left plane}
- {name: hamcircleir,            hex: E09D, latex_cmd: \hamcircleir,            category: movement, role: full circle in in-right plane}
- {name: hamcircleor,            hex: E09E, latex_cmd: \hamcircleor,            category: movement, role: full circle in out-right plane}
- {name: hamcircleil,            hex: E09F, latex_cmd: \hamcircleil,            category: movement, role: full circle in in-left plane}
- {name: hamcircleui,            hex: E0A0, latex_cmd: \hamcircleui,            category: movement, role: full circle in up-in plane}
- {name: hamcircledo,            hex: E0A1, latex_cmd: \hamcircledo,            category: movement, role: full circle in down-out plane}
- {name: hamcircleuo,            hex: E0A2, latex_cmd: \hamcircleuo,            category: movement, role: full circle in up-out plane}
- {name: hamcircledi,            hex: E0A3, latex_cmd: \hamcircledi,            category: movement, role: full circle in down-in plane}

# Local hand movements
- {name: hamfingerplay,          hex: E0A4, latex_cmd: \hamfingerplay,          category: movement, role: finger wiggle (sequential finger flexion)}
- {name: hamnodding,             hex: E0A5, latex_cmd: \hamnodding,             category: movement, role: hand nodding (wrist flexion)}
- {name: hamswinging,            hex: E0A6, latex_cmd: \hamswinging,            category: movement, role: hand swinging side-to-side at wrist}
- {name: hamtwisting,            hex: E0A7, latex_cmd: \hamtwisting,            category: movement, role: forearm rotation (twisting)}
- {name: hamstircw,              hex: E0A8, latex_cmd: \hamstircw,              category: movement, role: stir clockwise (small circle at wrist)}
- {name: hamstirccw,             hex: E0A9, latex_cmd: \hamstirccw,             category: movement, role: stir counter-clockwise}

# Special operators within movement
- {name: hamreplace,             hex: E0AA, latex_cmd: \hamreplace,             category: movement, role: replace marker (handshape changes during movement)}
- {name: hamnomotion,            hex: E0AF, latex_cmd: \hamnomotion,            category: movement, role: explicit no-movement marker}

# Clock-position partial arcs (1/8th-circle increments)
- {name: hamclocku,              hex: E0B0, latex_cmd: \hamclocku,              category: movement, role: arc from 12 o'clock position}
- {name: hamclockul,             hex: E0B1, latex_cmd: \hamclockul,             category: movement, role: arc from 10–11 o'clock position}
- {name: hamclockl,              hex: E0B2, latex_cmd: \hamclockl,              category: movement, role: arc from 9 o'clock position}
- {name: hamclockdl,             hex: E0B3, latex_cmd: \hamclockdl,             category: movement, role: arc from 7–8 o'clock position}
- {name: hamclockd,              hex: E0B4, latex_cmd: \hamclockd,              category: movement, role: arc from 6 o'clock position}
- {name: hamclockdr,             hex: E0B5, latex_cmd: \hamclockdr,             category: movement, role: arc from 4–5 o'clock position}
- {name: hamclockr,              hex: E0B6, latex_cmd: \hamclockr,              category: movement, role: arc from 3 o'clock position}
- {name: hamclockur,             hex: E0B7, latex_cmd: \hamclockur,             category: movement, role: arc from 1–2 o'clock position}
- {name: hamclockfull,           hex: E0B8, latex_cmd: \hamclockfull,           category: movement, role: full clock-style circle}

# Arc movements (curved paths in named directions)
- {name: hamarcl,                hex: E0B9, latex_cmd: \hamarcl,                category: movement, role: arc curving leftward}
- {name: hamarcu,                hex: E0BA, latex_cmd: \hamarcu,                category: movement, role: arc curving upward}
- {name: hamarcr,                hex: E0BB, latex_cmd: \hamarcr,                category: movement, role: arc curving rightward}
- {name: hamarcd,                hex: E0BC, latex_cmd: \hamarcd,                category: movement, role: arc curving downward}

# Path-shape modifiers
- {name: hamwavy,                hex: E0BD, latex_cmd: \hamwavy,                category: movement, role: wavy path}
- {name: hamzigzag,              hex: E0BE, latex_cmd: \hamzigzag,              category: movement, role: zigzag path}

# Ellipses (oriented circles)
- {name: hamellipseh,            hex: E0C0, latex_cmd: \hamellipseh,            category: movement, role: horizontal ellipse}
- {name: hamellipseur,           hex: E0C1, latex_cmd: \hamellipseur,           category: movement, role: ellipse with up-right major axis}
- {name: hamellipsev,            hex: E0C2, latex_cmd: \hamellipsev,            category: movement, role: vertical ellipse}
- {name: hamellipseul,           hex: E0C3, latex_cmd: \hamellipseul,           category: movement, role: ellipse with up-left major axis}

# Size dynamics
- {name: hamincreasing,          hex: E0C4, latex_cmd: \hamincreasing,          category: movement, role: movement size increases over time}
- {name: hamdecreasing,          hex: E0C5, latex_cmd: \hamdecreasing,          category: movement, role: movement size decreases over time}

# Speed and tension
- {name: hamfast,                hex: E0C8, latex_cmd: \hamfast,                category: movement, role: fast movement}
- {name: hamslow,                hex: E0C9, latex_cmd: \hamslow,                category: movement, role: slow movement}
- {name: hamtense,               hex: E0CA, latex_cmd: \hamtense,               category: movement, role: tense / forceful movement}
- {name: hamrest,                hex: E0CB, latex_cmd: \hamrest,                category: movement, role: relaxed movement}
- {name: hamhalt,                hex: E0CC, latex_cmd: \hamhalt,                category: movement, role: abrupt halt at end of movement}

# Contact and proximity
- {name: hamclose,               hex: E0D0, latex_cmd: \hamclose,               category: movement, role: hands move close}
- {name: hamtouch,               hex: E0D1, latex_cmd: \hamtouch,               category: movement, role: hands touch (light contact)}
- {name: haminterlock,           hex: E0D2, latex_cmd: \haminterlock,           category: movement, role: hands interlock}
- {name: hamcross,               hex: E0D3, latex_cmd: \hamcross,               category: movement, role: hands cross}
- {name: hamarmextended,         hex: E0D4, latex_cmd: \hamarmextended,         category: movement, role: arm fully extended}
- {name: hambehind,              hex: E0D5, latex_cmd: \hambehind,              category: movement, role: positioned behind reference}
- {name: hambrushing,            hex: E0D6, latex_cmd: \hambrushing,            category: movement, role: brushing contact during movement}
```

### 1.8 Movement modifiers (2 symbols)

```yaml
- {name: hamsmallmod,            hex: E0C6, latex_cmd: \hamsmallmod,            category: movement_modifier, role: movement is smaller than default}
- {name: hamlargemod,            hex: E0C7, latex_cmd: \hamlargemod,            category: movement_modifier, role: movement is larger than default}
```

### 1.9 Other / structural symbols (19 symbols)

These are the *grammar* of HamNoSys — repetition operators, sequence/parallel brackets, two-handed symmetry markers, and a few miscellaneous markers.

```yaml
# Repetition operators
- {name: hamrepeatfromstart,     hex: E0D8, latex_cmd: \hamrepeatfromstart,     category: other, role: repeat the entire sign from start}
- {name: hamrepeatfromstartseveral, hex: E0D9, latex_cmd: \hamrepeatfromstartseveral, category: other, role: repeat from start several times}
- {name: hamrepeatcontinue,      hex: E0DA, latex_cmd: \hamrepeatcontinue,      category: other, role: continue current motion repeatedly}
- {name: hamrepeatcontinueseveral, hex: E0DB, latex_cmd: \hamrepeatcontinueseveral, category: other, role: continue motion several times}
- {name: hamrepeatreverse,       hex: E0DC, latex_cmd: \hamrepeatreverse,       category: other, role: repeat in reverse direction}
- {name: hamalternatingmotion,   hex: E0DD, latex_cmd: \hamalternatingmotion,   category: other, role: hands alternate during repetition}

# Composition brackets
- {name: hamseqbegin,            hex: E0E0, latex_cmd: \hamseqbegin,            category: other, role: begin sequential composition group}
- {name: hamseqend,              hex: E0E1, latex_cmd: \hamseqend,              category: other, role: end sequential composition group}
- {name: hamparbegin,            hex: E0E2, latex_cmd: \hamparbegin,            category: other, role: begin parallel composition group (simultaneous actions)}
- {name: hamparend,              hex: E0E3, latex_cmd: \hamparend,              category: other, role: end parallel composition group}
- {name: hamfusionbegin,         hex: E0E4, latex_cmd: \hamfusionbegin,         category: other, role: begin fusion group (movements blend)}
- {name: hamfusionend,           hex: E0E5, latex_cmd: \hamfusionend,           category: other, role: end fusion group}
- {name: hambetween,             hex: E0E6, latex_cmd: \hambetween,             category: other, role: between operator (intermediate position)}
- {name: hamplus,                hex: E0E7, latex_cmd: \hamplus,                category: other, role: combine / additive operator}

# Two-handed symmetry (sentence-initial in two-handed signs)
- {name: hamsymmpar,             hex: E0E8, latex_cmd: \hamsymmpar,             category: other, role: parallel symmetry — both hands move in same direction}
- {name: hamsymmlr,              hex: E0E9, latex_cmd: \hamsymmlr,              category: other, role: mirror symmetry — hands mirror across midline}

# Hand designation
- {name: hamnondominant,         hex: E0EA, latex_cmd: \hamnondominant,         category: other, role: marks the non-dominant hand}
- {name: hamnonipsi,             hex: E0EB, latex_cmd: \hamnonipsi,             category: other, role: contralateral side marker}

# Miscellaneous
- {name: hametc,                 hex: E0EC, latex_cmd: \hametc,                 category: other, role: et cetera (continuation)}
- {name: hamorirelative,         hex: E0ED, latex_cmd: \hamorirelative,         category: other, role: orientation relative (not absolute)}
- {name: hammime,                hex: E0F0, latex_cmd: \hammime,                category: other, role: mime / iconic gesture marker}
```

### 1.10 Version symbol (1 symbol)

```yaml
- {name: hamversion40,           hex: E0F1, latex_cmd: \hamversionfourzero,     category: version, role: HamNoSys 4.0 version marker}
```

### 1.11 Regular Unicode characters (8 symbols)

These live at standard Unicode code points but render as HamNoSys glyphs when the HamNoSys font is active. Note: the `autofont` mechanism cannot auto-detect these because they coincide with normal text characters.

```yaml
- {name: hamspace,               hex: 0020, latex_cmd: \hamspace,               category: regular_unicode, role: HamNoSys space}
- {name: hamexclaim,             hex: 0021, latex_cmd: \hamexclaim,             category: regular_unicode, role: punctuation marker (exclamation)}
- {name: hamcomma,               hex: 002C, latex_cmd: \hamcomma,               category: regular_unicode, role: punctuation marker (comma)}
- {name: hamfullstop,            hex: 002E, latex_cmd: \hamfullstop,            category: regular_unicode, role: punctuation marker (full stop)}
- {name: hamquery,               hex: 003F, latex_cmd: \hamquery,               category: regular_unicode, role: punctuation marker (question)}
- {name: hamaltbegin,            hex: 007B, latex_cmd: \hamaltbegin,            category: regular_unicode, role: alternative-form bracket open}
- {name: hammetaalt,             hex: 007C, latex_cmd: \hammetaalt,             category: regular_unicode, role: meta-alternative separator}
- {name: hamaltend,              hex: 007D, latex_cmd: \hamaltend,              category: regular_unicode, role: alternative-form bracket close}
```

### 1.12 Obsolete spacing symbols (6 symbols)

Marked as obsolete in HamNoSys 4.0 but still present in the font. **Do not generate these** in new content; tolerate when parsing legacy data.

```yaml
- {name: hamwristtopulse,        hex: E07C, latex_cmd: \hamwristtopulse,        category: obsolete, role: deprecated wrist-to-pulse marker}
- {name: hamwristtoback,         hex: E07D, latex_cmd: \hamwristtoback,         category: obsolete, role: deprecated wrist-to-back marker}
- {name: hamwristtothumb,        hex: E07E, latex_cmd: \hamwristtothumb,        category: obsolete, role: deprecated wrist-to-thumb marker}
- {name: hamwristtopinky,        hex: E07F, latex_cmd: \hamwristtopinky,        category: obsolete, role: deprecated wrist-to-pinky marker}
- {name: hammovecross,           hex: E0AD, latex_cmd: \hammovecross,           category: obsolete, role: deprecated cross-movement marker}
- {name: hammoveX,               hex: E0AE, latex_cmd: \hammoveX,               category: obsolete, role: deprecated X-movement marker}
```

---

## 2. HamNoSys phonological structure

A well-formed HamNoSys sign has this canonical slot order (Schmaling & Hanke 2001, Hanke 2004):

```
[symmetry_operator] [handshape] [handshape_modifier]* [extended_finger_direction] [palm_orientation] [location] [location_modifier]* [movement_block]
```

Where `movement_block` may be a single movement symbol, a `hamseqbegin … hamseqend` sequence, a `hamparbegin … hamparend` parallel group, a `hamfusionbegin … hamfusionend` fusion, or any nesting of these. Movement modifiers (`hamsmallmod`, `hamlargemod`, `hamfast`, `hamslow`, etc.) attach to the preceding movement.

For **two-handed signs**, the sign starts with a symmetry operator (`hamsymmpar` or `hamsymmlr`), and only the dominant hand's parameters need be specified — the non-dominant hand is derived by symmetry. To break symmetry partially, use `hamnondominant` or `hamnonipsi`.

The `{ … | … }` brackets (`hamaltbegin`, `hammetaalt`, `hamaltend`) wrap *alternative variants* in a dictionary entry — read by humans, not animation engines.

---

## 3. SiGML element catalog

SiGML wraps HamNoSys (and a richer phonetic representation) in XML for the JASigning / CWASA avatar engines. Every SiGML document is pure element hierarchy — no text content. The root is always `<sigml>`. There are **three signing-unit forms** that may coexist within a single `<sigml>` document.

### 3.1 Document root

```yaml
- element: sigml
  role: document root; container for any number of signing units
  children: [hns_sign, hamgestural_sign, hamnosys_sign]+
  attrs: []
```

### 3.2 Form A — H-SiGML (tokenised HamNoSys)

The most common form. Each HamNoSys symbol becomes a named empty element. PUA Unicode codepoints are NOT preserved in SiGML — they are translated to tag names.

```yaml
- element: hns_sign
  role: tokenised HamNoSys sign wrapper
  children: [hamnosys_manual, hamnosys_nonmanual?]
  attrs:
    - {name: gloss,    type: CDATA, required: true,  desc: gloss label for the sign}

- element: hamnosys_manual
  role: container for manual (hands and arms) HamNoSys symbols
  children: [<one empty element per HamNoSys symbol>]+
  attrs: []

- element: hamnosys_nonmanual
  role: container for non-manual HamNoSys symbols (face, head, body)
  children: [<empty elements>]+
  attrs: []
```

Inside `<hamnosys_manual>` and `<hamnosys_nonmanual>` every HamNoSys symbol appears as an empty tag matching its name. Examples: `<hamfist/>`, `<hamflathand/>`, `<hamfinger2/>`, `<hamthumboutmod/>`, `<hampalmd/>`, `<hamforehead/>`, `<hamparbegin/>`, `<hammoveo/>`, `<hamparend/>`, `<hamsymmlr/>`. The mapping from PUA codepoint to tag name is the symbol's `name` field from §1 above.

### 3.3 Form B — HML (HamNoSys parse tree)

The intermediate parse representation produced by JASigning's ANTLR3.3 parser. Rarely hand-authored. Useful as a structured target for an LLM that wants to emit *parse trees* rather than glyph sequences.

```yaml
- element: hamnosysml
  role: HML wrapper
  children: [hamnosys_sign]+
  attrs: []

- element: hamnosys_sign
  role: sign as a typed parse tree
  children: [sign2 | sign1]
  attrs:
    - {name: gloss, type: CDATA, required: false}

- element: sign2
  role: two-handed sign with explicit symmetry
  children: [symmoperator, handshape1, handshape2?, location1, action]

- element: sign1
  role: one-handed sign
  children: [handshape1, location1, action]

- element: symmoperator
  role: symmetry operator (parallel, mirror, etc.)
  attrs:
    - {name: type, type: enum, values: [par, lr, ipsi, contra]}

- element: handshape1
  role: dominant-hand handshape configuration
  attrs:
    - {name: handshapeclass, type: ham_*, required: true,  desc: e.g. ham_finger2, ham_flathand}
    - {name: thumbpos,       type: ham_*, required: false, desc: e.g. ham_thumb_out}
    - {name: bend1,          type: ham_*, required: false, desc: bend of finger 1}
    - {name: bend2,          type: ham_*, required: false}
    - {name: bend3,          type: ham_*, required: false}
    - {name: bend4,          type: ham_*, required: false}
    - {name: bend5,          type: ham_*, required: false}
    - {name: extfidir,       type: ham_*, required: false, desc: extended-finger direction}
    - {name: palmor,         type: ham_*, required: false, desc: palm orientation}

- element: handshape2
  role: non-dominant-hand handshape (when asymmetric)
  attrs:  (same as handshape1)

- element: location1
  role: location of articulation
  attrs:
    - {name: location, type: ham_*, required: true,  desc: e.g. ham_forehead, ham_chest}
    - {name: side,     type: enum,  required: false, values: [dom, nondom, ipsi, contra]}

- element: action
  role: container for movement
  children: [simplemovement | complexmovement | actionseq | actionpar]

- element: simplemovement
  children: [straightmovement | curvedmovement | circularmovement | nomovement]

- element: straightmovement
  attrs:
    - {name: direction, type: ham_*, required: true,  desc: e.g. ham_move_o}
    - {name: arc,       type: ham_*, required: false, desc: e.g. ham_arc_u}
    - {name: size,      type: enum,  required: false, values: [small, large]}

- element: curvedmovement
  attrs:
    - {name: shape, type: enum, required: true, values: [wavy, zigzag, ellipse_h, ellipse_v]}

- element: circularmovement
  attrs:
    - {name: plane,     type: enum, required: true, values: [u, d, l, r, o, i, ul, ur, dl, dr, ol, or, il, ir, uo, ui, do, di]}
    - {name: clockwise, type: bool, required: false}
    - {name: extent,    type: enum, required: false, values: [eighth, quarter, half, three_quarter, full]}
```

### 3.4 Form C — G-SiGML (gestural / phonetic)

Flat, attribute-rich representation. Direct input to the CWASA avatar's animation engine. Essentially the SiGML equivalent of HamNoSys decoded into named features.

```yaml
- element: hamgestural_sign
  role: gestural-form sign
  children: [sign_manual, sign_nonmanual?]
  attrs:
    - {name: gloss, type: CDATA, required: false}

- element: sign_manual
  role: manual articulation (hands)
  children: [handconfig, handconstellation?, location_bodyarm?, par_motion | seq_motion | rpt_motion | tgt_motion | directedmotion | curvedmotion | circularmotion | changeposture | nomotion]
  attrs:
    - {name: both_hands, type: bool, required: false}
    - {name: lr_symm,    type: bool, required: false, desc: left-right symmetry}
    - {name: ud_symm,    type: bool, required: false, desc: up-down symmetry}

- element: handconfig
  role: handshape + finger directions + palm orientation
  attrs:
    - {name: handshape, type: enum, required: true,  desc: see §4.1}
    - {name: thumbpos,  type: enum, required: false, desc: see §4.2}
    - {name: bend1,     type: enum, required: false, desc: see §4.3}
    - {name: bend2,     type: enum, required: false}
    - {name: bend3,     type: enum, required: false}
    - {name: bend4,     type: enum, required: false}
    - {name: bend5,     type: enum, required: false}
    - {name: ceeopening,type: enum, required: false, values: [tight, loose, slack]}
    - {name: extfidir,  type: enum, required: false, desc: see §4.4}
    - {name: palmor,    type: enum, required: false, desc: see §4.5}

- element: split_handconfig
  role: different handconfig per hand
  children: [handconfig, handconfig]

- element: handconstellation
  role: relationship between the two hands
  attrs:
    - {name: contact, type: enum, required: false, values: [none, light, medium, firm, brushing]}

- element: location_bodyarm
  role: where the hand is located on the body
  attrs:
    - {name: location, type: enum, required: true,  desc: see §4.6}
    - {name: side,     type: enum, required: false, values: [dom, nondom, ipsi, contra, both]}
    - {name: contact,  type: enum, required: false, values: [touch, close, none]}
    - {name: digits,   type: CDATA, required: false, desc: which fingers, e.g. "1" or "1,2"}

- element: directedmotion
  role: straight-line movement
  attrs:
    - {name: direction, type: enum, required: true,  desc: see §4.7}
    - {name: size,      type: enum, required: false, values: [small, normal, large]}
    - {name: speed,     type: enum, required: false, values: [slow, normal, fast]}

- element: curvedmotion
  role: arced movement
  attrs:
    - {name: direction, type: enum, required: true}
    - {name: curve,     type: enum, required: true, values: [u, d, l, r]}
    - {name: size,      type: enum, required: false}

- element: circularmotion
  role: circular movement
  attrs:
    - {name: axis,      type: enum, required: true,  desc: rotation axis: x, y, z, ul, ur, dl, dr}
    - {name: clockwise, type: bool, required: false}
    - {name: extent,    type: enum, required: false, values: [eighth, quarter, half, three_quarter, full]}
    - {name: size,      type: enum, required: false}

- element: changeposture
  role: handshape changes during the sign
  children: [handconfig]

- element: par_motion
  role: parallel movements (simultaneous)
  children: [<motion>, <motion>+]

- element: seq_motion
  role: sequential movements
  children: [<motion>, <motion>+]

- element: rpt_motion
  role: repeated movement
  children: [<motion>]
  attrs:
    - {name: count,        type: enum, required: false, values: [twice, thrice, several, many]}
    - {name: alternating,  type: bool, required: false}
    - {name: from_start,   type: bool, required: false}
    - {name: reverse,      type: bool, required: false}

- element: tgt_motion
  role: targeted movement (toward a location)
  children: [<motion>, location_bodyarm]
  attrs:
    - {name: manner, type: enum, required: false, values: [targetted, brushed, pressed]}

- element: nomotion
  role: explicit no movement
  attrs: []

- element: split_motion
  role: different motions per hand
  children: [<motion>, <motion>]
```

### 3.5 Non-manual tiers

Inside `<sign_nonmanual>` (G-SiGML form), articulation is split into named "tiers" representing different parts of the body:

```yaml
- element: sign_nonmanual
  children: [shoulder_tier?, body_tier?, head_tier?, eyegaze_tier?, facialexpr_tier?, mouthing_tier?, extra_tier?]

- element: shoulder_tier
  role: shoulder movements (shrugs, raises)

- element: body_tier
  role: torso lean / rotation

- element: head_tier
  role: head movement (nod, shake, tilt)

- element: eyegaze_tier
  role: eye-gaze direction

- element: facialexpr_tier
  role: container for facial sub-tiers
  children: [eye_brows?, eye_lids?, nose?]

- element: eye_brows
  attrs:
    - {name: movement, type: enum, values: [raised, lowered, furrowed]}

- element: eye_lids
  attrs:
    - {name: state, type: enum, values: [wide_open, open, narrowed, closed_briefly, closed]}

- element: nose
  attrs:
    - {name: movement, type: enum, values: [wrinkled, twitched]}

- element: mouthing_tier
  role: mouth articulation
  children: [mouth_picture?, mouth_gesture?]

- element: mouth_picture
  role: mouthed word (visemes)
  attrs:
    - {name: picture, type: CDATA, required: true, desc: viseme sequence using SAMPA-like notation, e.g. "nEt"}

- element: mouth_gesture
  role: non-speech mouth gesture
  attrs:
    - {name: gesture, type: enum, values: [puff, suck, bite, lick, smile, frown]}

- element: extra_tier
  role: additional non-manual features
```

### 3.6 SiGML extensions (rarely-shipped)

The 2011 SLTAT paper "Extending the SiGML Notation" (Glauert & Elliott) proposed extensions that are partially in production:

```yaml
- element: pdts
  role: posture-defined transitions wrapper

- element: posture
  role: explicit static posture

- element: transition
  role: typed transition between postures
  attrs:
    - {name: type,  type: enum, values: [direct, arc, wave, zigzag]}
    - {name: angle, type: CDATA, desc: angle in degrees, e.g. "157.5deg"}

- element: trans-form
  role: transition shape parameters

- element: shift
  role: positional shift

- element: between
  role: weighted interpolation between postures
  attrs:
    - {name: weight, type: float, desc: 0.0 to 1.0}
```

---

## 4. SiGML attribute enumerations

These are the value sets for the attributes most commonly seen on `<handconfig>`, `<location_bodyarm>`, and motion elements. Drawn from `sigmlh4manual.dtd` (27 KB, the largest enumeration source) and the published Elliott 2004 paper. Where uncertainty exists, the field is marked `[partial]`.

### 4.1 `handshape` values

```
fist, flat, finger2, finger23, finger23spread, finger2345, pinch12, pinchall,
pinch12open, cee12, ceeall, ceeopen
```

(12 base handshapes, matching the HamNoSys handshape category one-to-one.)

### 4.2 `thumbpos` values

```
default, out, across, opposed
```

### 4.3 `bend1` … `bend5` values

```
straight, halfbent, bent, hooked, doublebent, doublehooked, round, halfround
```

### 4.4 `extfidir` values

```
u, ur, r, dr, d, dl, l, ul, ol, o, or, il, i, ir, ui, di, do, uo
```

(18 directions matching HamNoSys §1.3.)

### 4.5 `palmor` values

```
u, ur, r, dr, d, dl, l, ul
```

(8 directions matching HamNoSys §1.4.)

### 4.6 `location` values (body locations)

```
head, headtop, forehead, eyebrows, eyes, nose, nostrils, ear, earlobe, cheek,
lips, tongue, teeth, chin, underchin, neck, shouldertop, shoulders, chest,
stomach, belowstomach, neutral,
upperarm, elbow, elbowinside, lowerarm, wristback, wristpulse,
thumbball, palm, handback, thumbside, pinkyside,
thumb, indexfinger, middlefinger, ringfinger, pinky,
fingertip, fingernail, fingerpad, fingermidjoint, fingerbase, fingerside
```

(44 locations matching HamNoSys §1.5.)

### 4.7 `direction` values (motion)

```
u, ur, r, dr, d, dl, l, ul, ol, o, or, il, i, ir, ui, di, do, uo
```

### 4.8 `curve` / `arc` values

```
u, d, l, r
```

### 4.9 `axis` values (circular motion)

```
x, y, z, ul, ur, dl, dr, ol, or, il, ir, uo, ui, do, di
```

### 4.10 Boolean attributes

`both_hands`, `lr_symm`, `ud_symm`, `clockwise`, `alternating`, `from_start`, `reverse` — accept `"true"` / `"false"`.

### 4.11 Side / hand designation

```
dom, nondom, ipsi, contra, both
```

---

## 5. HamNoSys ↔ SiGML translation rules

### 5.1 Critical fact

**SiGML does not preserve raw HamNoSys Unicode codepoints inside its content.** Every PUA character is translated to its corresponding empty XML tag whose name matches the symbol's `name` field from §1.

Example HamNoSys string (Unicode): `\uE00A\uE00E\uE010\uE027\uE03D\uE042\uE059\uE0D0\uE0E2\uE082\uE0AA\uE007\uE010\uE0E3`

becomes in H-SiGML:

```xml
<hns_sign gloss="HAMBURG">
  <hamnosys_manual>
    <hamceeall/>
    <hamthumbopenmod/>
    <hamfingerstraightmod/>
    <hamextfingerul/>
    <hampalmdl/>
    <hamforehead/>
    <hamlrat/>
    <hamclose/>
    <hamparbegin/>
    <hammover/>
    <hamreplace/>
    <hampinchall/>
    <hamfingerstraightmod/>
    <hamparend/>
  </hamnosys_manual>
</hns_sign>
```

### 5.2 Translation algorithm

```
def hamnosys_to_sigml(hamnosys_string, gloss):
    PUA_TO_NAME = {0xE000: "hamfist", 0xE001: "hamflathand", ...}  # full table from §1
    REGULAR_TO_NAME = {0x0020: "hamspace", 0x0021: "hamexclaim", ...}  # §1.11

    tags = []
    for ch in hamnosys_string:
        cp = ord(ch)
        if cp in PUA_TO_NAME:
            tags.append(f"<{PUA_TO_NAME[cp]}/>")
        elif cp in REGULAR_TO_NAME:
            tags.append(f"<{REGULAR_TO_NAME[cp]}/>")
        else:
            raise ValueError(f"Unknown HamNoSys codepoint: U+{cp:04X}")

    body = "\n    ".join(tags)
    return (
        f'<hns_sign gloss="{gloss}">\n'
        f'  <hamnosys_manual>\n    {body}\n  </hamnosys_manual>\n'
        f'</hns_sign>'
    )
```

The reverse direction (SiGML → HamNoSys) is symmetric: parse the XML, walk children of `<hamnosys_manual>`, and map each tag name back to its codepoint via the inverse of `PUA_TO_NAME`.

### 5.3 SiGML document boilerplate

A complete minimal SiGML document:

```xml
<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE sigml SYSTEM "https://www.visicast.cmp.uea.ac.uk/sigml/sigml.dtd">
<sigml>
  <hns_sign gloss="HAMBURG">
    <hamnosys_manual>
      <hamceeall/><hamthumbopenmod/><hamfingerstraightmod/><hamextfingerul/>
      <hampalmdl/><hamforehead/><hamlrat/><hamclose/>
      <hamparbegin/><hammover/><hamreplace/><hampinchall/>
      <hamfingerstraightmod/><hamparend/>
    </hamnosys_manual>
  </hns_sign>
</sigml>
```

### 5.4 Validation

Validate any candidate SiGML against the canonical UEA DTDs:

```
xmllint --dtdvalid https://www.visicast.cmp.uea.ac.uk/sigml/sigml.dtd candidate.sigml
```

Or in Python:

```python
from lxml import etree
with open("hamnosysml11.dtd") as f:
    dtd = etree.DTD(f)
tree = etree.parse("candidate.sigml")
assert dtd.validate(tree), dtd.error_log
```

### 5.5 What the JASigning / CWASA player accepts

`<hns_sign>`, `<hamgestural_sign>`, and `<hamnosys_sign>` are interchangeable inputs to JASigning. Internally the player converts H-SiGML → HML → G-SiGML before animation. For an LLM generating SiGML, **target H-SiGML** — it's the most compact, the most documented, and the form every public SiGML corpus uses.

---

## License notes

- The HamNoSys symbol catalog is derived from the `hamnosys` LaTeX package by Marc Schulder and Thomas Hanke (2022), © Universität Hamburg, distributed under the **LaTeX Project Public License v1.3c**. The HamNoSysUnicode 4.0 TrueType font is additionally available under **Creative Commons Attribution 4.0 International**.
- SiGML element and attribute information is synthesized from publicly available UEA documentation (Elliott et al. 2004 LREC paper, Kennaway 2015 arXiv 1502.02961, Glauert & Elliott 2011 SLTAT, Neves et al. 2020 LREC, and the published DTD files at https://www.visicast.cmp.uea.ac.uk/sigml/). UEA's JASigning/CWASA distribution is currently licensed **CC BY-SA**.
