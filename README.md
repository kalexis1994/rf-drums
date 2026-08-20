# RF-Drums

A physically modelled drum kit for [RackForge](https://github.com/kalexis1994/rackforge),
built on the Concert Grand's philosophy: every sample is computed, none is
recorded. The whole kit compiles to a ~18 KB wasm component.

`docs/DRUM_MODEL.md` is the model's ledger — what is modelled with its
physics named, what is a stated placeholder, what is not yet modelled and why.

## Status

Milestone 1: the membrane engine and the toms.

- Circular-membrane modal bank (45 Bessel families), air-mass loading derived
  and tested against the published kettledrum ratios, two heads coupled
  through the cavity's breathing modes.
- Continuous strike position (centre thud → rim ring) — the model's core
  advantage over zoned samples.
- Voices: kick (35/36), snare **without wires yet** (38/40), floor tom
  (41/43), low tom (45/47), high tom (48/50) on the GM percussion map.
- Cymbals are the next phase (two-scale: low modes + statistical cloud).

## Build

Development expects a `rackforge` checkout adjacent to this repository.

```bash
cargo test                                           # engine + physics tests
cargo build --release --target wasm32-unknown-unknown  # the component
RF_DRUMS_RENDER=renders cargo test --release render_wavs -- --ignored  # listening WAVs
```
