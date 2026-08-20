# The RF-Drums model

RF-Drums carries the Concert Grand's philosophy to percussion: every sample is
computed, none is recorded. This document is the model's ledger, under the
same rule as `docs/PIANO_MODEL.md` in the RackForge repository: each mechanism
names the physics it comes from and the source that measured it; each
simplification is stated rather than hidden. Unit tests verify the claims
marked *tested*.

The piano's hard-won lessons apply from day one and are treated as law here:

* **A comparison is only a comparison when both sides are the same
  experiment.** Calibration happens against measured targets extracted from
  recordings with the same code, windows and normalisation on both sides —
  never against a formula, never by ear alone.
* **One curve per phenomenon.** No stacked corrections fitted on top of each
  other's errors.
* **A placeholder is stated as a placeholder.** The current voicings
  (`DrumSpec` in `src/lib.rs`) are plausible physical ranges, not
  measurements, and say so in the code.

## What is modelled

### The circular membrane: Bessel modes (tested)

A drumhead's modes are `J_m(α_mn·r/a)·cos(mθ)` with frequencies proportional
to the Bessel zeros `α_mn` — inharmonic from first principles, and that
inharmonicity is the sound (Fletcher & Rossing, *The Physics of Musical
Instruments*, ch. 3 and 18). The model carries 45 mode families (m ≤ 8,
n ≤ 5) as tabulated zeros; `math::besselj` (Miller's downward recurrence —
the power series cancels catastrophically in f32 at α ≈ 26 and is unusable)
evaluates the shapes, and a test closes the loop: every tabulated zero must
evaluate to zero through the model's own Bessel function, so a transcription
error cannot survive silently.

The textbook ratios are tested: (0,1) 0.628, (2,1) 1.340, (0,2) 1.441,
(3,1) 1.665, (1,2) 1.831, (4,1) 1.980 against the (1,1) reference.

### Air loading, derived rather than drawn (tested)

A real head drags a layer of air roughly one wavelength thick, so the added
mass per area goes as `ρ_air/k_mn` and the low modes are pulled down hardest:

    f_mn ∝ α_mn / √(1 + β·α₁₁/α_mn)

with β one number per drum. Fitted on a *single* ratio — the kettledrum's
(2,1) landing on Rossing's 1.5 (*Science of Percussion Instruments*, ch. 2) —
β ≈ 3.8 then **predicts** (3,1) at 2.01 and (4,1) at 2.54 against the
published 2 and 2.5, and pushes (0,1) below 0.55× where the kettledrum's
thump lives. Two of the three ratios are predictions and they land; the test
`air_loaded_ratios_match_the_kettledrum` holds this. Toms carry β well under
one, which is why a tom keeps the ideal membrane's clangour where a timpano
sings.

### The strike point in two dimensions (tested)

Where the stick lands is the modal projection `J_m(α·r/a)·cos(mθ)` — the
membrane's answer to the piano's `sin(nπx₀)` comb, and the model's core claim
over a zoned sample library: position changes *which modes exist*,
continuously. At dead centre every m ≥ 1 mode has a node under the stick and
only the breathing `(0,n)` family speaks — the pitchless thud; toward the rim
the high-m ladder opens into the ringing tone. Tested both at the shape level
(`centre_strike_speaks_only_through_the_breathing_modes`) and end-to-end on
the rendered spectrum (`strike_position_changes_the_spectrum`).

The modal norm `∫∫ψ²dA` — `(1/2)J_{m+1}(α)²` radially, π or 2π angularly —
turns each shape into a modal mass; without it the high modes come out
overdriven for the same reason the piano needed its strike projections
normalised.

### Two heads, one cavity (tested)

Only the axisymmetric modes sweep net volume — `∫cos(mθ)dθ = 0` kills every
other family's coupling to the enclosed air, so batter and resonant head talk
through `(0,1)`, `(0,2)` and their few relatives and through nothing else
(tested: `only_axisymmetric_modes_move_air`). Each breathing mode splits into
the pair a two-headed drum actually has: heads moving together (cavity volume
unchanged, frequency untouched) and heads squeezing the air spring (raised by
`√(1+2K/ω²)`). Structurally this is the Concert Grand's unison coupling one
door over, with its passivity lesson inherited for free: the split lives in
the frequencies, not in a feedback path that could gain energy.

### Touch reaches timbre through the contact (tested)

A soft blow lets the stick dwell; a hard one shortens the contact and
brightens the tone — the same road the piano's felt drives. Rendered as a
second-order low-pass over modal amplitudes at the contact time's reciprocal,
velocity swinging the contact. `a_hard_blow_is_brighter_than_a_soft_one`
holds the end-to-end claim on the rendered audio.

**Stated simplification:** this is the piano's *pre-0.60* felt filter, not
its emergent strike. The full nonlinear stick–membrane integration (the
membrane's `simulate_strike`, with the stick's mass and the head's returning
wave ending the contact) is planned, and the piano's version is the template
— including its documented defect list, so the hammer-never-separates trap
is known before it is stepped in.

### The crack: the stick's own impact (tested)

The user's first listening verdict — "suena como samples de batería
electrónica tipo 808" — was this section's absence, plus the three after it.
A drum hit's first ten milliseconds are mostly broadband noise, the stick
striking the head as a plate before modal motion establishes, and no
arrangement of modal amplitudes can stand in for it (the piano's action-noise
lesson, which percussion pays double: there its absence measured 25–31 dB).
Rendered as an exponentially dying, one-pole low-passed noise burst per
strike: bandwidth tied to the same contact time that filters the ladder
(hard blow → shorter contact → brighter crack), level on the Crack fader.
`the_attack_cracks_and_the_sustain_rings` holds attack-vs-sustain noisiness
on the rendered audio.

### Tension-modulation glide (tested)

A struck membrane is stretched by its own displacement: every mode starts
sharp and settles as the amplitude dies — Kirchhoff–Carrier, far larger on a
drumhead than on the piano's strings because the head moves millimetres. The
glide follows velocity² (an amplitude effect, absent pianissimo), reaches
~+1 semitone fortissimo and relaxes over ~50 ms at control rate by rebuilding
each rotation from its stored rest frequency — exactly norm-preserving, so
the piano's glide bug (decay factors scaled by √(1+step²) until A0 diverged)
cannot occur by construction. The 808 *fakes* this curve with a pitch
envelope; a model with neither the glide nor the crack reads as the 808.

### Degenerate twin pairs (tested)

Every `cos(mθ)` mode has a `sin(mθ)` twin at the same ideal frequency; real
heads split each pair a few cents through non-uniform hoop tension, and the
slow beat between twins is what makes a drum partial breathe instead of
holding a synthesizer's dead-straight sine (Rossing on near-degenerate pairs
in real drums). Both twins speak on every off-centre strike, split ~0.4% with
per-(drum, mode) jitter — uniform splitting would beat every partial at a
rate proportional to frequency, the piano's documented "shimmer" defect.

### The shell (tested, stated simplification)

The drum's wooden body: four stiff resonances above the head's pitch
(ratios ~3.3–8.4×, jittered per drum), dying at spruce-order loss
(η ≈ 3%, T60 = ln10³/(π·f·η) — tens of milliseconds), knocked at the strike.
The honest model drives the shell continuously through the bearing edge;
strike-seeding is the stated simplification, the same one the piano's clack
accepts. Level on the Shell fader.

### The contact's brightness constant (empirical, stated)

The modal low-pass at a naive `1/(π·t_contact)` put the floor tom's cutoff at
127 Hz and buried the entire Bessel ladder — fundamentals alone left
standing, which IS an 808 tom. The piano's ledger states the identical
lesson: a strict reciprocal reading "comes out far darker than measured,
because the felt hardens during contact". A stick tip on Mylar hardens more.
`CONTACT_BRIGHTNESS = 8` is empirical and stated as such until targets exist.

### The pair and the room (tested)

The Concert Grand's ledger closed this argument for every modelled
instrument: "a bone-dry direct-injected tone is precisely what an electric
piano is" — and a bone-dry dual-mono kit is precisely what a drum machine
is. Both constructions are the piano's, resized:

* **The pair.** Each drum sits at its place across the kit (kick centre,
  snare left, floor tom well right — drummer's perspective) and reaches a
  spaced overhead pair through equal-power gains AND its own arrival time
  per side, Δt from the geometry, fixed at note-on. A coincident pan-pot
  leaves the same signal at two levels; the delay is what decorrelates the
  channels, and the test measures exactly that (cross-correlation < 0.985).
* **The room.** Six-line FDN, Householder feedback (orthogonal, so the loop
  is unconditionally stable with per-line gains below one), mutually
  non-divisible line ratios, one-pole damping per path so highs die first.
  Line lengths follow the mean free path 4V/S of the described volume and
  gains follow a Sabine-order RT60 — a tracking booth (~15 m³, 0.25 s) to a
  live room (~250 m³, 1.1 s) on one Size control. The reverberant field is
  fed the unpanned sum: a diffuse field has forgotten where the drum was.
  Compare-and-wrap counters, not modulo — the piano measured that division
  at 12% of its callback.

### Rendering (tested)

Modal synthesis, the Concert Grand's engine: each mode is a damped quadrature
oscillator — a 2×2 rotation pre-scaled by the per-sample decay — four
multiplies and two adds per sample, no envelopes, no transcendentals in the
audio loop. Spent modes retire at block boundaries (the piano's cull).
A voice is at most 50 oscillators; eight voices of kit are a fraction of what
a single pedalled piano chord costs, which is the headroom the cymbal phase
will spend. `a_roll_across_the_kit_survives` is the stress guard: 200 blocks
of dense rolling with no NaN, no clip, no trap.

## Placeholder voicings, stated as such

`DrumSpec` holds per-drum numbers — pitch, air load, cavity stiffness, T60,
loss slope, contact time — in plausible physical ranges. **None is a
measurement yet.** The membrane loss curve (`mode_t60`) is one curve with the
right shape (losses climb with frequency; the drum darkens as it rings) and
no fitted constants. The radiation weight `1/(1+m/4)` stands in for the
multipole roll-off of high-m modes. All three are the calibration phase's
targets, and the model's claim until then is the mechanisms, not the values.

## Not yet modelled, and named so the absence is a plan

* **The snare's wires** — the defining nonlinearity of the instrument: ~20
  wires leaving and re-striking the resonant head, a collision problem with
  no linear modal answer. GM 38/40 currently sounds the drum with the snares
  thrown off, and says so. Plan: a collective wire model (mass-spring against
  the head's velocity, gated per control step), Avanzini/Serafin's family.
* **The resonant head as a full membrane** — today the bottom head exists
  only as the breathing-mode split. The real one carries its own complete
  mode set and its own tuning, and the *detuning between heads* is THE
  character control of a tom: equal heads sing, a lower resonant head gives
  the falling "doooom", a higher one dries the note — it is what a drummer
  tunes with the key.
* **The kick's port (Helmholtz)** — the modern kick's "whoomp" is largely
  the port's Helmholtz resonance; the cavity currently has a stiffness
  constant, no resonator and no vent. One more oscillator, cheap.
* **The kick's beater** — a longer, softer contact plus its own thump, and
  the burial (beater held against the head, killing the sustain — standard
  technique with no equivalent in the model yet).
* **The emergent strike** — the contact filter is still a drawn law; the
  stick-membrane integration (stick mass, tip stiffness, contact ended by
  the head throwing the stick off) brings the multiple micro-bounces of a
  real stroke with it. The piano's `simulate_strike` is the template,
  defect list included.
* **Radiation** — head displacement is radiated directly with a toy
  multipole weight; the real efficiency (self-cancelling m ≥ 1 multipoles,
  the near-piston (0,1), front/back head cancellation) is what shaped the
  piano's whole decay in the end, and will shape this one.
* **Membrane bending stiffness** — Mylar's analogue of the piano's B,
  sharpening the high ladder.
* **Kit sympathy** — striking the tom buzzes the snare's wires; the signature
  of standing in front of a real kit, and this model's pedal-halo analogue.
* **Cymbals** — the two-scale plan settled in design: ~40–60 low plate modes
  individualised (the crash's gong, the ride's ping, the bell), a statistical
  high-frequency cloud (small undamped FDN, the piano's open-register
  construction), and the von Kármán energy cascade rendered as an
  amplitude-dependent feed from the low modes into the cloud, its trajectory
  *measured* from recordings per zone and velocity rather than drawn.
  Baked offline: per-zone excitation vectors and the pruned internal-resonance
  coupling list. (Touzé, Thomas & Chaigne on nonlinear plate vibration.)
* **The hoop, the lugs, the continuous shell drive** — the rimshot and the
  rim click, hardware rattle under a hard blow, and the shell driven through
  the bearing edge for the note's whole life rather than seeded at the
  strike; they wait for measured targets.

## Calibration plan

The piano's method, ported: `tools/extract-drum-targets.py` (to be written,
from `extract-piano-targets.py`) reads an openly licensed recorded kit into
per-band, per-window energy and noisiness targets; a fitter scores renders
against them with identical windows and normalisation on both sides. For
percussion the *noisiness* measure is first-class, not a correction — most of
a drum's attack is exactly what a band-energy target cannot see.
