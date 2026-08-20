//! RF-Drums: a physically modelled drum kit, carrying the Concert Grand's
//! philosophy to percussion. Every sample is computed, none is recorded, and
//! `docs/DRUM_MODEL.md` is the ledger: each mechanism names its physics,
//! each simplification is stated rather than hidden.
//!
//! This first milestone is the membrane engine and the toms — the instrument
//! chosen to be built first because it exercises every new mechanism (2-D
//! Bessel modes, air loading, the coupled head pair, strike position) and
//! rings long enough that decay errors are audible. The kick reuses the same
//! engine with a heavier, shorter voicing. The snare speaks but is HONESTLY
//! INCOMPLETE: its wires are not modelled yet, and what sounds is the drum
//! under the snares-off lever. Cymbals are the next phase (low modes +
//! statistical cloud, per the two-scale plan).

#![cfg_attr(all(target_arch = "wasm32", not(test)), no_std)]

mod math;
pub mod membrane;

use math::{expf, powf, sincosf};
use membrane::{
    BESSEL_ZEROS, MODE_COUNT, air_loaded_ratio, angular_order, couple_heads, modal_norm,
    strike_shape, volume_displacement,
};
use rackforge_plugin_sdk::{MidiEvent, ParameterEvent, Processor, export_processor};

/// Simultaneous drum voices. A kit is not a piano: eight covers a two-handed
/// roll across the kit with the kick underneath, and each voice carries at
/// most `VOICE_MODES` oscillators, so the whole bank is far below the fuel
/// ceiling the Concert Grand taught us to respect.
const MAX_VOICES: usize = 8;

/// Modes per voice, in fixed layout:
///
/// * `[0..45)` — the membrane families (lower head-pair partner for m = 0);
/// * `[45..50)` — the upper (air-stiffened) breathing partners;
/// * `[50..90)` — the degenerate twins of every m ≥ 1 family. A `cos(mθ)`
///   mode has a `sin(mθ)` twin at the same ideal frequency; real heads split
///   the pair a few cents through non-uniform hoop tension, and the beat
///   between them is what makes a drum partial breathe instead of holding a
///   synthesizer's dead-straight sine (Rossing, *Science of Percussion
///   Instruments*, on near-degenerate mode pairs in real drums);
/// * `[90..94)` — the shell's wood modes.
const PAIR_BASE: usize = MODE_COUNT + membrane::RADIAL_ORDERS;
const SHELL_BASE: usize = PAIR_BASE + (MODE_COUNT - membrane::RADIAL_ORDERS);
const SHELL_MODES: usize = 4;
const VOICE_MODES: usize = SHELL_BASE + SHELL_MODES;

/// How much brighter the contact is than a naive 1/(π·t) reading of its
/// duration. The piano paid for this lesson first and its ledger states it:
/// a strict reciprocal-of-the-pulse cutoff "comes out far darker than
/// measured spectra, because the felt hardens during contact". A stick tip
/// on Mylar hardens harder than felt on wire. Without this the floor tom's
/// cutoff sat at 127 Hz — the entire Bessel ladder buried, fundamentals
/// alone left standing, which the user's ear named exactly: "suena como
/// samples de batería electrónica tipo 808". An 808 tom IS a decaying
/// sine; that is what a modal bank with its ladder filtered off becomes.
/// Empirical, and stated as such until targets exist.
const CONTACT_BRIGHTNESS: f32 = 8.0;

/// Tension-modulation glide: a struck membrane is stretched by its own
/// displacement, so every mode starts sharp by the same fraction and
/// settles as the amplitude dies — Kirchhoff–Carrier, the piano's ff glide
/// one instrument over, but far larger here because a drumhead moves
/// millimetres where a string moves tenths. The fortissimo onset reaches
/// GLIDE_MAX (≈ +1 semitone) and relaxes with GLIDE_TAU. Placeholder
/// values in the measured order of magnitude for toms; the 808 fakes this
/// very curve with a pitch envelope, which is why a model without it reads
/// as the 808 and not as the drum.
const GLIDE_MAX: f32 = 0.06;
const GLIDE_TAU_S: f32 = 0.05;
/// The twins' tension split, as a fraction of each mode's frequency, and
/// the per-mode jitter around it — a uniform split would make every pair
/// beat at a rate proportional to frequency, the piano's "shimmer" defect.
const PAIR_SPLIT: f32 = 0.004;

/// A struck mode below this magnitude² is spent; retired at block edges,
/// exactly the Concert Grand's cull.
const DEAD_MAGNITUDE_SQUARED: f32 = 1e-9;
const CULL_INTERVAL: u32 = 256;

// Parameters. The set is deliberately small until calibration exists —
// a control that cannot be measured against anything is a lie with a knob.
const PARAM_TUNE: u32 = 0;
const PARAM_DAMP: u32 = 1;
const PARAM_POSITION: u32 = 2;
const PARAM_LEVEL: u32 = 3;
/// Read by the packaging step when it writes `metadata/parameters.json`.
pub const PARAM_COUNT: usize = 4;

// The voicing table, exposed for calibration by ear: every `DrumSpec` field
// of every drum is a parameter, addressed as
//
//     index = SPEC_PARAM_BASE + drum * SPEC_PARAM_STRIDE + field
//
// and carrying its PHYSICAL value (hertz, seconds, dimensionless) rather
// than a normalized one — the dev frontend's faders and the JSON the user
// copies out of it are voicings a person can read, and what lands back in
// `default_specs` needs no conversion. The production `parameters.json`
// will not list these; they are the calibration surface, not the panel.
pub const SPEC_PARAM_BASE: u32 = 16;
pub const SPEC_PARAM_STRIDE: u32 = 16;
pub const SPEC_FIELDS: u32 = 10;
const FIELD_PITCH_HZ: u32 = 0;
const FIELD_AIR_LOAD: u32 = 1;
const FIELD_CAVITY: u32 = 2;
const FIELD_T60_S: u32 = 3;
const FIELD_LOSS_SLOPE: u32 = 4;
const FIELD_CONTACT_S: u32 = 5;
const FIELD_RADIUS: u32 = 6;
const FIELD_GAIN: u32 = 7;
const FIELD_CRACK: u32 = 8;
const FIELD_SHELL: u32 = 9;

/// One damped quadrature oscillator — the Concert Grand's `Component`,
/// unchanged: rotation matrix pre-scaled by the per-sample decay, four
/// multiplies and two adds per sample, no envelope, no transcendentals in
/// the audio loop.
#[derive(Clone, Copy, Default)]
struct Mode {
    s: f32,
    c: f32,
    rc: f32,
    rs: f32,
}

impl Mode {
    /// Starts from the state a strike leaves: zero displacement, full
    /// velocity — a struck membrane, like a struck string, leaves the
    /// contact moving, not displaced.
    fn strike(amp: f32, frequency: f32, decay_per_sample: f32, sample_rate: f32) -> Self {
        if amp == 0.0 || frequency <= 0.0 || frequency >= 0.5 * sample_rate {
            return Self::default();
        }
        let omega = core::f32::consts::TAU * frequency / sample_rate;
        let (sin, cos) = sincosf(omega);
        Self { s: 0.0, c: amp, rc: decay_per_sample * cos, rs: decay_per_sample * sin }
    }

    #[inline(always)]
    fn tick(&mut self) -> f32 {
        let s = self.s * self.rc + self.c * self.rs;
        let c = self.c * self.rc - self.s * self.rs;
        self.s = s;
        self.c = c;
        s
    }

    fn magnitude_squared(&self) -> f32 {
        self.s * self.s + self.c * self.c
    }

    fn retire(&mut self) {
        *self = Self::default();
    }

    fn is_live(&self) -> bool {
        self.rc != 0.0 || self.rs != 0.0
    }
}

/// What kind of drum a voice is: a preset of the one membrane engine.
///
/// Every number here is a PLACEHOLDER VOICING, stated as such: plausible
/// physical ranges, not measurements. The calibration phase (targets
/// extracted from recorded kits, the piano's `extract-piano-targets.py`
/// method) replaces them; until then the model's claim is the *mechanisms*,
/// not these values.
#[derive(Clone, Copy)]
struct DrumSpec {
    /// The (1,1) mode's frequency — the drum's perceived pitch.
    pitch_hz: f32,
    /// Air-mass loading β of the (1,1) mode. Kettledrum ≈ 3.8 (derived in
    /// `membrane`), toms well under 1, a thick kick head lower still.
    air_load: f32,
    /// The cavity spring's share for the breathing-mode split, `2K/ω²`.
    cavity_stiffness: f32,
    /// T60 of the (1,1) mode, seconds. Higher modes die faster on the
    /// membrane loss curve below.
    t60_s: f32,
    /// How steeply loss climbs with frequency (exponent on f/pitch).
    loss_slope: f32,
    /// Contact time of the stick or beater at full velocity, seconds. Soft
    /// blows lengthen it — the same touch-to-timbre road the piano's felt
    /// drives, rendered here as a low-pass over the modal amplitudes whose
    /// cutoff is the reciprocal of the contact (stated simplification: the
    /// full nonlinear stick-membrane integration is a later phase; the
    /// piano's `simulate_strike` is the template when it comes).
    contact_s: f32,
    /// Default strike radius (fraction of head radius) when the position
    /// parameter sits at its centre detent.
    strike_radius: f32,
    /// Overall voicing gain.
    gain: f32,
    /// The stick's impact noise level. The crack is most of a drum hit's
    /// first ten milliseconds and no arrangement of modal amplitudes can
    /// stand in for it — the piano's action-noise lesson, which percussion
    /// pays double.
    crack: f32,
    /// The wood shell's level: a handful of stiff, fast-dying resonances
    /// the strike knocks alongside the head. This is the drum's body in the
    /// literal sense.
    shell: f32,
}

/// The drums the kit voices, in spec-table order.
pub const DRUM_COUNT: usize = 5;

/// General MIDI note → index into the voicing table.
fn drum_for_note(note: u8) -> Option<usize> {
    Some(match note {
        35 | 36 => 0, // kick
        38 | 40 => 1, // snare
        41 | 43 => 2, // floor tom
        45 | 47 => 3, // low tom
        48 | 50 => 4, // high tom
        _ => return None,
    })
}

/// The shipped voicings — the numbers being calibrated by ear through the
/// dev frontend's spec faders. When a better set comes back as JSON, it
/// lands HERE, and nowhere else.
fn default_specs() -> [DrumSpec; DRUM_COUNT] {
    [
        // Kick 20": both heads heavy, ported, damped — the tone is mostly
        // the breathing pair plus the beater's thump.
        DrumSpec {
            pitch_hz: 55.0,
            air_load: 1.2,
            cavity_stiffness: 0.9,
            t60_s: 0.35,
            loss_slope: 1.6,
            contact_s: 0.008,
            strike_radius: 0.10,
            gain: 1.6,
            crack: 1.2,
            shell: 0.7,
        },
        // Snare 14" — WIRES NOT YET MODELLED; this is the drum with the
        // snares thrown off, and the ledger says so.
        DrumSpec {
            pitch_hz: 180.0,
            air_load: 0.35,
            cavity_stiffness: 0.55,
            t60_s: 0.5,
            loss_slope: 1.2,
            contact_s: 0.0018,
            strike_radius: 0.45,
            gain: 1.0,
            crack: 1.6,
            shell: 0.8,
        },
        // Floor tom 16"
        DrumSpec {
            pitch_hz: 82.0,
            air_load: 0.6,
            cavity_stiffness: 0.5,
            t60_s: 1.3,
            loss_slope: 1.3,
            contact_s: 0.0025,
            strike_radius: 0.4,
            gain: 1.25,
            crack: 1.0,
            shell: 0.7,
        },
        // Low tom 13"
        DrumSpec {
            pitch_hz: 110.0,
            air_load: 0.5,
            cavity_stiffness: 0.5,
            t60_s: 1.0,
            loss_slope: 1.3,
            contact_s: 0.002,
            strike_radius: 0.4,
            gain: 1.15,
            crack: 1.0,
            shell: 0.7,
        },
        // High tom 12"
        DrumSpec {
            pitch_hz: 140.0,
            air_load: 0.45,
            cavity_stiffness: 0.5,
            t60_s: 0.8,
            loss_slope: 1.3,
            contact_s: 0.0018,
            strike_radius: 0.4,
            gain: 1.1,
            crack: 1.0,
            shell: 0.7,
        },
    ]
}

impl DrumSpec {
    fn field(&self, field: u32) -> Option<f64> {
        Some(match field {
            FIELD_PITCH_HZ => self.pitch_hz as f64,
            FIELD_AIR_LOAD => self.air_load as f64,
            FIELD_CAVITY => self.cavity_stiffness as f64,
            FIELD_T60_S => self.t60_s as f64,
            FIELD_LOSS_SLOPE => self.loss_slope as f64,
            FIELD_CONTACT_S => self.contact_s as f64,
            FIELD_RADIUS => self.strike_radius as f64,
            FIELD_GAIN => self.gain as f64,
            FIELD_CRACK => self.crack as f64,
            FIELD_SHELL => self.shell as f64,
            _ => return None,
        })
    }

    /// Physical values, clamped to the range where the model stays sane
    /// rather than to taste — taste is what the faders are for.
    fn set_field(&mut self, field: u32, value: f32) -> bool {
        match field {
            FIELD_PITCH_HZ => self.pitch_hz = value.clamp(20.0, 1000.0),
            FIELD_AIR_LOAD => self.air_load = value.clamp(0.0, 8.0),
            FIELD_CAVITY => self.cavity_stiffness = value.clamp(0.0, 4.0),
            FIELD_T60_S => self.t60_s = value.clamp(0.02, 20.0),
            FIELD_LOSS_SLOPE => self.loss_slope = value.clamp(1.0, 4.0),
            FIELD_CONTACT_S => self.contact_s = value.clamp(0.000_2, 0.05),
            FIELD_RADIUS => self.strike_radius = value.clamp(0.02, 0.98),
            FIELD_GAIN => self.gain = value.clamp(0.0, 8.0),
            FIELD_CRACK => self.crack = value.clamp(0.0, 8.0),
            FIELD_SHELL => self.shell = value.clamp(0.0, 8.0),
            _ => return false,
        }
        true
    }
}

#[derive(Clone, Copy)]
struct Voice {
    modes: [Mode; VOICE_MODES],
    /// Each mode's rest frequency (rad/sample) and per-sample decay, kept so
    /// the glide can rebuild `rc`/`rs` at control rate. Rebuilding from the
    /// stored decay is exactly norm-preserving — the piano's tension-glide
    /// bug (each nudge scaling the decay by √(1+step²) until A0 diverged)
    /// cannot happen by construction.
    omega: [f32; VOICE_MODES],
    decay: [f32; VOICE_MODES],
    /// Current tension sharpening, as a fraction of each mode's frequency;
    /// decays toward zero at control rate.
    glide: f32,
    /// The crack: an exponentially dying, one-pole-low-passed noise burst.
    crack_amp: f32,
    crack_decay: f32,
    crack_lp: f32,
    crack_state: f32,
    rng: u32,
    active: bool,
    note: u8,
    age: u32,
}

impl Default for Voice {
    fn default() -> Self {
        Self {
            modes: [Mode::default(); VOICE_MODES],
            omega: [0.0; VOICE_MODES],
            decay: [0.0; VOICE_MODES],
            glide: 0.0,
            crack_amp: 0.0,
            crack_decay: 0.0,
            crack_lp: 0.0,
            crack_state: 0.0,
            rng: 1,
            active: false,
            note: 0,
            age: 0,
        }
    }
}

impl Voice {
    /// Installs a struck mode and remembers its rest rotation for the glide.
    fn install(&mut self, index: usize, amp: f32, frequency: f32, decay: f32, rate: f32) {
        self.modes[index] = Mode::strike(amp, frequency, decay, rate);
        if self.modes[index].is_live() {
            self.omega[index] = core::f32::consts::TAU * frequency / rate;
            self.decay[index] = decay;
        }
    }

    /// One control step of tension relaxation: every membrane mode's
    /// rotation is rebuilt at its rest frequency times (1 + glide). Shell
    /// modes ([SHELL_BASE..]) are wood and do not retension.
    fn relax(&mut self, glide_keep: f32) {
        if self.glide < 1.0e-4 {
            self.glide = 0.0;
            return;
        }
        self.glide *= glide_keep;
        let sharpen = 1.0 + self.glide;
        for index in 0..SHELL_BASE {
            if !self.modes[index].is_live() {
                continue;
            }
            let (sin, cos) = sincosf(self.omega[index] * sharpen);
            self.modes[index].rc = self.decay[index] * cos;
            self.modes[index].rs = self.decay[index] * sin;
        }
    }

    /// The crack's next sample: white noise through a one-pole low-pass,
    /// dying exponentially. Three multiplies and an LCG once the burst is
    /// spent below audibility it costs a compare.
    #[inline(always)]
    fn crack_tick(&mut self) -> f32 {
        if self.crack_amp < 1.0e-6 {
            return 0.0;
        }
        self.rng = self.rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let white = (self.rng >> 8) as f32 * (1.0 / 8_388_608.0) - 1.0;
        self.crack_state += self.crack_lp * (white - self.crack_state);
        self.crack_amp *= self.crack_decay;
        self.crack_state * self.crack_amp
    }
}

pub struct RfDrums {
    voices: [Voice; MAX_VOICES],
    specs: [DrumSpec; DRUM_COUNT],
    sample_rate: f32,
    since_cull: u32,
    // Parameters, all 0..1 with a documented mapping.
    tune: f32,
    damp: f32,
    position: f32,
    level: f32,
}

impl Default for RfDrums {
    fn default() -> Self {
        Self {
            voices: [Voice::default(); MAX_VOICES],
            specs: default_specs(),
            sample_rate: 48_000.0,
            since_cull: 0,
            tune: 0.5,
            damp: 0.5,
            position: 0.5,
            level: 0.8,
        }
    }
}

impl RfDrums {
    /// Membrane loss curve: T60 of a mode at `ratio` times the pitch.
    ///
    /// Mylar's internal losses plus air radiation both climb with frequency;
    /// the drum's high modes die first, which is why a tom darkens as it
    /// rings — the same shape as the piano's decay-against-frequency curve,
    /// with the same discipline owed: ONE curve, refit against measured
    /// targets when they exist. Placeholder form, stated as such.
    fn mode_t60(spec: &DrumSpec, ratio: f32, damp: f32) -> f32 {
        // damp 0..1 swings the whole curve half a decade around the voicing.
        let user = powf(10.0, 0.5 - damp);
        spec.t60_s * user / (1.0 + powf(ratio - 1.0, spec.loss_slope.max(1.0)).max(0.0))
    }

    fn strike(&mut self, note: u8, velocity: u8) {
        let Some(drum) = drum_for_note(note) else {
            return;
        };
        let spec = self.specs[drum];
        // Steal the oldest voice; a drum voice is short-lived and a kit
        // player retriggers the same drum constantly, so same-note steals
        // its own oldest instance first.
        let slot = self
            .voices
            .iter()
            .position(|voice| !voice.active)
            .unwrap_or_else(|| {
                // All busy: steal the oldest instance of the same drum if one
                // is ringing, else the oldest voice overall.
                let oldest = |indices: &mut dyn Iterator<Item = usize>| {
                    indices.max_by_key(|&index| self.voices[index].age)
                };
                let mut same_note =
                    (0..MAX_VOICES).filter(|&index| self.voices[index].note == note);
                oldest(&mut same_note)
                    .or_else(|| oldest(&mut (0..MAX_VOICES)))
                    .unwrap_or(0)
            });
        let voice = &mut self.voices[slot];
        *voice = Voice::default();
        voice.active = true;
        voice.note = note;

        let velocity_01 = velocity as f32 / 127.0;
        // Tune swings the pitch ±5 semitones around the voicing.
        let pitch = spec.pitch_hz * powf(2.0, (self.tune - 0.5) * 10.0 / 12.0);
        // Strike position: the parameter sweeps centre → rim around the
        // voicing's default. This is the 5-zone story made continuous — the
        // zones are just names for stretches of this axis.
        let radius = (spec.strike_radius + (self.position - 0.5) * 0.9).clamp(0.02, 0.98);
        // Contact time lengthens for soft blows (a stick thrown gently sinks
        // into the head longer), shortening — brightening — with velocity.
        let contact = spec.contact_s * (1.6 - 0.8 * velocity_01);
        // Second-order low-pass over modal amplitudes tied to the contact's
        // reciprocal — same construction as the piano's felt filter, and
        // with the piano's empirical brightness factor (see
        // CONTACT_BRIGHTNESS: the naive reciprocal buried the whole ladder
        // and left an 808).
        let cutoff = CONTACT_BRIGHTNESS / (core::f32::consts::PI * contact);

        let f11 = pitch;
        let velocity_amp = 0.12 * spec.gain * (0.25 + 0.75 * velocity_01 * velocity_01);

        // Tension modulation: the blow stretches the head, the pitch starts
        // sharp and settles. Kirchhoff–Carrier scales with the square of the
        // displacement, so the glide follows velocity².
        voice.glide = GLIDE_MAX * velocity_01 * velocity_01;

        // The crack: the stick's own broadband impact, brighter and louder
        // for the hard blow, its bandwidth tied to the same contact time
        // that shapes the modal ladder.
        let crack_cutoff =
            (2.5 / (core::f32::consts::TAU * contact)).min(0.45 * self.sample_rate);
        voice.crack_amp = 0.5 * spec.crack * spec.gain * (0.1 + 0.9 * velocity_01 * velocity_01);
        voice.crack_decay = expf(-1.0 / (2.5 * contact * self.sample_rate));
        voice.crack_lp =
            1.0 - expf(-core::f32::consts::TAU * crack_cutoff / self.sample_rate);
        voice.rng = 0x9e37_79b9 ^ ((note as u32) << 8) ^ (velocity as u32);

        for (index, &alpha) in BESSEL_ZEROS.iter().enumerate() {
            let m = angular_order(index);
            let frequency = f11 * air_loaded_ratio(alpha, spec.air_load);
            let shape = strike_shape(m, alpha, radius, 0.0);
            let norm = modal_norm(m, alpha);
            if norm <= 0.0 {
                continue;
            }
            // Amplitude: projection over modal mass, through the contact's
            // low-pass. The radiated weight of high-m modes falls as their
            // cancelling lobes shorten; a first-order 1/(1+m/4) stands in
            // for the multipole radiation roll-off (stated placeholder —
            // proper piston-in-baffle weights come with calibration).
            let ratio = frequency / cutoff;
            let contact_lp = 1.0 / (1.0 + ratio * ratio);
            let radiation = 1.0 / (1.0 + m as f32 * 0.25);
            let amp = velocity_amp * shape / norm * contact_lp * radiation;
            if amp.abs() < 1e-6 {
                continue;
            }
            let ratio_11 = frequency / f11;
            let t60 = Self::mode_t60(&spec, ratio_11, self.damp);
            let decay = decay_per_sample(t60, self.sample_rate);
            if m == 0 {
                // The breathing pair: this head's (0,n) splits against the
                // resonant head through the cavity. Both partners speak;
                // the upper (air-stiffened) one carries the attack's punch
                // and dies a little faster, the lower carries the boom.
                let (lower, upper) = couple_heads(frequency, spec.cavity_stiffness);
                let swept = volume_displacement(m, alpha).abs();
                let n_index = index - (m as usize) * membrane::RADIAL_ORDERS;
                voice.install(index, amp * 0.6, lower, decay, self.sample_rate);
                let upper_t60 = t60 * 0.7;
                let upper_decay = decay_per_sample(upper_t60, self.sample_rate);
                voice.install(
                    MODE_COUNT + n_index,
                    amp * 0.55 * (0.5 + swept),
                    upper,
                    upper_decay,
                    self.sample_rate,
                );
            } else {
                // The degenerate pair: cos(mθ) and its sin(mθ) twin, split a
                // few cents by non-uniform hoop tension, sharing the strike's
                // energy. Their beat is what keeps the partial alive to the
                // ear. The split is jittered per (drum, mode) — uniform
                // splitting is the shimmer defect the piano documented.
                let split = frequency
                    * PAIR_SPLIT
                    * (0.4 + 1.2 * hash01((note as u32) << 8 | index as u32));
                voice.install(index, amp * 0.6, frequency, decay, self.sample_rate);
                voice.install(
                    PAIR_BASE + (index - membrane::RADIAL_ORDERS),
                    amp * 0.5,
                    frequency + split,
                    decay,
                    self.sample_rate,
                );
            }
        }

        // The shell: the drum's wooden body, knocked through the bearing
        // edge. A handful of stiff resonances well above the head's pitch,
        // dying at wood's loss factor in tens of milliseconds — the knock
        // that says the head is mounted on something. Frequencies follow the
        // pitch (a bigger drum has a bigger, deeper shell), jittered per
        // drum so no two shells are the same barrel. Driven-through-the-rim
        // continuously is the honest model; strike-seeded is the stated
        // simplification, the same one the piano's clack accepts.
        let shell_ratios = [3.3f32, 4.9, 6.4, 8.4];
        for (slot, &ratio) in shell_ratios.iter().enumerate() {
            let jitter = 0.9 + 0.2 * hash01((note as u32) << 16 | slot as u32);
            let frequency = f11 * ratio * jitter;
            let ratio_cut = frequency / cutoff;
            let contact_lp = 1.0 / (1.0 + ratio_cut * ratio_cut);
            let amp = velocity_amp * spec.shell * 0.35 * contact_lp
                / (1.0 + slot as f32 * 0.6);
            // Wood at ~3% loss: T60 = ln(1000)/(π·f·η).
            let t60 = 6.907_755 / (core::f32::consts::PI * frequency * 0.03);
            let decay = decay_per_sample(t60, self.sample_rate);
            voice.install(SHELL_BASE + slot, amp, frequency, decay, self.sample_rate);
        }
    }

    fn handle_midi(&mut self, event: &MidiEvent) {
        match event.data[0] & 0xf0 {
            0x90 if event.data[2] > 0 => self.strike(event.data[1], event.data[2]),
            // Note-off is meaningless on a drum; choking (hi-hat, cymbal
            // grab) arrives with the cymbal phase.
            _ => {}
        }
    }

    /// The control step: tension relaxation, then the cull.
    fn cull(&mut self) {
        let glide_keep = expf(-(CULL_INTERVAL as f32) / (GLIDE_TAU_S * self.sample_rate));
        for voice in self.voices.iter_mut() {
            if !voice.active {
                continue;
            }
            voice.relax(glide_keep);
            voice.age = voice.age.saturating_add(1);
            let mut live = false;
            for mode in voice.modes.iter_mut() {
                if !mode.is_live() {
                    continue;
                }
                if mode.magnitude_squared() < DEAD_MAGNITUDE_SQUARED {
                    mode.retire();
                } else {
                    live = true;
                }
            }
            if !live {
                voice.active = false;
            }
        }
    }

    fn set(&mut self, index: u32, value: f64) -> bool {
        if index >= SPEC_PARAM_BASE {
            let drum = ((index - SPEC_PARAM_BASE) / SPEC_PARAM_STRIDE) as usize;
            let field = (index - SPEC_PARAM_BASE) % SPEC_PARAM_STRIDE;
            let Some(spec) = self.specs.get_mut(drum) else {
                return false;
            };
            return spec.set_field(field, value as f32);
        }
        let value = value as f32;
        match index {
            PARAM_TUNE => self.tune = value.clamp(0.0, 1.0),
            PARAM_DAMP => self.damp = value.clamp(0.0, 1.0),
            PARAM_POSITION => self.position = value.clamp(0.0, 1.0),
            PARAM_LEVEL => self.level = value.clamp(0.0, 1.0),
            _ => return false,
        }
        true
    }

    fn get(&self, index: u32) -> Option<f64> {
        if index >= SPEC_PARAM_BASE {
            let drum = ((index - SPEC_PARAM_BASE) / SPEC_PARAM_STRIDE) as usize;
            let field = (index - SPEC_PARAM_BASE) % SPEC_PARAM_STRIDE;
            return self.specs.get(drum)?.field(field);
        }
        Some(match index {
            PARAM_TUNE => self.tune as f64,
            PARAM_DAMP => self.damp as f64,
            PARAM_POSITION => self.position as f64,
            PARAM_LEVEL => self.level as f64,
            _ => return None,
        })
    }
}

fn decay_per_sample(t60: f32, sample_rate: f32) -> f32 {
    // e^(−ln(1000)/(T60·rate)); ln(1000) = 6.9078.
    expf(-6.907_755 / (t60.max(0.01) * sample_rate))
}

/// Deterministic 0..1 hash (Wang-style avalanche) — the Concert Grand's
/// source of per-partial irregularity, and this model's too. Same drum, same
/// mode, same split: repeatability is part of sounding like one particular
/// kit.
fn hash01(mut seed: u32) -> f32 {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed = seed.wrapping_mul(9);
    seed ^= seed >> 4;
    seed = seed.wrapping_mul(0x27d4_eb2d);
    seed ^= seed >> 15;
    (seed >> 8) as f32 * (1.0 / 16_777_216.0)
}

impl Processor for RfDrums {
    fn prepare(
        &mut self,
        sample_rate: f64,
        _maximum_frames: u32,
        _input_channels: u32,
        output_channels: u32,
    ) -> bool {
        if output_channels < 1 {
            return false;
        }
        self.sample_rate = sample_rate as f32;
        self.reset();
        true
    }

    fn set_parameter(&mut self, index: u32, value: f64) -> bool {
        self.set(index, value)
    }

    fn get_parameter(&self, index: u32) -> Option<f64> {
        self.get(index)
    }

    fn reset(&mut self) {
        self.voices = [Voice::default(); MAX_VOICES];
        self.since_cull = 0;
    }

    fn process(
        &mut self,
        _input: &[f32],
        output: &mut [f32],
        midi: &[MidiEvent],
        parameters: &[ParameterEvent],
        frames: u32,
        _input_channels: u32,
        output_channels: u32,
    ) {
        let channels = output_channels as usize;
        let level = self.level * self.level;
        let mut midi_index = 0;
        let mut parameter_index = 0;
        for frame in 0..frames as usize {
            while let Some(event) = midi.get(midi_index) {
                if event.frame as usize != frame {
                    break;
                }
                self.handle_midi(event);
                midi_index += 1;
            }
            while let Some(event) = parameters.get(parameter_index) {
                if event.frame as usize != frame {
                    break;
                }
                let _ = self.set(event.index, event.value);
                parameter_index += 1;
            }
            let mut sum = 0.0f32;
            for voice in self.voices.iter_mut() {
                if !voice.active {
                    continue;
                }
                for mode in voice.modes.iter_mut() {
                    sum += mode.tick();
                }
                sum += voice.crack_tick();
            }
            let sample = soft_clip(sum * level);
            for channel in 0..channels {
                output[frame * channels + channel] = sample;
            }
            self.since_cull += 1;
            if self.since_cull >= CULL_INTERVAL {
                self.since_cull = 0;
                self.cull();
            }
        }
    }
}

/// The same output guard the host expects of every instrument: a tanh-like
/// rational soft clip, unity below the knee.
fn soft_clip(x: f32) -> f32 {
    let x = x.clamp(-3.0, 3.0);
    x * (27.0 + x * x) / (27.0 + 9.0 * x * x)
}

export_processor!(
    RfDrums,
    max_frames = 4096,
    max_input_channels = 0,
    max_output_channels = 2,
    max_midi_events = 256,
    max_parameter_events = 256,
    max_transfer_bytes = 4096
);

#[cfg(all(target_arch = "wasm32", not(test)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo<'_>) -> ! {
    core::arch::wasm32::unreachable()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(drums: &mut RfDrums, frames: usize, midi: &[MidiEvent]) -> Vec<f32> {
        let mut output = vec![0.0f32; frames * 2];
        let mut done = 0;
        let mut first = true;
        while done < frames {
            let block = (frames - done).min(512);
            let events: &[MidiEvent] = if first { midi } else { &[] };
            let start = done * 2;
            drums.process(&[], &mut output[start..start + block * 2], events, &[], block as u32, 0, 2);
            first = false;
            done += block;
        }
        output
    }

    fn strike(note: u8, velocity: u8) -> Vec<MidiEvent> {
        vec![MidiEvent { frame: 0, data: [0x99, note, velocity], length: 3 }]
    }

    /// Goertzel band energy, mono mix.
    fn band_energy(samples: &[f32], rate: f32, low: f32, high: f32) -> f32 {
        let mono: Vec<f32> = samples.chunks(2).map(|f| 0.5 * (f[0] + f[1])).collect();
        let n = mono.len();
        let mut total = 0.0f64;
        let mut f = low;
        while f < high {
            let omega = core::f32::consts::TAU * f / rate;
            let coeff = 2.0 * omega.cos();
            let (mut s1, mut s2) = (0.0f32, 0.0f32);
            for &x in &mono {
                let s0 = x + coeff * s1 - s2;
                s2 = s1;
                s1 = s0;
            }
            let power = (s1 * s1 + s2 * s2 - coeff * s1 * s2) as f64 / n as f64;
            total += power;
            f *= 1.06; // ~semitone steps
        }
        total as f32
    }

    #[test]
    fn a_tom_speaks_and_dies() {
        let mut drums = RfDrums::default();
        assert!(drums.prepare(48_000.0, 512, 0, 2));
        let out = render(&mut drums, 48_000 * 3, &strike(41, 110));
        let peak = out.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(peak > 0.01, "silent tom, peak {peak}");
        assert!(peak < 1.0, "clipping tom, peak {peak}");
        assert!(out.iter().all(|x| x.is_finite()), "non-finite output");
        // The last half second must be far below the first.
        let early = band_energy(&out[..48_000], 48_000.0, 40.0, 4000.0);
        let late = band_energy(&out[out.len() - 48_000..], 48_000.0, 40.0, 4000.0);
        assert!(
            late < early * 0.05,
            "tom does not decay: early {early}, late {late}"
        );
    }

    /// The tom's spectrum must peak near its air-loaded mode frequencies and
    /// hold energy in the inharmonic upper ladder — a membrane, not a sine.
    #[test]
    fn the_tom_is_a_membrane_not_an_oscillator() {
        let mut drums = RfDrums::default();
        assert!(drums.prepare(48_000.0, 512, 0, 2));
        let out = render(&mut drums, 48_000, &strike(41, 110));
        let fundamental = band_energy(&out, 48_000.0, 70.0, 100.0);
        let upper = band_energy(&out, 48_000.0, 120.0, 400.0);
        assert!(fundamental > 0.0);
        // The (2,1)/(0,2)/(3,1) region of an 82 Hz tom lives roughly
        // 110–250 Hz; a sine would have nothing there.
        assert!(
            upper > fundamental * 0.05,
            "no upper ladder: fundamental {fundamental}, upper {upper}"
        );
    }

    /// Centre versus edge: the strike position must change which modes exist.
    /// At the centre only the breathing family speaks; at the rim the m ≥ 1
    /// ladder takes over. This is the model's core claim over a 3-zone
    /// sample library, so it is tested, not assumed.
    #[test]
    fn strike_position_changes_the_spectrum() {
        let render_at = |position: f64| {
            let mut drums = RfDrums::default();
            assert!(drums.prepare(48_000.0, 512, 0, 2));
            assert!(drums.set_parameter(PARAM_POSITION, position));
            render(&mut drums, 48_000, &strike(41, 110))
        };
        // position 0 pulls the radius to the centre clamp, 1.0 to the rim.
        let centre = render_at(0.0);
        let rim = render_at(1.0);
        // (1,1) of the 82 Hz tom sits at the pitch; the breathing (0,1)
        // sits well below it (air-loaded, ~0.55×).
        let centre_ring = band_energy(&centre, 48_000.0, 74.0, 92.0);
        let centre_thud = band_energy(&centre, 48_000.0, 38.0, 58.0);
        let rim_ring = band_energy(&rim, 48_000.0, 74.0, 92.0);
        let rim_thud = band_energy(&rim, 48_000.0, 38.0, 58.0);
        let centre_balance = centre_ring / centre_thud;
        let rim_balance = rim_ring / rim_thud;
        assert!(
            rim_balance > centre_balance * 3.0,
            "position does nothing: centre {centre_balance}, rim {rim_balance}"
        );
    }

    /// Velocity must brighten, not just louden — the contact filter at work.
    #[test]
    fn a_hard_blow_is_brighter_than_a_soft_one() {
        let render_at = |velocity: u8| {
            let mut drums = RfDrums::default();
            assert!(drums.prepare(48_000.0, 512, 0, 2));
            render(&mut drums, 24_000, &strike(41, velocity))
        };
        let soft = render_at(30);
        let hard = render_at(120);
        let brightness = |out: &[f32]| {
            let low = band_energy(out, 48_000.0, 60.0, 200.0);
            let high = band_energy(out, 48_000.0, 300.0, 1500.0);
            high / low.max(1e-12)
        };
        let soft_ratio = brightness(&soft);
        let hard_ratio = brightness(&hard);
        assert!(
            hard_ratio > soft_ratio * 1.3,
            "velocity does not brighten: soft {soft_ratio}, hard {hard_ratio}"
        );
    }

    /// The kick and every tom must speak; unmapped notes must not.
    #[test]
    fn the_map_speaks_where_it_claims_to() {
        for note in [36u8, 38, 41, 45, 48] {
            let mut drums = RfDrums::default();
            assert!(drums.prepare(48_000.0, 512, 0, 2));
            let out = render(&mut drums, 4800, &strike(note, 100));
            let peak = out.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            assert!(peak > 0.005, "note {note} silent, peak {peak}");
        }
        let mut drums = RfDrums::default();
        assert!(drums.prepare(48_000.0, 512, 0, 2));
        let out = render(&mut drums, 4800, &strike(60, 100));
        let peak = out.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(peak == 0.0, "unmapped note speaks, peak {peak}");
    }

    /// A fast roll across the kit must neither trap nor clip nor leak NaN —
    /// the Concert Grand's stress lesson, applied from day one.
    #[test]
    fn a_roll_across_the_kit_survives() {
        let mut drums = RfDrums::default();
        assert!(drums.prepare(48_000.0, 512, 0, 2));
        let mut output = vec![0.0f32; 512 * 2];
        let notes = [36u8, 38, 41, 45, 48, 38, 36, 41];
        for round in 0..200u32 {
            let note = notes[(round as usize) % notes.len()];
            let velocity = 60 + ((round * 13) % 67) as u8;
            let events = [MidiEvent { frame: 0, data: [0x99, note, velocity], length: 3 }];
            drums.process(&[], &mut output, &events, &[], 512, 0, 2);
            assert!(
                output.iter().all(|x| x.is_finite() && x.abs() <= 1.0),
                "round {round} produced bad samples"
            );
        }
    }

    /// The attack must be NOISY — broadband crack over the modal ladder —
    /// and the noise must be gone by the sustain. A clean modal attack is
    /// the 808 the user's ear caught; the piano's noisiness metric made the
    /// same absence measurable there.
    #[test]
    fn the_attack_cracks_and_the_sustain_rings() {
        let mut drums = RfDrums::default();
        assert!(drums.prepare(48_000.0, 512, 0, 2));
        let out = render(&mut drums, 24_000, &strike(41, 115));
        let attack = &out[..2 * 1440]; // first 30 ms
        let sustain = &out[2 * 14_400..2 * 15_840]; // 300-330 ms
        let high = |window: &[f32]| band_energy(window, 48_000.0, 2500.0, 8000.0);
        let attack_high = high(attack);
        let sustain_high = high(sustain);
        assert!(
            attack_high > sustain_high * 30.0,
            "no crack: attack 2.5-8 kHz {attack_high}, sustain {sustain_high}"
        );
    }

    /// The struck head starts sharp and settles: tension-modulation glide.
    /// State-level check — the render-level pitch trace belongs to the
    /// calibration tooling.
    #[test]
    fn a_hard_strike_starts_sharp_and_relaxes() {
        let mut drums = RfDrums::default();
        assert!(drums.prepare(48_000.0, 512, 0, 2));
        let mut output = vec![0.0f32; 512 * 2];
        drums.process(&[], &mut output, &strike(41, 127), &[], 512, 0, 2);
        let just_struck = drums.voices[0].glide;
        assert!(
            just_struck > 0.04,
            "ff strike not sharp: glide {just_struck}"
        );
        for _ in 0..40 {
            // ~430 ms
            drums.process(&[], &mut output, &[], &[], 512, 0, 2);
        }
        let settled = drums.voices[0].glide;
        assert!(
            settled < just_struck * 0.05,
            "glide never settles: {just_struck} -> {settled}"
        );
        // And a soft blow barely bends: the glide is amplitude physics, not
        // an envelope bolted to the note.
        let mut soft = RfDrums::default();
        assert!(soft.prepare(48_000.0, 512, 0, 2));
        soft.process(&[], &mut output, &strike(41, 30), &[], 512, 0, 2);
        assert!(
            soft.voices[0].glide < just_struck * 0.15,
            "soft blow bends like a hard one: {}",
            soft.voices[0].glide
        );
    }

    /// Every m >= 1 partial is a detuned twin pair — the beat that keeps a
    /// drum partial breathing. An off-centre strike must light both banks.
    #[test]
    fn the_partials_come_in_beating_pairs() {
        let mut drums = RfDrums::default();
        assert!(drums.prepare(48_000.0, 512, 0, 2));
        let mut output = vec![0.0f32; 512 * 2];
        drums.process(&[], &mut output, &strike(41, 110), &[], 512, 0, 2);
        let voice = &drums.voices[0];
        let live_pairs = (PAIR_BASE..SHELL_BASE)
            .filter(|&index| voice.modes[index].is_live())
            .count();
        assert!(live_pairs > 10, "only {live_pairs} twin modes live");
        let live_shell = (SHELL_BASE..VOICE_MODES)
            .filter(|&index| voice.modes[index].is_live())
            .count();
        assert!(live_shell >= 2, "shell silent: {live_shell} modes");
    }

    /// The spec faders must reach the engine: retuning the floor tom's pitch
    /// through the parameter surface must move the rendered spectrum, and
    /// reading it back must return what was written.
    #[test]
    fn spec_parameters_reach_the_membrane() {
        let base = SPEC_PARAM_BASE + 2 * SPEC_PARAM_STRIDE; // floor tom
        let mut drums = RfDrums::default();
        assert!(drums.prepare(48_000.0, 512, 0, 2));
        assert!(drums.set_parameter(base + FIELD_PITCH_HZ, 130.0));
        assert_eq!(drums.get_parameter(base + FIELD_PITCH_HZ), Some(130.0));
        let retuned_render = render(&mut drums, 24_000, &strike(41, 110));
        let mut stock = RfDrums::default();
        assert!(stock.prepare(48_000.0, 512, 0, 2));
        let stock_render = render(&mut stock, 24_000, &strike(41, 110));
        // The (1,1) region of the retuned drum, in both renders: only the
        // drum whose pitch fader moved may hold it. (Comparing bands within
        // ONE render confounds — a 130 Hz tom's breathing (0,1) lands right
        // back in the 82 Hz drum's fundamental band.)
        // A narrow band on the retuned (1,1) itself: the stock drum's (2,1)
        // sits at ~115 Hz and its crack is broadband, both legitimately near
        // the old wide band now that the ladder actually radiates.
        let retuned = band_energy(&retuned_render, 48_000.0, 124.0, 137.0);
        let stock = band_energy(&stock_render, 48_000.0, 124.0, 137.0);
        assert!(
            retuned > stock * 3.0,
            "pitch fader inert: stock 120-142 Hz {stock}, retuned {retuned}"
        );
        // Out-of-table drums and unknown fields must refuse, not wrap.
        assert!(!drums.set_parameter(SPEC_PARAM_BASE + 5 * SPEC_PARAM_STRIDE, 100.0));
        assert!(!drums.set_parameter(base + SPEC_FIELDS, 1.0));
    }

    /// Not a test: renders each drum to a WAV for listening and calibration.
    /// RF_DRUMS_RENDER=/path/to/dir cargo test --release render_wavs -- --ignored
    #[test]
    #[ignore]
    fn render_wavs() {
        let Ok(out_dir) = std::env::var("RF_DRUMS_RENDER") else {
            eprintln!("set RF_DRUMS_RENDER to an output directory");
            return;
        };
        for (name, note, velocity, seconds) in [
            ("kick", 36u8, 115u8, 2.0f32),
            ("kick-soft", 36, 50, 2.0),
            ("snare-nowires", 38, 110, 2.0),
            ("floor-tom", 41, 110, 4.0),
            ("floor-tom-soft", 41, 45, 4.0),
            ("low-tom", 45, 110, 3.0),
            ("high-tom", 48, 110, 3.0),
        ] {
            let mut drums = RfDrums::default();
            assert!(drums.prepare(48_000.0, 512, 0, 2));
            let frames = (48_000.0 * seconds) as usize;
            let out = render(&mut drums, frames, &strike(note, velocity));
            // Mono mix, 16-bit PCM WAV.
            let mono: Vec<f32> = out.chunks(2).map(|f| 0.5 * (f[0] + f[1])).collect();
            let mut bytes: Vec<u8> = Vec::new();
            let data_len = (mono.len() * 2) as u32;
            bytes.extend(b"RIFF");
            bytes.extend((36 + data_len).to_le_bytes());
            bytes.extend(b"WAVEfmt ");
            bytes.extend(16u32.to_le_bytes());
            bytes.extend(1u16.to_le_bytes());
            bytes.extend(1u16.to_le_bytes());
            bytes.extend(48_000u32.to_le_bytes());
            bytes.extend((48_000u32 * 2).to_le_bytes());
            bytes.extend(2u16.to_le_bytes());
            bytes.extend(16u16.to_le_bytes());
            bytes.extend(b"data");
            bytes.extend(data_len.to_le_bytes());
            for sample in &mono {
                bytes.extend(((sample.clamp(-1.0, 1.0) * 32_767.0) as i16).to_le_bytes());
            }
            std::fs::write(format!("{out_dir}/{name}.wav"), bytes).unwrap();
        }
    }
}
