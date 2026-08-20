//! RF-Drums: a physically modelled drum kit, carrying the Concert Grand's
//! philosophy to percussion. Every sample is computed, none is recorded, and
//! `docs/DRUM_MODEL.md` is the ledger: each mechanism names its physics,
//! each simplification is stated rather than hidden.
//!
//! This first milestone is the membrane engine and the toms — the instrument
//! chosen to be built first because it exercises every new mechanism (2-D
//! Bessel modes, air loading, the coupled head pair, strike position) and
//! rings long enough that decay errors are audible. The kick reuses the same
//! engine with a heavier, shorter voicing. The snare carries a stated
//! collective wire model; individual wire collisions remain future work.
//! Cymbals are the next phase (low modes + statistical cloud, per the
//! two-scale plan).

#![cfg_attr(all(target_arch = "wasm32", not(test)), no_std)]

mod math;
pub mod membrane;

use math::{expf, powf, roundf, sincosf, sqrtf};
use membrane::{
    BESSEL_ZEROS, MODE_COUNT, air_loaded_ratio, angular_order, couple_detuned_heads,
    modal_norm, strike_shape, volume_displacement,
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
/// The resonant head's own m >= 1 ladder (its breathing modes already live
/// inside the coupled eigenpairs): one partner per batter family, reached
/// through the shell and the cavity, unstruck, so it rings longer than the
/// head that was hit.
const RES_BASE: usize = SHELL_BASE + SHELL_MODES;
const RES_MODES: usize = MODE_COUNT - membrane::RADIAL_ORDERS;
/// The kick port's Helmholtz resonance: one oscillator.
const PORT_SLOT: usize = RES_BASE + RES_MODES;
const VOICE_MODES: usize = PORT_SLOT + 1;

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
/// millimetres where a string moves tenths. Onset amount and relaxation are
/// part of each `DrumSpec`: a loose floor tom bends farther and longer than
/// a tight snare. The 808 fakes this very curve with a pitch envelope, which
/// is why a model without it reads as the 808 and not as the drum.
///
/// The twins' tension split, as a fraction of each mode's frequency, and
/// the per-mode jitter around it — a uniform split would make every pair
/// beat at a rate proportional to frequency, the piano's "shimmer" defect.
const PAIR_SPLIT: f32 = 0.004;

/// The head's bending stiffness — the piano's inharmonicity B, in two
/// dimensions. Mylar is a membrane with a little plate in it: bending adds
/// a restoring force growing with the wavenumber squared, so each mode is
/// raised by √(1 + B·(α/α₁₁)²) over the ideal-membrane position — a few
/// cents in the middle of the table, ~+9% for the highest mode carried.
/// Without it the whole upper ladder sits systematically flat of a real
/// head. One number for all drums (film thickness varies less than head
/// diameter); placeholder magnitude in the plausible Mylar range, stated
/// as such until measured targets pin it.
const HEAD_BENDING_B: f32 = 0.004;

/// A mode's stiffness sharpening at `ratio` = α/α₁₁.
fn bending_sharpen(ratio: f32) -> f32 {
    sqrtf(1.0 + HEAD_BENDING_B * ratio * ratio)
}

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
const PARAM_ROOM_MIX: u32 = 4;
const PARAM_ROOM_SIZE: u32 = 5;
/// Read by the packaging step when it writes `metadata/parameters.json`.
pub const PARAM_COUNT: usize = 6;

// ---------------------------------------------------------------------------
// The room and the pair. The Concert Grand's ledger closed this argument:
// "a bone-dry direct-injected tone is precisely what an electric piano is" —
// and a bone-dry, dual-mono drum kit is precisely what a drum machine is.
// A kit is never heard direct: it is heard in a room, through two ears or
// two overheads. Both constructions below are the piano's, resized.
// ---------------------------------------------------------------------------

/// Six-line feedback delay network with a Householder feedback matrix —
/// mutually non-divisible line ratios for a dense, colourless tail, one-pole
/// damping in each feedback path so highs die faster than lows. The
/// Householder matrix is orthogonal, so with per-line gains below one the
/// loop is unconditionally stable.
const ROOM_LINES: usize = 6;
const ROOM_BUFFER: usize = 2048;
const ROOM_SPREAD: [f32; ROOM_LINES] = [0.62, 0.76, 0.90, 1.09, 1.23, 1.43];
/// Alternating injection signs decorrelate the lines from the start.
const ROOM_INJECT: [f32; ROOM_LINES] = [0.35, -0.35, 0.35, -0.35, 0.35, -0.35];
const SOUND_SPEED: f32 = 343.0;

/// Where each drum sits across the kit, metres from centre, drummer's
/// perspective (snare left of centre, floor tom well right), and the
/// overhead pair that hears them: spaced HALF-metre-class, above and ahead.
/// Each voice reaches each side with its own arrival time — the piano's
/// per-string arrival taps, one instrument over: Δt = geometry, fixed at
/// note-on, nothing interpolates. A coincident pair would hear no time
/// differences, and that is what "sounds like a drum machine pan-pot"
/// means.
const DRUM_X_M: [f32; DRUM_COUNT] = [0.0, -0.30, 0.55, -0.10, 0.20];
const PAIR_SPACING_M: f32 = 0.60;
const PAIR_DISTANCE_M: f32 = 1.40;
/// The direct bus: a short stereo ring the voices write into ahead of the
/// read point, one integer delay per (voice, side). 64 samples covers the
/// widest geometry at 48 kHz with room to spare.
const DIRECT_BUFFER: usize = 64;

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
pub const SPEC_FIELDS: u32 = 15;
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
const FIELD_RES_TUNE: u32 = 10;
const FIELD_PORT: u32 = 11;
const FIELD_WIRES: u32 = 12;
const FIELD_GLIDE: u32 = 13;
const FIELD_GLIDE_TAU: u32 = 14;

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
        if frequency <= 0.0 || frequency >= 0.5 * sample_rate {
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
    /// The resonant head's tuning as a ratio of the batter's — THE tom
    /// tuning gesture. Equal heads sing; a lower resonant head gives the
    /// falling "doooom"; the snare-side head sits far above (~1.35).
    res_tune: f32,
    /// The port's level: the vent hole's Helmholtz resonance, the modern
    /// kick's "whoomp". Zero on unported drums. A port also relieves the
    /// cavity spring (an open cavity squeezes less).
    port: f32,
    /// The snare wires' level. Zero everywhere but the snare; the model is
    /// collective (see `Voice::wire_tick`), not per-wire.
    wires: f32,
    /// Fractional onset sharpening at a full-velocity strike. Tension
    /// modulation is strongly body-dependent: a loose floor-tom head bends
    /// farther, and for longer, than a tight snare head.
    glide: f32,
    /// Time constant of the pitch relaxation, seconds.
    glide_tau_s: f32,
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
            res_tune: 0.90,
            port: 0.8,
            wires: 0.0,
            glide: 0.10,
            glide_tau_s: 0.09,
        },
        // Snare 14" — collective wire collision model; the ledger names
        // what remains before this becomes a per-wire simulation.
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
            res_tune: 1.35,
            port: 0.0,
            wires: 1.2,
            glide: 0.025,
            glide_tau_s: 0.035,
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
            res_tune: 0.94,
            port: 0.0,
            wires: 0.0,
            glide: 0.085,
            glide_tau_s: 0.11,
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
            res_tune: 0.95,
            port: 0.0,
            wires: 0.0,
            glide: 0.07,
            glide_tau_s: 0.09,
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
            res_tune: 0.95,
            port: 0.0,
            wires: 0.0,
            glide: 0.055,
            glide_tau_s: 0.07,
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
            FIELD_RES_TUNE => self.res_tune as f64,
            FIELD_PORT => self.port as f64,
            FIELD_WIRES => self.wires as f64,
            FIELD_GLIDE => self.glide as f64,
            FIELD_GLIDE_TAU => self.glide_tau_s as f64,
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
            FIELD_RES_TUNE => self.res_tune = value.clamp(0.6, 1.6),
            FIELD_PORT => self.port = value.clamp(0.0, 4.0),
            FIELD_WIRES => self.wires = value.clamp(0.0, 8.0),
            FIELD_GLIDE => self.glide = value.clamp(0.0, 0.2),
            FIELD_GLIDE_TAU => self.glide_tau_s = value.clamp(0.01, 0.5),
            _ => return false,
        }
        true
    }
}

#[derive(Clone, Copy)]
struct Voice {
    modes: [Mode; VOICE_MODES],
    /// Modal velocity delivered by the current stick/head contact. Unlike
    /// an impulse that fills every oscillator in one sample, this force is
    /// spread over a short raised-sine pulse and therefore lets each mode
    /// rotate while the stick is still pushing the head.
    drive: [f32; VOICE_MODES],
    contact_frame: u16,
    contact_frames: u16,
    contact_norm: f32,
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
    glide_tau_s: f32,
    /// The crack: an exponentially dying, band-limited noise burst.
    crack_amp: f32,
    crack_decay: f32,
    crack_lp: f32,
    crack_hp: f32,
    crack_state: f32,
    crack_low_state: f32,
    rng: u32,
    /// The wires, collectively: gate level fixed at strike, an asymmetric
    /// envelope follower over the head's motion, and a band-pass state pair
    /// for the sizzle's colour.
    wires_level: f32,
    wire_env: f32,
    wire_lp: f32,
    wire_band: f32,
    /// The pair's view of this drum: equal-power gains and integer arrival
    /// delays per side, fixed at note-on from the kit geometry.
    pan_left: f32,
    pan_right: f32,
    delay_left: usize,
    delay_right: usize,
    active: bool,
    note: u8,
    age: u32,
}

impl Default for Voice {
    fn default() -> Self {
        Self {
            modes: [Mode::default(); VOICE_MODES],
            drive: [0.0; VOICE_MODES],
            contact_frame: 0,
            contact_frames: 0,
            contact_norm: 0.0,
            omega: [0.0; VOICE_MODES],
            decay: [0.0; VOICE_MODES],
            glide: 0.0,
            glide_tau_s: 0.05,
            crack_amp: 0.0,
            crack_decay: 0.0,
            crack_lp: 0.0,
            crack_hp: 0.0,
            crack_state: 0.0,
            crack_low_state: 0.0,
            rng: 1,
            wires_level: 0.0,
            wire_env: 0.0,
            wire_lp: 0.0,
            wire_band: 0.0,
            pan_left: core::f32::consts::FRAC_1_SQRT_2,
            pan_right: core::f32::consts::FRAC_1_SQRT_2,
            delay_left: 0,
            delay_right: 0,
            active: false,
            note: 0,
            age: 0,
        }
    }
}

impl Voice {
    /// Installs a struck mode and remembers its rest rotation for the glide.
    fn install(&mut self, index: usize, amp: f32, frequency: f32, decay: f32, rate: f32) {
        self.modes[index] = Mode::strike(amp, frequency * (1.0 + self.glide), decay, rate);
        if self.modes[index].is_live() {
            self.omega[index] = core::f32::consts::TAU * frequency / rate;
            self.decay[index] = decay;
        }
    }

    /// Installs an oscillator at rest and records the velocity that the
    /// finite stick contact will deliver to it. The target is the same modal
    /// projection an ideal impulse would leave, but its phase now emerges
    /// from the actual contact duration.
    fn install_driven(
        &mut self,
        index: usize,
        amp: f32,
        frequency: f32,
        decay: f32,
        rate: f32,
    ) {
        self.install(index, 0.0, frequency, decay, rate);
        self.drive[index] = amp;
    }

    fn start_contact(&mut self, contact_s: f32, sample_rate: f32) {
        // The measured geometric contact is wider than the force peak: the
        // tip hardens as it compresses. CONTACT_BRIGHTNESS carries that
        // ratio until measured force traces replace it. The upper bound
        // keeps an extreme calibration value below one control interval.
        let frames = roundf(contact_s * sample_rate / CONTACT_BRIGHTNESS)
            .clamp(2.0, CULL_INTERVAL as f32) as u16;
        self.contact_frame = 0;
        self.contact_frames = frames;
        // Exact normalization for samples sin(pi*(k+1/2)/N): their sum is
        // csc(pi/(2N)), so multiplying by sin(pi/(2N)) gives unit impulse.
        let (norm, _) = sincosf(core::f32::consts::PI / (2.0 * frames as f32));
        self.contact_norm = norm;
    }

    #[inline(always)]
    fn contact_tick(&mut self) -> f32 {
        if self.contact_frame >= self.contact_frames {
            return 0.0;
        }
        let phase = core::f32::consts::PI
            * (self.contact_frame as f32 + 0.5)
            / self.contact_frames as f32;
        self.contact_frame += 1;
        let (force, _) = sincosf(phase);
        force * self.contact_norm
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

    /// The wires, collectively — the snare's defining nonlinearity, in its
    /// stated first form.
    ///
    /// Twenty steel wires lie against the resonant head. While the head
    /// moves hard they are thrown off and re-strike it many times per
    /// period — a self-gating noise source whose loudness follows the
    /// head's motion and whose colour is the wires' own bright band. The
    /// honest model is per-wire collision (Avanzini & Serafin's family);
    /// this is the collective reading of it, and the ledger says so: an
    /// asymmetric envelope follower over the voice's motion (fast to rise,
    /// ~60 ms to fall — wires settle after the head does), a soft-knee gate
    /// so a whisper of motion leaves the wires seated, and white noise
    /// through a 1.5–7 kHz band-pass under that gate. What this omits, and
    /// will be measured missing: the wires re-exciting the head's high
    /// modes, and the wires' own ring after separation.
    ///
    /// Cost: two LCG draws, two one-pole states, per sample, on one drum.
    #[inline(always)]
    fn wire_tick(&mut self, head: f32) -> f32 {
        if self.wires_level == 0.0 {
            return 0.0;
        }
        let drive = if head < 0.0 { -head } else { head };
        // Fast attack, slow release: the sizzle hangs behind the hit.
        let rate = if drive > self.wire_env { 0.4 } else { 0.000_35 };
        self.wire_env += rate * (drive - self.wire_env);
        // Soft knee: below the knee the wires stay seated and the gate
        // closes quadratically, which is what "gated by motion" sounds
        // like against a linear fader's constant hiss.
        const KNEE: f32 = 0.004;
        let gate = self.wire_env * self.wire_env / (self.wire_env + KNEE);
        if gate < 1.0e-6 {
            return 0.0;
        }
        self.rng = self.rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let white = (self.rng >> 8) as f32 * (1.0 / 8_388_608.0) - 1.0;
        // Band-pass: one-pole low at ~7 kHz minus one-pole low at ~1.5 kHz.
        self.wire_lp += 0.65 * (white - self.wire_lp);
        self.wire_band += 0.18 * (self.wire_lp - self.wire_band);
        (self.wire_lp - self.wire_band) * gate * self.wires_level
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
        // A stick click is a band, not low-passed DC noise. The first pole
        // removes ultrasonic grit, the second removes the low body already
        // carried by the membrane. Their difference is the short woody edge
        // heard before the head has completed even one cycle.
        self.crack_state += self.crack_lp * (white - self.crack_state);
        self.crack_low_state += self.crack_hp * (self.crack_state - self.crack_low_state);
        self.crack_amp *= self.crack_decay;
        (self.crack_state - self.crack_low_state) * self.crack_amp
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
    room_mix: f32,
    room_size: f32,
    // The room: six delay lines, compare-and-wrap counters (the piano's
    // lesson: an integer division per line per sample was 12% of the
    // callback), one-pole damping state and a per-line gain from RT60.
    room: [[f32; ROOM_BUFFER]; ROOM_LINES],
    room_len: [usize; ROOM_LINES],
    room_pos: [usize; ROOM_LINES],
    room_gain: [f32; ROOM_LINES],
    room_damp_state: [f32; ROOM_LINES],
    room_damp: f32,
    room_dirty: bool,
    // The direct bus: voices write ahead of the read point by their
    // arrival delay.
    direct_left: [f32; DIRECT_BUFFER],
    direct_right: [f32; DIRECT_BUFFER],
    direct_pos: usize,
    /// Monotonic within an instance, used only to vary strike bearing and
    /// contact noise. Modal frequencies remain those of one particular kit.
    strike_counter: u32,
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
            room_mix: 0.3,
            room_size: 0.35,
            room: [[0.0; ROOM_BUFFER]; ROOM_LINES],
            room_len: [ROOM_BUFFER; ROOM_LINES],
            room_pos: [0; ROOM_LINES],
            room_gain: [0.0; ROOM_LINES],
            room_damp_state: [0.0; ROOM_LINES],
            room_damp: 0.3,
            room_dirty: true,
            direct_left: [0.0; DIRECT_BUFFER],
            direct_right: [0.0; DIRECT_BUFFER],
            direct_pos: 0,
            strike_counter: 0,
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
        self.strike_counter = self.strike_counter.wrapping_add(1);
        let hit_seed = self.strike_counter ^ ((note as u32) << 16) ^ 0x9e37_79b9;
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
        // Even a trained hand does not land on the same molecule twice.
        // Two percent of radius is enough to break identical-machine-gun
        // spectra in a roll without turning the position control into a
        // random zone selector.
        let radius_jitter = (hash01(hit_seed ^ 0xa511_e9b3) - 0.5) * 0.04;
        let radius =
            (spec.strike_radius + (self.position - 0.5) * 0.9 + radius_jitter).clamp(0.02, 0.98);
        // Bearing selects the cos/sin members of every degenerate family.
        // The panel exposes radius because that is the strong timbral axis;
        // bearing varies subtly from hit to hit like a real stick path.
        let strike_angle = core::f32::consts::TAU * hash01(hit_seed ^ 0x63d8_35f1);
        // Contact time lengthens for soft blows (a stick thrown gently sinks
        // into the head longer), shortening — brightening — with velocity.
        // The swing is ~4x across the dynamic range (soft ~2.4x the spec
        // time, hard ~0.55x), the order stick contacts actually span; the
        // first cut used 1.67x and the velocity-to-timbre road measured
        // almost flat once the room diluted it.
        let contact = spec.contact_s * (2.45 - 1.9 * velocity_01);
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
        voice.glide = spec.glide * velocity_01 * velocity_01;
        voice.glide_tau_s = spec.glide_tau_s;

        // The crack: the stick's own broadband impact, brighter and louder
        // for the hard blow, its bandwidth tied to the same contact time
        // that shapes the modal ladder.
        let crack_cutoff = (12.0 / contact).min(0.45 * self.sample_rate);
        let crack_floor = (700.0f32).max(4.0 * pitch).min(0.25 * self.sample_rate);
        voice.crack_amp =
            1.5 * spec.crack * spec.gain * (0.06 + 0.94 * velocity_01 * velocity_01);
        voice.crack_decay = expf(-1.0 / (0.75 * contact * self.sample_rate));
        voice.crack_lp =
            1.0 - expf(-core::f32::consts::TAU * crack_cutoff / self.sample_rate);
        voice.crack_hp =
            1.0 - expf(-core::f32::consts::TAU * crack_floor / self.sample_rate);
        voice.rng = hit_seed;
        voice.start_contact(contact, self.sample_rate);
        // The wires' gate level; the ×4 sets the fader's centre near the
        // audible balance and is voicing, not physics.
        voice.wires_level = spec.wires * 4.0;

        // Where the pair hears this drum: equal-power gains from its lateral
        // place, and each side's own arrival time — path length to each
        // microphone over the speed of sound, the earlier side at zero.
        let x = DRUM_X_M[drum];
        let pan = (x / 0.8).clamp(-1.0, 1.0);
        voice.pan_left = sqrtf(0.5 * (1.0 - pan));
        voice.pan_right = sqrtf(0.5 * (1.0 + pan));
        let path = |mic_x: f32| {
            let dx = x - mic_x;
            sqrtf(dx * dx + PAIR_DISTANCE_M * PAIR_DISTANCE_M) / SOUND_SPEED
        };
        let (t_left, t_right) = (path(-0.5 * PAIR_SPACING_M), path(0.5 * PAIR_SPACING_M));
        let earlier = if t_left < t_right { t_left } else { t_right };
        voice.delay_left =
            (((t_left - earlier) * self.sample_rate) as usize).min(DIRECT_BUFFER - 1);
        voice.delay_right =
            (((t_right - earlier) * self.sample_rate) as usize).min(DIRECT_BUFFER - 1);

        for (index, &alpha) in BESSEL_ZEROS.iter().enumerate() {
            let m = angular_order(index);
            let frequency = f11
                * air_loaded_ratio(alpha, spec.air_load)
                * bending_sharpen(alpha / membrane::ALPHA_11);
            // At angle zero `strike_shape` is the radial Bessel projection.
            // The angular share is applied explicitly below so the two
            // degenerate partners receive cos(mθ) and sin(mθ), respectively.
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
                // The breathing pair, with the real tuning between heads:
                // the coupled eigenmodes of batter and resonant head, each
                // taking the batter's blow in its eigenvector share. The
                // port relieves the cavity spring — an open cavity squeezes
                // less — which is half of what porting a kick does.
                let cavity =
                    spec.cavity_stiffness * (1.0 - 0.5 * spec.port.clamp(0.0, 1.0));
                let ((f_lower, w_lower), (f_upper, w_upper)) =
                    couple_detuned_heads(frequency, frequency * spec.res_tune, cavity);
                let swept = volume_displacement(m, alpha).abs();
                let base = amp * (0.6 + 0.45 * swept);
                let n_index = index - (m as usize) * membrane::RADIAL_ORDERS;
                voice.install_driven(index, base * w_lower, f_lower, decay, self.sample_rate);
                let upper_t60 = t60 * 0.7;
                let upper_decay = decay_per_sample(upper_t60, self.sample_rate);
                voice.install_driven(
                    MODE_COUNT + n_index,
                    base * w_upper,
                    f_upper,
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
                let (angular_sin, angular_cos) = sincosf(m as f32 * strike_angle);
                // 0.72 keeps the pair's total energy close to the old
                // voicing while replacing its unphysical fixed 60/50 split.
                voice.install_driven(
                    index,
                    amp * 0.72 * angular_cos,
                    frequency,
                    decay,
                    self.sample_rate,
                );
                voice.install_driven(
                    PAIR_BASE + (index - membrane::RADIAL_ORDERS),
                    amp * 0.72 * angular_sin,
                    frequency + split,
                    decay,
                    self.sample_rate,
                );
                // The resonant head's own ladder: the same family at the
                // bottom head's tuning, reached through shell and cavity
                // rather than struck, so it speaks softer and — unstruck,
                // undamped by the stick — rings longer. The beat of the two
                // (1,1)s at their detuning is the sung centre of a tom's
                // sustain. Coupling weight is a stated placeholder until
                // the shell drive is continuous.
                let radial_order = index % membrane::RADIAL_ORDERS;
                let transfer = 0.28
                    / (1.0 + 0.12 * m as f32 + 0.10 * radial_order as f32);
                let polarity = if hash01(0x51ed_270b ^ ((drum as u32) << 8) ^ index as u32) < 0.5 {
                    -1.0
                } else {
                    1.0
                };
                voice.install(
                    RES_BASE + (index - membrane::RADIAL_ORDERS),
                    amp * transfer * polarity,
                    frequency * spec.res_tune,
                    decay_per_sample(t60 * 1.6, self.sample_rate),
                    self.sample_rate,
                );
            }
        }

        // The port: the vent's Helmholtz resonance, rung by the breathing
        // modes' displaced volume. One oscillator, air-fast decay, absent
        // on unported drums. Its frequency follows the described drum (a
        // stated placeholder pending real cavity/port geometry) and its
        // relief of the cavity spring is applied above.
        if spec.port > 0.01 {
            let f_port = (0.75 * pitch).clamp(30.0, 90.0);
            voice.install(
                PORT_SLOT,
                velocity_amp * spec.port * 1.4,
                f_port,
                decay_per_sample(0.18, self.sample_rate),
                self.sample_rate,
            );
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

    /// Derives the room from the size control: line lengths from the mean
    /// free path of the described volume, per-line gains from a Sabine-order
    /// RT60, damping so highs die faster — the Concert Grand's chamber,
    /// sized for a tracking room rather than a hall. Runs off the audio
    /// path, flagged by `room_dirty`.
    fn tune_room(&mut self) {
        self.room_dirty = false;
        // size 0..1 sweeps a close booth (~15 m³) to a live tracking room
        // (~250 m³). Mean free path 4V/S for a plausible shoebox of that
        // volume; base delay = mfp / c.
        let volume = 15.0 + 235.0 * self.room_size * self.room_size;
        // Shoebox with 2.5:2:1 proportions: V = 5·k³ → surfaces from k.
        let k = powf(volume / 5.0, 1.0 / 3.0);
        let surface = 2.0 * (2.5 * k * 2.0 * k + 2.5 * k * k + 2.0 * k * k);
        let mean_free_path = 4.0 * volume / surface;
        let base = mean_free_path / SOUND_SPEED * self.sample_rate;
        // A tracking room's RT60: 0.25 s in the booth to ~1.1 s live.
        let rt60 = 0.25 + 0.85 * self.room_size;
        for line in 0..ROOM_LINES {
            let length = ((base * ROOM_SPREAD[line]) as usize).clamp(31, ROOM_BUFFER - 1);
            self.room_len[line] = length;
            self.room_pos[line] = 0;
            // Gain for -60 dB over RT60 across this line's round trip.
            self.room_gain[line] =
                powf(10.0, -3.0 * length as f32 / (rt60 * self.sample_rate));
            self.room[line] = [0.0; ROOM_BUFFER];
            self.room_damp_state[line] = 0.0;
        }
        // One-pole in each feedback path: highs die roughly twice as fast.
        self.room_damp = 1.0 - expf(-core::f32::consts::TAU * 4200.0 / self.sample_rate);
    }

    /// The control step: tension relaxation, then the cull.
    fn cull(&mut self) {
        for voice in self.voices.iter_mut() {
            if !voice.active {
                continue;
            }
            let glide_keep = expf(
                -(CULL_INTERVAL as f32) / (voice.glide_tau_s * self.sample_rate),
            );
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
            PARAM_ROOM_MIX => self.room_mix = value.clamp(0.0, 1.0),
            PARAM_ROOM_SIZE => {
                self.room_size = value.clamp(0.0, 1.0);
                self.room_dirty = true;
            }
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
            PARAM_ROOM_MIX => self.room_mix as f64,
            PARAM_ROOM_SIZE => self.room_size as f64,
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
        self.direct_left = [0.0; DIRECT_BUFFER];
        self.direct_right = [0.0; DIRECT_BUFFER];
        self.direct_pos = 0;
        self.strike_counter = 0;
        self.room_dirty = true;
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
        if self.room_dirty {
            self.tune_room();
        }
        let level = self.level * self.level;
        // Perceived-loudness taper for the mix control; the reverberant
        // field's power is the square.
        let room_send = self.room_mix * self.room_mix * 0.8;
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
            // Each voice lands on the pair through its own gains and
            // arrival delays (writes ahead of the read point); the room is
            // fed the unpanned sum — the reverberant field has forgotten
            // where the drum was, which is what "diffuse" means.
            let mut room_in = 0.0f32;
            for voice in self.voices.iter_mut() {
                if !voice.active {
                    continue;
                }
                let contact_force = voice.contact_tick();
                let mut sum = 0.0f32;
                for (index, mode) in voice.modes.iter_mut().enumerate() {
                    if contact_force != 0.0 {
                        mode.c += voice.drive[index] * contact_force;
                    }
                    sum += mode.tick();
                }
                sum += voice.wire_tick(sum);
                sum += voice.crack_tick();
                let write = self.direct_pos;
                self.direct_left[(write + voice.delay_left) & (DIRECT_BUFFER - 1)] +=
                    sum * voice.pan_left;
                self.direct_right[(write + voice.delay_right) & (DIRECT_BUFFER - 1)] +=
                    sum * voice.pan_right;
                room_in += sum;
            }
            let direct_l = self.direct_left[self.direct_pos];
            let direct_r = self.direct_right[self.direct_pos];
            self.direct_left[self.direct_pos] = 0.0;
            self.direct_right[self.direct_pos] = 0.0;
            self.direct_pos = (self.direct_pos + 1) & (DIRECT_BUFFER - 1);

            // The room: read all six lines, Householder-reflect the sum back
            // with per-line gain and damping, take the tail off alternating
            // lines per side — decorrelated left and right by construction.
            let mut reads = [0.0f32; ROOM_LINES];
            let mut total = 0.0f32;
            for line in 0..ROOM_LINES {
                reads[line] = self.room[line][self.room_pos[line]];
                total += reads[line];
            }
            let householder = total * (2.0 / ROOM_LINES as f32);
            for line in 0..ROOM_LINES {
                let feedback = (reads[line] - householder) * self.room_gain[line];
                self.room_damp_state[line] +=
                    self.room_damp * (feedback - self.room_damp_state[line]);
                self.room[line][self.room_pos[line]] =
                    room_in * room_send * ROOM_INJECT[line] + self.room_damp_state[line];
                self.room_pos[line] += 1;
                if self.room_pos[line] >= self.room_len[line] {
                    self.room_pos[line] = 0;
                }
            }
            let room_l = reads[0] + reads[2] + reads[4];
            let room_r = reads[1] + reads[3] + reads[5];

            let left = soft_clip((direct_l + room_l) * level);
            let right = soft_clip((direct_r + room_r) * level);
            match channels {
                0 => {}
                1 => output[frame] = 0.5 * (left + right),
                _ => {
                    output[frame * channels] = left;
                    output[frame * channels + 1] = right;
                    for channel in 2..channels {
                        output[frame * channels + channel] = 0.0;
                    }
                }
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
    // The final clamp is not styling: the rational form reaches 1.0 exactly
    // at the rail and f32 rounding can land a ulp above it (measured:
    // 1.0000001 in a dense roll), which a host is entitled to reject.
    (x * (27.0 + x * x) / (27.0 + 9.0 * x * x)).clamp(-1.0, 1.0)
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
            let peak = output.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            let finite = output.iter().all(|x| x.is_finite());
            assert!(
                finite && peak <= 1.0,
                "round {round} produced bad samples: finite {finite}, peak {peak}"
            );
        }
    }

    fn channel_rms(samples: &[f32], channel: usize) -> f32 {
        let sum: f64 = samples
            .chunks(2)
            .map(|frame| (frame[channel] as f64) * (frame[channel] as f64))
            .sum();
        ((sum / (samples.len() / 2) as f64) as f32).sqrt()
    }

    /// The kit must be a stereo image, not dual mono: the floor tom sits
    /// right of centre, the snare left, and each drum's two channels must
    /// actually differ (gains AND arrival times — a pan-pot alone leaves
    /// identical shapes at different levels; the delay decorrelates them).
    #[test]
    fn the_kit_sits_in_a_stereo_image() {
        let render_drum = |note: u8| {
            let mut drums = RfDrums::default();
            assert!(drums.prepare(48_000.0, 512, 0, 2));
            render(&mut drums, 24_000, &strike(note, 110))
        };
        let floor = render_drum(41);
        assert!(
            channel_rms(&floor, 1) > channel_rms(&floor, 0) * 1.15,
            "floor tom not right of centre: L {} R {}",
            channel_rms(&floor, 0),
            channel_rms(&floor, 1)
        );
        let snare = render_drum(38);
        assert!(
            channel_rms(&snare, 0) > channel_rms(&snare, 1) * 1.1,
            "snare not left of centre: L {} R {}",
            channel_rms(&snare, 0),
            channel_rms(&snare, 1)
        );
        // And the sides are genuinely different signals, not one signal at
        // two levels: normalized cross-correlation at lag zero well below 1.
        let (mut dot, mut left_sq, mut right_sq) = (0.0f64, 0.0f64, 0.0f64);
        for frame in floor.chunks(2) {
            dot += frame[0] as f64 * frame[1] as f64;
            left_sq += (frame[0] as f64).powi(2);
            right_sq += (frame[1] as f64).powi(2);
        }
        let correlation = dot / (left_sq.sqrt() * right_sq.sqrt()).max(1e-30);
        assert!(
            correlation < 0.985,
            "channels are the same signal: correlation {correlation}"
        );
    }

    /// The room must leave a tail the dry kit does not have, and the loop
    /// must be stable: the tail decays instead of ringing on.
    #[test]
    fn the_room_leaves_a_decaying_tail() {
        let render_with_mix = |mix: f64| {
            let mut drums = RfDrums::default();
            assert!(drums.prepare(48_000.0, 512, 0, 2));
            assert!(drums.set_parameter(PARAM_ROOM_MIX, mix));
            assert!(drums.set_parameter(PARAM_ROOM_SIZE, 0.6));
            render(&mut drums, 48_000 * 3, &strike(36, 120))
        };
        let dry = render_with_mix(0.0);
        let wet = render_with_mix(0.5);
        // The kick's own modes are gone within a second; what lives at
        // 1.2-1.6 s in the wet render is the room.
        let window = |out: &[f32], from: usize, to: usize| {
            channel_rms(&out[2 * from..2 * to], 0) + channel_rms(&out[2 * from..2 * to], 1)
        };
        let dry_tail = window(&dry, 57_600, 76_800);
        let wet_tail = window(&wet, 57_600, 76_800);
        assert!(
            wet_tail > dry_tail * 3.0 + 1e-9,
            "room silent: dry tail {dry_tail}, wet tail {wet_tail}"
        );
        // Stability: the last quarter second must sit well below the tail.
        let late = window(&wet, 48_000 * 3 - 12_000, 48_000 * 3);
        assert!(
            late < wet_tail * 0.5,
            "room does not decay: tail {wet_tail}, late {late}"
        );
        assert!(wet.iter().all(|x| x.is_finite()));
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

    /// The stick contact must contribute a genuinely broadband edge, not a
    /// second low drum thump. This compares the same head with only the
    /// contact-noise path removed, over the first 12 ms.
    #[test]
    fn the_stick_contact_has_a_broadband_edge() {
        let base = SPEC_PARAM_BASE + 2 * SPEC_PARAM_STRIDE; // floor tom
        let render_with_crack = |crack: f64| {
            let mut drums = RfDrums::default();
            assert!(drums.prepare(48_000.0, 512, 0, 2));
            assert!(drums.set_parameter(PARAM_ROOM_MIX, 0.0));
            assert!(drums.set_parameter(base + FIELD_CRACK, crack));
            render(&mut drums, 576, &strike(41, 115))
        };
        let with_stick = render_with_crack(1.0);
        let head_only = render_with_crack(0.0);
        let stick_high = band_energy(&with_stick, 48_000.0, 1_000.0, 8_000.0);
        let head_high = band_energy(&head_only, 48_000.0, 1_000.0, 8_000.0);
        assert!(
            stick_high > head_high * 3.0,
            "stick has no broadband edge: with {stick_high}, head only {head_high}"
        );
    }

    /// Consecutive hits move around the head by a tiny amount, changing
    /// modal amplitudes, while the physical modal frequencies stay fixed.
    #[test]
    fn repeated_hits_vary_projection_not_tuning() {
        let mut drums = RfDrums::default();
        assert!(drums.prepare(48_000.0, 512, 0, 2));
        drums.strike(41, 100);
        drums.strike(41, 100);
        let first = &drums.voices[0];
        let second = &drums.voices[1];
        let different_amplitudes = first
            .modes
            .iter()
            .zip(second.modes.iter())
            .filter(|(a, b)| (a.c - b.c).abs() > 1.0e-5)
            .count();
        assert!(different_amplitudes > 10, "successive hits are identical");
        for (a, b) in first.omega.iter().zip(second.omega.iter()) {
            assert!((a - b).abs() < 1.0e-7, "strike variation detuned the drum");
        }
    }

    /// A stick delivers a normalized force over several samples. Softer
    /// strokes stay in contact longer, and the modal bank begins at rest
    /// instead of being filled instantaneously.
    #[test]
    fn stick_force_has_finite_velocity_dependent_contact() {
        let mut hard = RfDrums::default();
        assert!(hard.prepare(48_000.0, 512, 0, 2));
        hard.strike(41, 120);
        let hard_frames = hard.voices[0].contact_frames;
        assert!(hard_frames >= 2);
        assert!(hard.voices[0].drive.iter().any(|amp| amp.abs() > 1.0e-4));
        assert!(hard.voices[0].modes[..SHELL_BASE]
            .iter()
            .all(|mode| mode.c == 0.0 && mode.s == 0.0));

        let mut soft = RfDrums::default();
        assert!(soft.prepare(48_000.0, 512, 0, 2));
        soft.strike(41, 30);
        assert!(soft.voices[0].contact_frames > hard_frames);

        let voice = &mut hard.voices[0];
        let mut impulse = 0.0;
        for _ in 0..voice.contact_frames {
            impulse += voice.contact_tick();
        }
        assert!((impulse - 1.0).abs() < 1.0e-4, "contact impulse {impulse}");
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

    /// The wires must sizzle while the head moves and settle after it: the
    /// snare with wires holds far more sustained high band than the same
    /// drum with the strainer thrown off, and the sizzle must FOLLOW the
    /// motion (more of it early than late) rather than hiss statically.
    #[test]
    fn the_wires_sizzle_with_the_head_and_settle_after_it() {
        let base = SPEC_PARAM_BASE + SPEC_PARAM_STRIDE; // snare, drum 1
        let render_with_wires = |wires: f64| {
            let mut drums = RfDrums::default();
            assert!(drums.prepare(48_000.0, 512, 0, 2));
            assert!(drums.set_parameter(base + FIELD_WIRES, wires));
            render(&mut drums, 48_000, &strike(38, 110))
        };
        let with_wires = render_with_wires(1.2);
        let thrown_off = render_with_wires(0.0);
        let band = |out: &[f32], from: usize, to: usize| {
            band_energy(&out[2 * from..2 * to], 48_000.0, 2000.0, 7000.0)
        };
        // Sustained sizzle: 60-250 ms, past the crack, before the settle.
        let sizzle = band(&with_wires, 2_880, 12_000);
        let dry = band(&thrown_off, 2_880, 12_000);
        assert!(
            sizzle > dry * 3.0,
            "wires silent: with {sizzle}, thrown off {dry}"
        );
        // And gated by motion: the early sizzle must beat the late tail.
        let late = band(&with_wires, 28_800, 38_400);
        assert!(
            sizzle > late * 2.0,
            "wires hiss statically: early {sizzle}, late {late}"
        );
    }

    /// Bending stiffness must sharpen the ladder upward, monotonically, and
    /// by the plate-order magnitude — not rewrite the bottom.
    #[test]
    fn bending_sharpens_the_top_of_the_ladder() {
        assert!((bending_sharpen(1.0) - 1.0).abs() < 0.003, "the pitch reference barely moves");
        let top = membrane::BESSEL_ZEROS[membrane::MODE_COUNT - 1] / membrane::ALPHA_11;
        let sharpened = bending_sharpen(top);
        assert!(
            sharpened > 1.05 && sharpened < 1.2,
            "top of the ladder off the plate order: {sharpened}"
        );
        assert!(bending_sharpen(3.0) > bending_sharpen(2.0));
    }

    /// The resonant head must be audible AS a second tuning: dropping
    /// res_tune moves its ladder where the batter's is not, and the band at
    /// the moved (1,1) partner must light up against the stock render.
    #[test]
    fn the_resonant_head_is_a_second_tuning() {
        let base = SPEC_PARAM_BASE + 2 * SPEC_PARAM_STRIDE; // floor tom
        let mut retuned = RfDrums::default();
        assert!(retuned.prepare(48_000.0, 512, 0, 2));
        assert!(retuned.set_parameter(base + FIELD_RES_TUNE, 0.80));
        assert_eq!(retuned.get_parameter(base + FIELD_RES_TUNE), Some(0.800000011920929));
        let moved = render(&mut retuned, 24_000, &strike(41, 110));
        let mut stock = RfDrums::default();
        assert!(stock.prepare(48_000.0, 512, 0, 2));
        let stock_render = render(&mut stock, 24_000, &strike(41, 110));
        // The (1,1) partner at 0.80 x 82 = 65.6 Hz sits in a gap of the
        // stock spectrum (stock partner rings at 77 Hz).
        let moved_band = band_energy(&moved, 48_000.0, 62.0, 70.0);
        let stock_band = band_energy(&stock_render, 48_000.0, 62.0, 70.0);
        assert!(
            moved_band > stock_band * 2.5,
            "res head inert: stock 62-70 Hz {stock_band}, res_tune 0.8 {moved_band}"
        );
    }

    /// The ported kick must breathe below its head pitch — the Helmholtz
    /// voice — and closing the port must take that breath away.
    #[test]
    fn the_kick_breathes_through_its_port() {
        let base = SPEC_PARAM_BASE; // kick is drum 0
        let render_with_port = |port: f64| {
            let mut drums = RfDrums::default();
            assert!(drums.prepare(48_000.0, 512, 0, 2));
            assert!(drums.set_parameter(base + FIELD_PORT, port));
            render(&mut drums, 24_000, &strike(36, 120))
        };
        let ported = render_with_port(0.8);
        let closed = render_with_port(0.0);
        // The port voice sits at 0.75 x 55 = 41 Hz.
        let ported_band = band_energy(&ported, 48_000.0, 36.0, 47.0);
        let closed_band = band_energy(&closed, 48_000.0, 36.0, 47.0);
        assert!(
            ported_band > closed_band * 1.5,
            "port silent: closed {closed_band}, ported {ported_band}"
        );
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
            ("snare", 38, 110, 2.0),
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
