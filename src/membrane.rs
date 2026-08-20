//! The circular membrane: where a drum's partials come from.
//!
//! A string's ladder is one line — `f_n = n·f₀·√(1+Bn²)` — and every partial
//! is a small perturbation of a harmonic series. A drumhead has no harmonic
//! series to perturb. Its modes are the zeros of Bessel functions, they are
//! inharmonic from first principles, and *that inharmonicity is the sound*.
//!
//! Reference: Fletcher & Rossing, *The Physics of Musical Instruments*, ch. 3
//! (the ideal circular membrane) and ch. 18 (drums); Rossing, *Science of
//! Percussion Instruments*, ch. 2–4.

use crate::math::{besselj, sqrtf};

/// Radial orders `n = 1..=5` and angular orders `m = 0..=8`, giving 45 mode
/// families. Higher orders exist and are audible on a large head; they are
/// added by the shell and plate work rather than by extending this table,
/// because past this point the modes are dense enough that individual
/// identity stops mattering — the same two-scale argument the Concert Grand
/// uses for its open register.
pub const RADIAL_ORDERS: usize = 5;
pub const ANGULAR_ORDERS: usize = 9;
pub const MODE_COUNT: usize = ANGULAR_ORDERS * RADIAL_ORDERS;

/// The `n`-th positive zero of `J_m`, laid out as `[m][n]` flattened —
/// `BESSEL_ZEROS[m * RADIAL_ORDERS + (n - 1)]`.
///
/// These are data, not fits, and `math::tabulated_zeros_are_zeros` checks
/// them against this crate's own `besselj` so a transcription error cannot
/// survive: a wrong zero is a wrong partial *and* a wrong strike projection,
/// and nothing else in the model would report it.
pub static BESSEL_ZEROS: [f32; MODE_COUNT] = [
    // m = 0
    2.404_826, 5.520_078, 8.653_728, 11.791_534, 14.930_918,
    // m = 1
    3.831_706, 7.015_587, 10.173_468, 13.323_692, 16.470_63,
    // m = 2
    5.135_622, 8.417_244, 11.619_841, 14.795_952, 17.959_819,
    // m = 3
    6.380_162, 9.761_023, 13.015_201, 16.223_466, 19.409_415,
    // m = 4
    7.588_342, 11.064_709, 14.372_537, 17.615_966, 20.826_933,
    // m = 5
    8.771_484, 12.338_604, 15.700_174, 18.980_134, 22.217_8,
    // m = 6
    9.936_11, 13.589_29, 17.003_82, 20.320_789, 23.586_084,
    // m = 7
    11.086_37, 14.821_269, 18.287_582, 21.641_542, 24.934_928,
    // m = 8
    12.225_092, 16.037_774, 19.554_536, 22.945_173, 26.266_814,
];

/// The (1,1) mode's zero. Every frequency in the model is quoted against it,
/// because (1,1) — not (0,1) — is the mode a drum's perceived pitch sits on:
/// it is the lowest mode that survives a normal off-centre strike with any
/// strength, and on a timpano it is the one the tuning gauge reads.
pub const ALPHA_11: f32 = 3.831_706;

/// Index of mode `(m, n)`, `n` counted from 1.
pub const fn index_of(m: usize, n: usize) -> usize {
    m * RADIAL_ORDERS + (n - 1)
}

/// The angular order of a flattened index.
pub const fn angular_order(index: usize) -> u32 {
    (index / RADIAL_ORDERS) as u32
}

/// How much slower a loaded mode runs than the ideal membrane's, and why the
/// ratio is not a constant.
///
/// An ideal membrane vibrates in vacuum. A real head drags a layer of air
/// with it, and that added mass lowers every mode — but not equally. The
/// layer a mode entrains is about one wavelength thick, so its added mass per
/// unit area goes as `ρ_air/k_mn`, and the wavenumber is `k_mn = α_mn/a`.
/// The low modes, with the longest wavelengths, are loaded hardest:
///
/// ```text
/// f_mn ∝ α_mn / √(1 + β·α₁₁/α_mn)
/// ```
///
/// with `β` the loading of the (1,1) mode itself — one number carrying the
/// head's mass per unit area, the air's density and the drum's radius.
///
/// **This is derived, not drawn, and the measurement it lands on is not the
/// one it was fitted to.** Rossing reports that a kettledrum's air loading
/// pulls (1,1), (2,1), (3,1) and (4,1) into near 1 : 1.5 : 2 : 2.5 — the
/// harmonic-sounding set that gives a timpano a definite pitch, against the
/// ideal membrane's 1 : 1.34 : 1.67 : 1.98. Solving the law above for β on
/// the (2,1) ratio *alone* gives β ≈ 3.8, and at that value the other two
/// come out at 2.01 and 2.54 against the published 2 and 2.5. Two of the
/// three ratios are predictions, and they land. `air_loaded_ratios` holds
/// this.
///
/// A tom or a snare carries a heavier head on a shallower body and loads far
/// less — β well under one — which is exactly why a tom keeps the ideal
/// membrane's clangour where a timpano sings.
pub fn air_loaded_ratio(alpha: f32, air_load: f32) -> f32 {
    let ideal = alpha / ALPHA_11;
    let loading_here = 1.0 + air_load * ALPHA_11 / alpha;
    let loading_reference = 1.0 + air_load;
    ideal * sqrtf(loading_reference / loading_here)
}

/// The mode shape `J_m(α·r/a)·cos(mθ)` evaluated where the stick lands.
///
/// `radius` is the strike's distance from the centre as a fraction of the
/// head's radius, and `angle` is its bearing in radians. This is the
/// membrane's answer to the piano's `sin(nπx₀)` strike-point comb, and it
/// does the same job: a mode with a node under the stick is not excited, so
/// where a drummer hits changes *which partials exist*, not just the level.
///
/// Centre and edge are the two ends of it. At `radius = 0` every `m ≥ 1` mode
/// has `J_m(0) = 0` and only the `(0, n)` family speaks — the dull, pitchless
/// thud of a dead-centre hit. Toward the rim the high-`m` modes take over and
/// the sound opens into the ringing, pitched tone a drummer plays for.
pub fn strike_shape(m: u32, alpha: f32, radius: f32, angle: f32) -> f32 {
    let radial = besselj(m, alpha * radius.clamp(0.0, 1.0));
    if m == 0 {
        return radial;
    }
    let (_, cos) = crate::math::sincosf(m as f32 * angle);
    radial * cos
}

/// The modal norm `∫∫ψ² dA / a²`, which is what turns a shape into a mass.
///
/// For `ψ = J_m(α r/a)·cos(mθ)`, the radial integral is the standard
/// `(a²/2)·J_{m+1}(α)²` and the angular one contributes `π` for `m ≥ 1` and
/// `2π` for the axisymmetric family. Without it the high modes come out
/// grossly overdriven, because their shapes are small everywhere but their
/// masses are smaller still.
pub fn modal_norm(m: u32, alpha: f32) -> f32 {
    let j = besselj(m + 1, alpha);
    let angular = if m == 0 {
        core::f32::consts::TAU
    } else {
        core::f32::consts::PI
    };
    0.5 * j * j * angular
}

/// The net volume a mode sweeps, per unit amplitude, divided by `a²`.
///
/// This is the whole reason the two heads of a drum are coupled at all, and
/// the reason the coupling is *selective*. For `m ≥ 1` the shape carries
/// `cos(mθ)`, whose integral around the head is exactly zero: those modes
/// push as much air out on one side as they pull in on the other, sweep no
/// net volume, and therefore do not talk to the cavity at all. Only the
/// axisymmetric `(0, n)` family breathes:
///
/// ```text
/// ∫∫ J₀(α r/a) dA = 2π a² J₁(α)/α
/// ```
///
/// So a tom's batter and resonant heads are coupled through `(0,1)`, `(0,2)`
/// and their few relatives, and through nothing else. Every ringing, pitched
/// mode of the head is on its own.
pub fn volume_displacement(m: u32, alpha: f32) -> f32 {
    if m != 0 {
        return 0.0;
    }
    core::f32::consts::TAU * besselj(1, alpha) / alpha
}

/// Splits one breathing mode into the pair two coupled heads actually have.
///
/// Batter and resonant head share the enclosed air. Their `(0, n)` modes come
/// in two combinations: the two heads moving the same way in space — one
/// following the other, the cavity's volume nearly unchanged — and the two
/// moving oppositely, squeezing the air between them. The first sees no air
/// spring and stays near the free head's frequency; the second is stiffened
/// by it and rises. Returned as `(lower, upper)`.
///
/// `stiffness` is `2·K_air/ω₀²` — the air spring's share, dimensionless.
///
/// Structurally this is the same problem as the Concert Grand's unison: two
/// oscillators that would be independent, made into a fast pair and a slow
/// pair by one shared termination. The lesson carried over from that work is
/// that the coupling must be *passive*, or configurations appear that gain
/// energy; here that is automatic, because the split is in the frequencies
/// rather than in a feedback path.
pub fn couple_heads(frequency: f32, stiffness: f32) -> (f32, f32) {
    (frequency, frequency * sqrtf(1.0 + stiffness.max(0.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ideal membrane's textbook ratios, against the table.
    #[test]
    fn ideal_ratios_are_the_published_ones() {
        let ratio = |m, n| BESSEL_ZEROS[index_of(m, n)] / ALPHA_11;
        assert!((ratio(0, 1) - 0.628).abs() < 0.002, "{}", ratio(0, 1));
        assert!((ratio(2, 1) - 1.340).abs() < 0.002, "{}", ratio(2, 1));
        assert!((ratio(0, 2) - 1.441).abs() < 0.002, "{}", ratio(0, 2));
        assert!((ratio(3, 1) - 1.665).abs() < 0.002, "{}", ratio(3, 1));
        assert!((ratio(1, 2) - 1.831).abs() < 0.002, "{}", ratio(1, 2));
        assert!((ratio(4, 1) - 1.980).abs() < 0.002, "{}", ratio(4, 1));
    }

    /// Air loading fitted on ONE ratio predicts the other two.
    ///
    /// β is solved from (2,1) landing on Rossing's 1.5 and nothing else. If
    /// (3,1) and (4,1) then land on 2 and 2.5, the `ρ_air/k` law is carrying
    /// real physics; if they did not, β would be a knob with a story attached.
    #[test]
    fn air_loaded_ratios_match_the_kettledrum() {
        const BETA: f32 = 3.8;
        let ratio = |m, n| air_loaded_ratio(BESSEL_ZEROS[index_of(m, n)], BETA);
        assert!((ratio(1, 1) - 1.0).abs() < 1e-5, "{}", ratio(1, 1));
        // Fitted:
        assert!((ratio(2, 1) - 1.5).abs() < 0.02, "(2,1) = {}", ratio(2, 1));
        // Predicted:
        assert!((ratio(3, 1) - 2.0).abs() < 0.04, "(3,1) = {}", ratio(3, 1));
        assert!((ratio(4, 1) - 2.5).abs() < 0.06, "(4,1) = {}", ratio(4, 1));
        // And the axisymmetric mode is pushed well below the pitch, where a
        // kettledrum's (0,1) is heard as a separate thump rather than as part
        // of the tone.
        assert!(ratio(0, 1) < 0.55, "(0,1) = {}", ratio(0, 1));
    }

    /// No air, no change: the law must reduce to the ideal membrane.
    #[test]
    fn zero_air_load_is_the_ideal_membrane() {
        for (index, &alpha) in BESSEL_ZEROS.iter().enumerate() {
            let loaded = air_loaded_ratio(alpha, 0.0);
            let ideal = alpha / ALPHA_11;
            assert!(
                (loaded - ideal).abs() < 1e-4,
                "index {index}: {loaded} vs {ideal}"
            );
        }
    }

    /// A dead-centre strike excites the axisymmetric family and nothing else.
    #[test]
    fn centre_strike_speaks_only_through_the_breathing_modes() {
        for (index, &alpha) in BESSEL_ZEROS.iter().enumerate() {
            let m = angular_order(index);
            let shape = strike_shape(m, alpha, 0.0, 0.0);
            if m == 0 {
                assert!(shape.abs() > 0.9, "(0,n) at centre = {shape}");
            } else {
                assert!(shape.abs() < 1e-6, "m={m} at centre = {shape}");
            }
        }
    }

    /// Only the breathing modes couple to the cavity. This is the fact the
    /// whole two-head model rests on, so it gets its own test.
    #[test]
    fn only_axisymmetric_modes_move_air() {
        for (index, &alpha) in BESSEL_ZEROS.iter().enumerate() {
            let m = angular_order(index);
            let swept = volume_displacement(m, alpha);
            if m == 0 {
                assert!(swept.abs() > 1e-3, "(0,n) sweeps {swept}");
            } else {
                assert_eq!(swept, 0.0, "m={m} sweeps {swept}");
            }
        }
    }

    /// The air spring stiffens one combination and leaves the other alone.
    #[test]
    fn coupled_heads_split_around_the_free_frequency() {
        let (lower, upper) = couple_heads(100.0, 0.4);
        assert!((lower - 100.0).abs() < 1e-4, "lower {lower}");
        assert!(upper > 117.0 && upper < 119.0, "upper {upper}");
        let (l0, u0) = couple_heads(100.0, 0.0);
        assert!((l0 - u0).abs() < 1e-4, "no cavity, no split");
    }
}
