//! Control-rate float math for a `no_std` component.
//!
//! The audio loop needs none of this — rendering is multiplies and adds by
//! construction. These run at note-on and prepare time only, so they favour
//! clarity over speed, and each states its accuracy.
//!
//! `sincosf`, `expf`, `lnf`, `powf`, `sqrtf` and `roundf` are the Concert
//! Grand's, unchanged: the same component ABI, the same `no_std` constraint,
//! the same stated accuracies. `besselj` is new, and it is the one piece of
//! mathematics a membrane needs that a string did not.

/// `sin` and `cos` together, exact quadrant logic, polynomial on [-π/4, π/4].
/// Absolute error a few parts in 10⁶ for |x| < 100.
pub fn sincosf(x: f32) -> (f32, f32) {
    const FRAC_2_PI: f32 = 0.636_619_77;
    const FRAC_PI_2: f32 = 1.570_796_3;
    let quadrant = roundf(x * FRAC_2_PI);
    let r = x - quadrant * FRAC_PI_2;
    let r2 = r * r;
    let sin = r * (1.0 - r2 / 6.0 * (1.0 - r2 / 20.0 * (1.0 - r2 / 42.0)));
    let cos = 1.0 - r2 / 2.0 * (1.0 - r2 / 12.0 * (1.0 - r2 / 30.0));
    match (quadrant as i32).rem_euclid(4) {
        0 => (sin, cos),
        1 => (cos, -sin),
        2 => (-sin, -cos),
        _ => (-cos, sin),
    }
}

/// `e^x` via exponent split and a degree-6 polynomial on the remainder.
/// Relative error below 1e-6 across the range this model uses (|x| < 30).
pub fn expf(x: f32) -> f32 {
    const LN_2: f32 = 0.693_147_2;
    if x < -87.0 {
        return 0.0;
    }
    if x > 88.0 {
        return f32::INFINITY;
    }
    let k = roundf(x / LN_2);
    let r = x - k * LN_2;
    let mut power = 1.0;
    let mut term = 1.0;
    for n in 1..8 {
        term *= r / n as f32;
        power += term;
    }
    scale_by_power_of_two(power, k as i32)
}

/// Natural log from the significand's `atanh` series plus the exponent.
/// Relative error below 1e-6 for normal positive inputs.
pub fn lnf(x: f32) -> f32 {
    const LN_2: f32 = 0.693_147_2;
    const SQRT_2: f32 = 1.414_213_6;
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let bits = x.to_bits();
    let mut exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let mut mantissa = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    if mantissa > SQRT_2 {
        mantissa *= 0.5;
        exponent += 1;
    }
    let t = (mantissa - 1.0) / (mantissa + 1.0);
    let t2 = t * t;
    let series =
        t * (2.0 + t2 * (2.0 / 3.0 + t2 * (2.0 / 5.0 + t2 * (2.0 / 7.0 + t2 * (2.0 / 9.0)))));
    series + exponent as f32 * LN_2
}

/// `x^y` for positive `x`, through `exp(y·ln x)`.
pub fn powf(x: f32, y: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    expf(y * lnf(x))
}

/// Newton's method seeded from the bit pattern; three refinements give
/// relative error below 1e-7 for normal positive inputs.
pub fn sqrtf(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut y = f32::from_bits((x.to_bits() >> 1) + 0x1fbd_1df5);
    for _ in 0..3 {
        y = 0.5 * (y + x / y);
    }
    y
}

pub fn roundf(x: f32) -> f32 {
    let truncated = x as i32 as f32;
    let fraction = x - truncated;
    if fraction > 0.5 {
        truncated + 1.0
    } else if fraction < -0.5 {
        truncated - 1.0
    } else {
        truncated
    }
}

fn scale_by_power_of_two(value: f32, power: i32) -> f32 {
    let exponent = power.clamp(-126, 127) + 127;
    value * f32::from_bits((exponent as u32) << 23)
}

/// Bessel function of the first kind, `J_m(x)`, for `m >= 0` and `x >= 0`.
///
/// This is the membrane's shape function: mode (m, n) of a circular head is
/// `J_m(α_mn·r/a)·cos(mθ)`, so every strike position in the model is a call
/// to this — the exact analogue of the string's `sin(nπx₀)`, and the reason
/// a drum's overtones are not a harmonic series.
///
/// **The power series is unusable here and that is worth stating.** The
/// obvious `Σ (-1)^k (x/2)^(2k+m) / (k!(k+m)!)` is mathematically correct and
/// numerically hopeless in f32 at the arguments this model uses: the model's
/// highest mode has α ≈ 26, where the series' largest term is ~10⁸ while the
/// sum it converges to is ~0.15. Nine digits cancel; f32 carries seven. The
/// result is noise, not a Bessel function.
///
/// So this uses Miller's algorithm — downward recurrence, which is the stable
/// direction. `J_{m-1} = (2m/x)·J_m − J_{m+1}` is run from a starting order
/// well above both `m` and `x` with an arbitrary seed, and the whole ladder is
/// then normalised by the identity `J₀ + 2·(J₂ + J₄ + …) = 1`. Downward
/// recurrence damps the contaminating second solution instead of amplifying
/// it, so the seed's arbitrariness washes out. Verified against published
/// values and against the tabulated zeros in the test below: every zero
/// `α_mn` this model uses evaluates to |J_m| < 2e-5.
pub fn besselj(m: u32, x: f32) -> f32 {
    if x < 0.0 {
        // J_m(-x) = (-1)^m J_m(x); the model never asks, but the identity is
        // cheaper than a wrong answer.
        let value = besselj(m, -x);
        return if m % 2 == 0 { value } else { -value };
    }
    if x == 0.0 {
        return if m == 0 { 1.0 } else { 0.0 };
    }
    // Small arguments: the series is both accurate and cheap here, and Miller
    // needs its start order raised uncomfortably high as x → 0.
    if x < 1.0 {
        let half = 0.5 * x;
        let mut term = 1.0f32;
        for k in 1..=m {
            term *= half / k as f32;
        }
        let mut sum = term;
        let x2 = half * half;
        for k in 1..12 {
            term *= -x2 / (k as f32 * (k + m) as f32);
            sum += term;
        }
        return sum;
    }
    // Miller: start above max(m, x) with room to spare, seed, recur down.
    let start = (m as f32 + x + 24.0) as u32 | 1;
    let mut j_next = 0.0f32; // J_{start+1}
    let mut j_here = 1.0e-24f32; // J_start, arbitrary
    let mut wanted = 0.0f32;
    // Normalisation: J₀ + 2·Σ J_{2k} = 1, accumulated as the ladder descends.
    let mut normaliser = 0.0f32;
    let mut order = start;
    while order > 0 {
        let j_prev = (2.0 * order as f32 / x) * j_here - j_next;
        j_next = j_here;
        j_here = j_prev;
        order -= 1;
        // `j_here` now holds order `order`.
        if order == m {
            wanted = j_here;
        }
        if order != 0 && order % 2 == 0 {
            normaliser += 2.0 * j_here;
        }
        // Downward recurrence grows without bound for orders below x; rescale
        // the whole state before f32 overflows, which for a 26-order ladder
        // it otherwise does.
        if j_here.abs() > 1.0e20 {
            const SHRINK: f32 = 1.0e-20;
            j_here *= SHRINK;
            j_next *= SHRINK;
            wanted *= SHRINK;
            normaliser *= SHRINK;
        }
    }
    normaliser += j_here; // J₀
    if m == 0 {
        wanted = j_here;
    }
    wanted / normaliser
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_std_within_stated_accuracy() {
        for i in 0..2000 {
            let x = (i as f32 - 1000.0) * 0.01;
            let (sin, cos) = sincosf(x);
            assert!((sin - x.sin()).abs() < 5e-6, "sin({x})");
            assert!((cos - x.cos()).abs() < 5e-6, "cos({x})");
        }
        for i in 1..2000 {
            let x = i as f32 * 0.37;
            assert!((lnf(x) - x.ln()).abs() < 2e-6 * (1.0 + x.ln().abs()), "ln({x})");
            assert!((sqrtf(x) - x.sqrt()).abs() / x.sqrt() < 1e-6, "sqrt({x})");
        }
    }

    /// Published values (Abramowitz & Stegun, table 9.1).
    #[test]
    fn bessel_matches_published_values() {
        let cases: [(u32, f32, f32); 10] = [
            (0, 0.0, 1.0),
            (0, 1.0, 0.765_197_7),
            (0, 2.0, 0.223_890_78),
            (0, 5.0, -0.177_596_77),
            (0, 10.0, -0.245_935_76),
            (1, 1.0, 0.440_050_59),
            (1, 5.0, -0.327_579_14),
            (1, 10.0, 0.043_472_75),
            (2, 5.0, 0.046_565_12),
            (3, 10.0, 0.058_379_37),
        ];
        for (m, x, expected) in cases {
            let got = besselj(m, x);
            assert!(
                (got - expected).abs() < 2e-5,
                "J_{m}({x}) = {got}, expected {expected}"
            );
        }
    }

    /// The tabulated zeros must actually be zeros of the function this model
    /// evaluates. If the table and `besselj` ever disagree, every strike
    /// position in the instrument is wrong and nothing else would say so.
    #[test]
    fn tabulated_zeros_are_zeros() {
        for (index, &alpha) in crate::membrane::BESSEL_ZEROS.iter().enumerate() {
            let m = (index / crate::membrane::RADIAL_ORDERS) as u32;
            let value = besselj(m, alpha);
            assert!(
                value.abs() < 2e-5,
                "J_{m}({alpha}) = {value}, should be zero"
            );
        }
    }
}
