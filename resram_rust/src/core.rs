use ndarray::prelude::*;
use num_complex::Complex64;
use std::f64::consts::PI;
use rayon::prelude::*;

/// Implements a "valid" convolution similar to np.convolve(a, v, 'valid')
pub fn convolve_valid(input: &Array1<f64>, kernel: &Array1<f64>) -> Array1<f64> {
    let (a, v) = if input.len() >= kernel.len() {
        (input, kernel)
    } else {
        (kernel, input)
    };
    let n = a.len();
    let m = v.len();
    let out_len = n - m + 1;
    let mut output = Array1::zeros(out_len);
    for i in 0..out_len {
        let mut sum = 0.0;
        for j in 0..m {
            // v[m-1-j] effectively reverses the kernel, matching np.convolve
            sum += a[i + j] * v[m - 1 - j];
        }
        output[i] = sum;
    }
    output
}

pub fn calculate_cross_sections(
    wg: &ArrayView1<f64>,
    s_factors: &ArrayView1<f64>,
    eta: &ArrayView1<f64>,
    delta: &ArrayView1<f64>,
    th: &ArrayView1<f64>,
    el: &ArrayView1<f64>,
    conv_el: &ArrayView1<f64>,
    e0_range: &ArrayView1<f64>,
    d: f64,
    l: f64,
    beta: f64,
    e0: f64,
    m: f64,
    pre_a: f64,
    pre_f: f64,
    pre_r: f64,
    theta: f64,
    _dt_unused: f64, // Spacing is calculated from 'th' to ensure consistency with np.trapz
) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let th_len = th.len();
    let el_full_len = el.len();
    let wg_len = wg.len();
    
    // Spacing for trapezoidal integration
    // Python's np.trapezoid defaults to dx=1.0 when x is not provided
    let d_th = 1.0;

    // 1. Pre-calculate time-dependent Kernels (Trapezoidal weighted)
    // kernel = A(t) * exp(-g(t)) * weight
    let mut kernels = Vec::with_capacity(th_len);
    let mut kernels_f = Vec::with_capacity(th_len);

    for j in 0..th_len {
        let t = th[j];
        
        // g(t)
        let g_real = (d / l).powi(2) * (l * t - 1.0 + (-l * t).exp());
        let g_imag = (beta * d.powi(2)) / (2.0 * l) * (1.0 - (-l * t).exp());
        let g_t = Complex64::new(g_real, g_imag);

        // A(t)
        let mut sum_k = Complex64::new(0.0, 0.0);
        for i in 0..wg_len {
            let k_k = (1.0 + eta[i]) * s_factors[i] * (1.0 - Complex64::from_polar(1.0, -wg[i] * t))
                    + eta[i] * s_factors[i] * (1.0 - Complex64::from_polar(1.0, wg[i] * t));
            sum_k += k_k;
        }
        let a_t = Complex64::new(m.powi(2), 0.0) * (-sum_k).exp();
        
        let weight = if j == 0 || j == th_len - 1 { 0.5 } else { 1.0 };
        kernels.push(a_t * (-g_t).exp() * weight);
        kernels_f.push(a_t.conj() * (-g_t).conj().exp() * weight);
    }

    // 2. Absorption and Fluorescence Integrals (over full EL grid)
    let (integ_a_vec, integ_f_vec): (Vec<f64>, Vec<f64>) = (0..el_full_len).into_par_iter().map(|i| {
        let energy = el[i];
        let mut sum_a = Complex64::new(0.0, 0.0);
        let mut sum_f = Complex64::new(0.0, 0.0);
        
        for j in 0..th_len {
            let phase = (energy - e0) * th[j];
            let exp_phase = Complex64::from_polar(1.0, phase);
            sum_a += exp_phase * kernels[j];
            sum_f += exp_phase * kernels_f[j];
        }
        
        ((sum_a * d_th).re, (sum_f * d_th).re)
    }).unzip();

    // 3. Inhomogeneous Broadening Kernel H
    let h: Array1<f64> = if theta < 1e-10 {
        let mut h_delta = Array1::from_elem(e0_range.len(), 0.0);
        h_delta[e0_range.len() / 2] = 1.0;
        h_delta
    } else {
        e0_range.mapv(|val| {
            (1.0 / (theta * (2.0 * PI).sqrt())) * (-val.powi(2) / (2.0 * theta.powi(2))).exp()
        })
    };
    let h_sum: f64 = h.sum();

    // 4. Convolution and Final Scaling for Abs/Fl
    let abs_conv = convolve_valid(&Array1::from_vec(integ_a_vec), &h) / h_sum;
    let fl_conv = convolve_valid(&Array1::from_vec(integ_f_vec), &h) / h_sum;
    
    let mut abs_cross = Array1::<f64>::zeros(conv_el.len());
    let mut fl_cross = Array1::<f64>::zeros(conv_el.len());
    for i in 0..conv_el.len() {
        abs_cross[i] = abs_conv[i] * pre_a * conv_el[i];
        fl_cross[i] = fl_conv[i] * pre_f * conv_el[i];
    }

    // 5. Raman Cross Sections (Parallelized over modes)
    let raman_cross_vec: Vec<Array1<f64>> = (0..wg_len).into_par_iter().map(|idx| {
        let mut integ_r_mag_sq = Array1::<f64>::zeros(el_full_len);
        let factor = ((1.0 + eta[idx]).sqrt() * delta[idx]) / 2.0f64.sqrt();
        
        for i in 0..el_full_len {
            let energy = el[i];
            let mut sum_r = Complex64::new(0.0, 0.0);
            for j in 0..th_len {
                let phase = (energy - e0) * th[j];
                let exp_phase = Complex64::from_polar(1.0, phase);
                let q_r = factor * (1.0 - Complex64::from_polar(1.0, -wg[idx] * th[j]));
                sum_r += exp_phase * kernels[j] * q_r;
            }
            // Magnitude squared BEFORE convolution (matching Python)
            integ_r_mag_sq[i] = (sum_r * d_th).norm_sqr();
        }
        
        let conv_r = convolve_valid(&integ_r_mag_sq, &h) / h_sum;
        let mut final_r = Array1::<f64>::zeros(conv_el.len());
        for k in 0..conv_el.len() {
            // obj.preR * convEL * (convEL - wg)**3 * conv_result
            final_r[k] = pre_r * conv_el[k] * (conv_el[k] - wg[idx]).powi(3) * conv_r[k];
        }
        final_r
    }).collect();

    let mut raman_cross = Array2::<f64>::zeros((wg_len, conv_el.len()));
    for i in 0..wg_len {
        raman_cross.row_mut(i).assign(&raman_cross_vec[i]);
    }

    (abs_cross, fl_cross, raman_cross)
}

pub fn corrcoef(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 { return 1.0; }
    let mean_x = x.sum() / n;
    let mean_y = y.sum() / n;
    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    let den = (den_x.sqrt() * den_y.sqrt());
    if den == 0.0 { return 0.0; }
    num / den
}
