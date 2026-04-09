use ndarray::prelude::*;
use num_complex::Complex64;
use std::f64::consts::PI;
use rayon::prelude::*;
use crate::models::{ResRamConfig, VibrationalMode, SimulationResult};

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
    _dt: f64,
) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let th_len = th.len();
    let el_full_len = el.len();
    let wg_len = wg.len();
    
    let d_th = 1.0; // Python np.trapezoid default dx=1.0
    let dt_step = if th_len > 1 { th[1] - th[0] } else { 0.0 };

    // 1. Pre-calculate time-dependent Kernels
    let mut kernels = Vec::with_capacity(th_len);
    let mut kernels_f = Vec::with_capacity(th_len);

    for j in 0..th_len {
        let t = th[j];
        
        let g_real = (d / l).powi(2) * (l * t - 1.0 + (-l * t).exp());
        let g_imag = (beta * d.powi(2)) / (2.0 * l) * (1.0 - (-l * t).exp());
        let g_t = Complex64::new(g_real, g_imag);

        let mut sum_k = Complex64::new(0.0, 0.0);
        for i in 0..wg_len {
            let phase = wg[i] * t;
            let (sin_p, cos_p) = phase.sin_cos();
            let k_k_re = s_factors[i] * (1.0 + 2.0 * eta[i]) * (1.0 - cos_p);
            let k_k_im = s_factors[i] * sin_p;
            sum_k.re += k_k_re;
            sum_k.im += k_k_im;
        }
        let a_t = Complex64::new(m.powi(2), 0.0) * (-sum_k).exp();
        
        let weight = if j == 0 || j == th_len - 1 { 0.5 } else { 1.0 };
        kernels.push(a_t * (-g_t).exp() * weight);
        kernels_f.push(a_t.conj() * (-g_t).conj().exp() * weight);
    }

    // 2. Abs/FL Integrals
    let (integ_a_vec, integ_f_vec): (Vec<f64>, Vec<f64>) = (0..el_full_len).into_par_iter().map(|i| {
        let phase_diff = el[i] - e0;
        let mut sum_a = Complex64::new(0.0, 0.0);
        let mut sum_f = Complex64::new(0.0, 0.0);
        
        let delta_phase = phase_diff * dt_step;
        let (sin_dp, cos_dp) = delta_phase.sin_cos();
        let exp_delta = Complex64::new(cos_dp, sin_dp);
        let mut current_exp = Complex64::new(1.0, 0.0);

        for j in 0..th_len {
            sum_a += current_exp * kernels[j];
            sum_f += current_exp * kernels_f[j];
            
            let next_re = current_exp.re * exp_delta.re - current_exp.im * exp_delta.im;
            let next_im = current_exp.re * exp_delta.im + current_exp.im * exp_delta.re;
            current_exp.re = next_re;
            current_exp.im = next_im;
            
            if (j & 255) == 255 {
                let norm = current_exp.norm();
                current_exp.re /= norm;
                current_exp.im /= norm;
            }
        }
        
        ((sum_a * d_th).re, (sum_f * d_th).re)
    }).unzip();

    // 3. Inhomogeneous Broadening H
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

    // 4. Abs/FL Results
    let abs_conv = convolve_valid(&Array1::from_vec(integ_a_vec), &h) / h_sum;
    let fl_conv = convolve_valid(&Array1::from_vec(integ_f_vec), &h) / h_sum;
    
    let mut abs_cross = Array1::<f64>::zeros(conv_el.len());
    let mut fl_cross = Array1::<f64>::zeros(conv_el.len());
    for i in 0..conv_el.len() {
        abs_cross[i] = abs_conv[i] * pre_a * conv_el[i];
        fl_cross[i] = fl_conv[i] * pre_f * conv_el[i];
    }

    // 5. Raman REPs
    let raman_cross_vec: Vec<Array1<f64>> = (0..wg_len).into_par_iter().map(|idx| {
        let mut integ_r_mag_sq = Array1::<f64>::zeros(el_full_len);
        let factor = ((1.0 + eta[idx]).sqrt() * delta[idx]) / 2.0f64.sqrt();
        
        let mut modified_kernels = Vec::with_capacity(th_len);
        for j in 0..th_len {
            let phase = wg[idx] * th[j];
            let (sin_p, cos_p) = phase.sin_cos();
            let q_r = factor * Complex64::new(1.0 - cos_p, sin_p);
            modified_kernels.push(kernels[j] * q_r);
        }
        
        for i in 0..el_full_len {
            let phase_diff = el[i] - e0;
            let mut sum_r = Complex64::new(0.0, 0.0);
            
            let delta_phase = phase_diff * dt_step;
            let (sin_dp, cos_dp) = delta_phase.sin_cos();
            let exp_delta = Complex64::new(cos_dp, sin_dp);
            let mut current_exp = Complex64::new(1.0, 0.0);

            for j in 0..th_len {
                sum_r += current_exp * modified_kernels[j];
                
                let next_re = current_exp.re * exp_delta.re - current_exp.im * exp_delta.im;
                let next_im = current_exp.re * exp_delta.im + current_exp.im * exp_delta.re;
                current_exp.re = next_re;
                current_exp.im = next_im;
                
                if (j & 255) == 255 {
                    let norm = current_exp.norm();
                    current_exp.re /= norm;
                    current_exp.im /= norm;
                }
            }
            integ_r_mag_sq[i] = (sum_r * d_th).norm_sqr();
        }
        
        let conv_r = convolve_valid(&integ_r_mag_sq, &h) / h_sum;
        let mut final_r = Array1::<f64>::zeros(conv_el.len());
        for k in 0..conv_el.len() {
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

pub fn compute_spectra(
    config: &ResRamConfig,
    modes: &[VibrationalMode],
    rpumps: &[f64],
    th: &ArrayView1<f64>,
    el: &ArrayView1<f64>,
    conv_el: &ArrayView1<f64>,
    e0_range: &ArrayView1<f64>,
) -> SimulationResult {
    let wg: Array1<f64> = modes.iter().map(|m| m.frequency).collect();
    let delta: Array1<f64> = modes.iter().map(|m| m.displacement).collect();
    let s_factors = delta.mapv(|d| d.powi(2) / 2.0);
    
    let k_b_t = 0.695 * config.temp;
    let beta = if config.temp > 0.1 { 1.0 / k_b_t } else { 1e10 };
    let eta: Array1<f64> = if config.temp > 0.1 {
        wg.mapv(|w| 1.0 / ((w / k_b_t).exp() - 1.0))
    } else {
        Array1::zeros(wg.len())
    };

    let d_param = config.gamma * (1.0 + 0.85 * config.kappa + 0.88 * config.kappa.powi(2)) / (2.355 + 1.76 * config.kappa);
    let l_param = config.kappa * d_param;

    let n = config.n;
    let ts = config.time_step;
    let pre_a = ((5.744e-3) / n) * ts;
    let pre_f = pre_a * n.powi(2);
    let pre_r = 2.08e-20 * ts.powi(2);

    let (abs_cross, mut fl_cross, raman_cross) = calculate_cross_sections(
        &wg.view(), &s_factors.view(), &eta.view(), &delta.view(),
        th, el, conv_el, e0_range,
        d_param, l_param, beta, config.e0, config.m,
        pre_a, pre_f, pre_r, config.theta, ts
    );

    // Apply FL w3 correction: fl * convEL^2 / E0^2
    for i in 0..conv_el.len() {
        fl_cross[i] = fl_cross[i] * (conv_el[i].powi(2) / config.e0.powi(2));
    }

    // Generate Raman Shift axis
    let rshift: Vec<f64> = (0..)
        .map(|i| config.raman_start + (i as f64) * config.raman_step)
        .take_while(|&v| v < config.raman_end)
        .collect();
    let rshift_arr = Array1::from_vec(rshift.clone());

    // Generate Raman Spectra for each pump
    let mut raman_spec = Vec::new();
    let mut rp_indices = Vec::new();
    for &pump in rpumps {
        // Find index of pump in conv_el
        let mut min_diff = f64::MAX;
        let mut pump_idx = 0;
        for (i, &e) in conv_el.iter().enumerate() {
            let diff = (e - pump).abs();
            if diff < min_diff {
                min_diff = diff;
                pump_idx = i;
            }
        }
        rp_indices.push(pump_idx);

        let mut spec = Array1::zeros(rshift_arr.len());
        for j in 0..wg.len() {
            let intensity = raman_cross[[j, pump_idx]];
            let mode_freq = wg[j];
            let res = config.raman_res;
            
            // Lorentzian: (1/PI) * (0.5*res) / ((rshift - wg)^2 + (0.5*res)^2)
            let lor = rshift_arr.mapv(|rs| {
                (1.0 / PI) * (0.5 * res) / ((rs - mode_freq).powi(2) + (0.5 * res).powi(2))
            });
            spec += &(lor * intensity);
        }
        raman_spec.push(spec.to_vec());
    }

    SimulationResult {
        abs_cross: abs_cross.to_vec(),
        fl_cross: fl_cross.to_vec(),
        abs_exp: None,
        fl_exp: None,
        profs_exp: None,
        raman_cross: raman_cross.rows().into_iter().map(|r| r.to_vec()).collect(),
        raman_spec,
        rp_indices,
        conv_el: conv_el.to_vec(),
        rshift,
    }
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
    let den = den_x.sqrt() * den_y.sqrt();
    if den == 0.0 { return 0.0; }
    num / den
}
