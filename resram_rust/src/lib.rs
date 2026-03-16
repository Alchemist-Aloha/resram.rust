use ndarray::prelude::*;
use num_complex::Complex64;
use numpy::{PyArray1, PyArray2, IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::f64::consts::PI;
use rayon::prelude::*;

fn calculate_cross_sections(
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
    dt: f64,
) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let th_len = th.len();
    let el_len = el.len();
    let wg_len = wg.len();

    // 1. Calculate g(t)
    let g_t: Vec<Complex64> = th.iter().map(|&t| {
        let term1 = (d / l).powi(2) * (l * t - 1.0 + (-l * t).exp());
        let term2 = (beta * d.powi(2)) / (2.0 * l) * (1.0 - (-l * t).exp());
        Complex64::new(term1, term2)
    }).collect();

    // 2. Calculate A(t)
    let a_t: Vec<Complex64> = th.iter().map(|&t| {
        let mut sum_k = Complex64::new(0.0, 0.0);
        for i in 0..wg_len {
            let k_k = (1.0 + eta[i]) * s_factors[i] * (1.0 - Complex64::from_polar(1.0, -wg[i] * t))
                    + eta[i] * s_factors[i] * (1.0 - Complex64::from_polar(1.0, wg[i] * t));
            sum_k += k_k;
        }
        Complex64::from_polar(m.powi(2), 0.0) * (-sum_k).exp()
    }).collect();

    // 3. Kernels and Integration
    let mut integ_a = Array1::<f64>::zeros(el_len);
    let mut integ_f = Array1::<f64>::zeros(el_len);

    for i in 0..el_len {
        let mut sum_a = Complex64::new(0.0, 0.0);
        let mut sum_f = Complex64::new(0.0, 0.0);
        for j in 0..th_len {
            let phase = (el[i] - e0) * th[j];
            let exp_phase = Complex64::from_polar(1.0, phase);
            let val_a = exp_phase * (-g_t[j]).exp() * a_t[j];
            let val_f = exp_phase * g_t[j].conj().exp() * a_t[j].conj();
            let weight = if j == 0 || j == th_len - 1 { 0.5 } else { 1.0 };
            sum_a += val_a * weight;
            sum_f += val_f * weight;
        }
        integ_a[i] = (sum_a * dt).re;
        integ_f[i] = (sum_f * dt).re;
    }

    // 4. Inhomogeneous broadening H
    let h: Array1<f64> = if theta == 0.0 {
        Array1::ones(e0_range.len())
    } else {
        e0_range.mapv(|val| {
            (1.0 / (theta * (2.0 * PI).sqrt())) * (-val.powi(2) / (2.0 * theta.powi(2))).exp()
        })
    };
    let h_sum: f64 = h.sum();

    // 5. Convolution
    fn convolve_valid(input: &Array1<f64>, kernel: &Array1<f64>) -> Array1<f64> {
        let n = input.len();
        let m = kernel.len();
        if n < m { return Array1::zeros(0); }
        let out_len = n - m + 1;
        let mut output = Array1::zeros(out_len);
        for i in 0..out_len {
            let mut sum = 0.0;
            for j in 0..m {
                sum += input[i + j] * kernel[m - 1 - j];
            }
            output[i] = sum;
        }
        output
    }

    let abs_cross = convolve_valid(&integ_a, &h) / h_sum * pre_a * conv_el;
    let fl_cross = convolve_valid(&integ_f, &h) / h_sum * pre_f * conv_el;

    // 6. Raman
    let raman_cross_vec: Vec<Array1<f64>> = (0..wg_len).into_par_iter().map(|idx| {
        let mut integ_r = Array1::<f64>::zeros(el_len);
        let sqrt2 = 2.0f64.sqrt();
        let factor = ((1.0 + eta[idx]).sqrt() * delta[idx]) / sqrt2;
        for i in 0..el_len {
            let mut sum_r = Complex64::new(0.0, 0.0);
            for j in 0..th_len {
                let phase = (el[i] - e0) * th[j];
                let exp_phase = Complex64::from_polar(1.0, phase);
                let q_r = factor * (1.0 - Complex64::from_polar(1.0, -wg[idx] * th[j]));
                let val_r = exp_phase * (-g_t[j]).exp() * a_t[j] * q_r;
                let weight = if j == 0 || j == th_len - 1 { 0.5 } else { 1.0 };
                sum_r += val_r * weight;
            }
            integ_r[i] = (sum_r * dt).norm_sqr();
        }
        let conv_r = convolve_valid(&integ_r, &h) / h_sum;
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

fn corrcoef(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
    let n = x.len() as f64;
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
    num / (den_x.sqrt() * den_y.sqrt())
}

#[pyfunction]
fn raman_residual_rust(
    wg: PyReadonlyArray1<f64>,
    eta: PyReadonlyArray1<f64>,
    delta: PyReadonlyArray1<f64>,
    th: PyReadonlyArray1<f64>,
    el: PyReadonlyArray1<f64>,
    conv_el: PyReadonlyArray1<f64>,
    e0_range: PyReadonlyArray1<f64>,
    abs_exp: PyReadonlyArray1<f64>,
    profs_exp: PyReadonlyArray2<f64>,
    rp: PyReadonlyArray1<usize>,
    gamma: f64,
    m: f64,
    kappa: f64,
    theta: f64,
    e0: f64,
    beta: f64,
    pre_a: f64,
    pre_f: f64,
    pre_r: f64,
    dt: f64,
) -> PyResult<(f64, f64, f64)> {
    let wg = wg.as_array();
    let eta = eta.as_array();
    let delta = delta.as_array();
    let th = th.as_array();
    let el = el.as_array();
    let conv_el = conv_el.as_array();
    let e0_range = e0_range.as_array();
    let abs_exp = abs_exp.as_array();
    let profs_exp = profs_exp.as_array();
    let rp = rp.as_array();

    let s_factors = delta.mapv(|d| d.powi(2) / 2.0);
    let d_param = gamma * (1.0 + 0.85 * kappa + 0.88 * kappa.powi(2)) / (2.355 + 1.76 * kappa);
    let l_param = kappa * d_param;

    let (abs_cross, _, raman_cross) = calculate_cross_sections(
        &wg, &s_factors.view(), &eta, &delta, &th, &el, &conv_el, &e0_range,
        d_param, l_param, beta, e0, m, pre_a, pre_f, pre_r, theta, dt
    );

    let correlation = corrcoef(&abs_cross.view(), &abs_exp);
    
    let mut total_sigma = 0.0;
    for (idx, &mode_rp) in rp.iter().enumerate() {
        // Raman cross is (modes, energies)
        // profs_exp is (modes, pumps)
        // We need raman_cross[mode, rp[pump]]
        for mode_idx in 0..wg.len() {
            let diff = raman_cross[[mode_idx, mode_rp]] - profs_exp[[mode_idx, idx]];
            total_sigma += 1e7 * diff.powi(2);
        }
    }

    let loss = total_sigma + 30.0 * (1.0 - correlation);
    let abs_mismatch = 100.0 * (1.0 - correlation);

    Ok((loss, total_sigma, abs_mismatch))
}

#[pyfunction]
fn cross_sections_rust(
    py: Python<'_>,
    wg: PyReadonlyArray1<f64>,
    s_factors: PyReadonlyArray1<f64>,
    eta: PyReadonlyArray1<f64>,
    delta: PyReadonlyArray1<f64>,
    th: PyReadonlyArray1<f64>,
    el: PyReadonlyArray1<f64>,
    conv_el: PyReadonlyArray1<f64>,
    e0_range: PyReadonlyArray1<f64>,
    d: f64,
    l: f64,
    beta: f64,
    e0: f64,
    m: f64,
    pre_a: f64,
    pre_f: f64,
    pre_r: f64,
    theta: f64,
    dt: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray2<f64>>)> {
    let (abs_cross, fl_cross, raman_cross) = calculate_cross_sections(
        &wg.as_array(), &s_factors.as_array(), &eta.as_array(), &delta.as_array(),
        &th.as_array(), &el.as_array(), &conv_el.as_array(), &e0_range.as_array(),
        d, l, beta, e0, m, pre_a, pre_f, pre_r, theta, dt
    );

    Ok((
        abs_cross.into_pyarray(py).unbind(),
        fl_cross.into_pyarray(py).unbind(),
        raman_cross.into_pyarray(py).unbind(),
    ))
}

#[pymodule]
fn resram_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cross_sections_rust, m)?)?;
    m.add_function(wrap_pyfunction!(raman_residual_rust, m)?)?;
    Ok(())
}
