pub mod models;
pub mod core;
pub mod optimizer;
pub mod config;

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::prelude::*;
use crate::core::calculate_cross_sections;

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

    let correlation = crate::core::corrcoef(&abs_cross.view(), &abs_exp);
    
    let mut total_sigma = 0.0;
    for (idx, &mode_rp) in rp.iter().enumerate() {
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
