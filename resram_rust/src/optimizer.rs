use nlopt::*;
use ndarray::prelude::*;
use crate::models::{ResRamConfig, VibrationalMode};
use crate::core::{calculate_cross_sections, corrcoef};
use std::sync::Arc;

pub type ProgressCallback = Arc<dyn Fn(u32, f64, &[f64]) + Send + Sync>;

pub struct OptimizationContext {
    pub config: ResRamConfig,
    pub modes: Vec<VibrationalMode>,
    pub th: Array1<f64>,
    pub el: Array1<f64>,
    pub conv_el: Array1<f64>,
    pub e0_range: Array1<f64>,
    pub abs_exp: Array1<f64>,
    pub profs_exp: Array2<f64>,
    pub rp: Vec<usize>,
    pub pre_a: f64,
    pub pre_f: f64,
    pub pre_r: f64,
    pub beta: f64,
    pub dt: f64,
    pub fit_indices: Vec<usize>, 
    pub fit_gamma: bool,
    pub fit_m: bool,
    pub fit_theta: bool,
    pub fit_kappa: bool,
    pub fit_e0: bool,
    pub iteration: u32,
    pub progress_callback: Option<ProgressCallback>,
}

fn objective_function(x: &[f64], _gradient: Option<&mut [f64]>, context: &mut OptimizationContext) -> f64 {
    let mut current_config = context.config.clone();
    let mut current_modes = context.modes.clone();
    
    let mut cursor = 0;
    for &idx in &context.fit_indices {
        current_modes[idx].displacement = x[cursor];
        cursor += 1;
    }
    
    if context.fit_gamma {
        current_config.gamma = x[cursor];
        cursor += 1;
    }
    
    if context.fit_m {
        current_config.m = x[cursor];
        cursor += 1;
    }

    if context.fit_theta {
        current_config.theta = x[cursor];
        cursor += 1;
    }

    if context.fit_kappa {
        current_config.kappa = x[cursor];
        cursor += 1;
    }

    if context.fit_e0 {
        current_config.e0 = x[cursor];
        cursor += 1;
    }

    // Calculate loss
    let wg: Array1<f64> = current_modes.iter().map(|m| m.frequency).collect();
    let delta: Array1<f64> = current_modes.iter().map(|m| m.displacement).collect();
    let s_factors = delta.mapv(|d| d.powi(2) / 2.0);
    
    let k_b_t = 0.695 * current_config.temp;
    let eta: Array1<f64> = if current_config.temp > 0.1 {
        wg.mapv(|w| 1.0 / ((w / k_b_t).exp() - 1.0))
    } else {
        Array1::zeros(wg.len())
    };

    let d_param = current_config.gamma * (1.0 + 0.85 * current_config.kappa + 0.88 * current_config.kappa.powi(2)) / (2.355 + 1.76 * current_config.kappa);
    let l_param = current_config.kappa * d_param;

    let (abs_cross, _, raman_cross) = calculate_cross_sections(
        &wg.view(), &s_factors.view(), &eta.view(), &delta.view(), 
        &context.th.view(), &context.el.view(), &context.conv_el.view(), &context.e0_range.view(),
        d_param, l_param, context.beta, current_config.e0, current_config.m, 
        context.pre_a, context.pre_f, context.pre_r, current_config.theta, context.dt
    );

    let correlation = corrcoef(&abs_cross.view(), &context.abs_exp.view());
    
    let mut total_sigma = 0.0;
    for (idx, &mode_rp) in context.rp.iter().enumerate() {
        for mode_idx in 0..wg.len() {
            let diff = raman_cross[[mode_idx, mode_rp]] - context.profs_exp[[mode_idx, idx]];
            total_sigma += 1e7 * diff.powi(2);
        }
    }

    let loss = total_sigma + 30.0 * (1.0 - correlation);
    
    context.iteration += 1;
    if let Some(ref cb) = context.progress_callback {
        cb(context.iteration, loss, x);
    }
    
    loss
}

pub fn run_optimization(
    algorithm: Algorithm,
    context: OptimizationContext,
    max_eval: u32,
    ftol_rel: f64,
) -> Result<(Vec<f64>, f64), String> {
    let mut initial_x = Vec::new();
    let mut lower_bounds = Vec::new();
    let mut upper_bounds = Vec::new();

    for &idx in &context.fit_indices {
        initial_x.push(context.modes[idx].displacement);
        lower_bounds.push(0.0);
        upper_bounds.push(1.0);
    }

    if context.fit_gamma {
        let val = context.config.gamma.max(10.0).min(1000.0);
        initial_x.push(val);
        lower_bounds.push(10.0);
        upper_bounds.push(1000.0);
    }

    if context.fit_m {
        initial_x.push(context.config.m);
        lower_bounds.push(0.8 * context.config.m);
        upper_bounds.push(1.2 * context.config.m);
    }

    if context.fit_theta {
        let val = context.config.theta.max(0.0).min(10.0);
        initial_x.push(val);
        lower_bounds.push(0.0);
        upper_bounds.push(10.0);
    }

    if context.fit_kappa {
        let val = context.config.kappa.max(0.0).min(1.0);
        initial_x.push(val);
        lower_bounds.push(0.0);
        upper_bounds.push(1.0);
    }

    if context.fit_e0 {
        initial_x.push(context.config.e0);
        lower_bounds.push(0.95 * context.config.e0);
        upper_bounds.push(1.05 * context.config.e0);
    }

    let n_params = initial_x.len();
    if n_params == 0 {
        return Err("No parameters selected for fitting".into());
    }

    let mut opt = Nlopt::new(algorithm, n_params, objective_function, Target::Minimize, context);
    
    opt.set_lower_bounds(&lower_bounds).map_err(|e| format!("Failed to set lower bounds: {:?}", e))?;
    opt.set_upper_bounds(&upper_bounds).map_err(|e| format!("Failed to set upper bounds: {:?}", e))?;
    opt.set_ftol_rel(ftol_rel).map_err(|e| format!("Failed to set ftol_rel: {:?}", e))?;
    opt.set_maxeval(max_eval).map_err(|e| format!("Failed to set maxeval: {:?}", e))?;

    let mut x = initial_x;
    match opt.optimize(&mut x) {
        Ok((_, loss)) => Ok((x, loss)),
        Err((status, _)) => Err(format!("Optimization failed with status {:?}", status)),
    }
}
