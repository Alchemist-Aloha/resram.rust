use resram_rust::config::{load_config, load_vibrational_data, migrate_txt_to_toml, save_vibrational_data, write_config_txt, write_config_toml, load_fit_config, save_fit_config};
use resram_rust::models::{ResRamConfig, VibrationalMode, SimulationResult, FitConfig};
use resram_rust::core::compute_spectra;
use resram_rust::optimizer::{run_optimization, OptimizationContext};
use std::path::Path;
use serde::Serialize;
use ndarray::prelude::*;
use std::sync::Arc;
use tauri::{Emitter, Runtime};

fn parse_spectrum_xy(path: &Path) -> Result<(Vec<f64>, Vec<f64>), String> {
    let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut x = Vec::new();
    let mut y = Vec::new();

    for line in content.lines() {
        let nums: Vec<f64> = line
            .split_whitespace()
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();
        if nums.len() >= 2 {
            x.push(nums[0]);
            y.push(nums[1]);
        } else if nums.len() == 1 {
            y.push(nums[0]);
        }
    }

    if !x.is_empty() {
        if x.len() != y.len() {
            return Err(format!("Invalid spectrum format in {}", path.display()));
        }
        if x.first().unwrap_or(&0.0) > x.last().unwrap_or(&0.0) {
            x.reverse();
            y.reverse();
        }
        Ok((x, y))
    } else {
        // 1-column fallback assumes evenly spaced points on current convEL.
        Ok((Vec::new(), y))
    }
}

fn interp_linear(x: &[f64], y: &[f64], x_new: &Array1<f64>) -> Vec<f64> {
    if x.is_empty() {
        if y.len() == x_new.len() {
            return y.to_vec();
        }
        return vec![0.0; x_new.len()];
    }
    if x.len() == 1 {
        return vec![y[0]; x_new.len()];
    }

    let mut out = Vec::with_capacity(x_new.len());
    for &xn in x_new {
        if xn <= x[0] {
            out.push(y[0]);
            continue;
        }
        if xn >= x[x.len() - 1] {
            out.push(y[y.len() - 1]);
            continue;
        }

        let mut lo = 0usize;
        let mut hi = x.len() - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if x[mid] <= xn {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let x0 = x[lo];
        let x1 = x[hi];
        let y0 = y[lo];
        let y1 = y[hi];
        let t = if x1 != x0 { (xn - x0) / (x1 - x0) } else { 0.0 };
        out.push(y0 + t * (y1 - y0));
    }
    out
}

fn load_exp_interp(path: &Path, conv_el: &Array1<f64>) -> Result<Option<Vec<f64>>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let (x, y) = parse_spectrum_xy(path)?;
    if y.is_empty() {
        return Ok(None);
    }
    Ok(Some(interp_linear(&x, &y, conv_el)))
}

fn load_exp_matrix(path: &Path) -> Result<Option<Vec<Vec<f64>>>, String> {
    if !path.exists() {
        return Ok(None);
    }

    let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let rows: Vec<Vec<f64>> = content
        .lines()
        .map(|l| l.split_whitespace().filter_map(|s| s.parse::<f64>().ok()).collect())
        .filter(|r: &Vec<f64>| !r.is_empty())
        .collect();

    if rows.is_empty() {
        Ok(None)
    } else {
        Ok(Some(rows))
    }
}

fn parse_algorithm(name: &str) -> nlopt::Algorithm {
    match name.to_lowercase().as_str() {
        "powell" | "praxis" => nlopt::Algorithm::Praxis,
        "cobyla" => nlopt::Algorithm::Cobyla,
        "bobyqa" => nlopt::Algorithm::Bobyqa,
        "newuoa" => nlopt::Algorithm::Newuoa,
        "newuoa_bound" | "newuoabound" => nlopt::Algorithm::NewuoaBound,
        "neldermead" | "nelder-mead" => nlopt::Algorithm::Neldermead,
        "sbplx" | "subplex" => nlopt::Algorithm::Sbplx,
        _ => nlopt::Algorithm::Praxis,
    }
}

fn load_scalar_list(path: &Path) -> Result<Vec<f64>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    Ok(content
        .lines()
        .filter_map(|l| l.split_whitespace().next().and_then(|s| s.parse::<f64>().ok()))
        .collect())
}

fn write_vec(path: &Path, values: &[f64]) -> Result<(), String> {
    let content = values
        .iter()
        .map(|v| format!("{:.12e}", v))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(path, content).map_err(|e| e.to_string())
}

fn write_matrix(path: &Path, rows: &[Vec<f64>]) -> Result<(), String> {
    let content = rows
        .iter()
        .map(|row| row.iter().map(|v| format!("{:.12e}", v)).collect::<Vec<_>>().join("\t"))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(path, content).map_err(|e| e.to_string())
}

fn transpose_matrix(input: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if input.is_empty() || input[0].is_empty() {
        return Vec::new();
    }
    let rows = input.len();
    let cols = input[0].len();
    let mut out = vec![vec![0.0; rows]; cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c][r] = input[r][c];
        }
    }
    out
}

#[derive(Clone, Serialize)]
struct ProgressPayload {
    iteration: u32,
    loss: f64,
    parameters: Vec<f64>,
}

#[derive(Clone, Serialize)]
struct FitResultPayload {
    config: ResRamConfig,
    modes: Vec<VibrationalMode>,
    folder_name: String,
}

fn save_data_impl(
    dir: &str, 
    config: &ResRamConfig, 
    modes: &[VibrationalMode],
    fit_config: &FitConfig
) -> Result<String, String> {
    use std::fs;
    use chrono::Local;

    let root = Path::new(dir);
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let folder_name = format!("data_{}", timestamp);
    let output_dir = root.join(&folder_name);

    fs::create_dir_all(&output_dir).map_err(|e| format!("Failed to create directory: {}", e))?;

    // Save updated files
    let toml_path = output_dir.join("inp.toml");
    write_config_toml(&toml_path, config).map_err(|e| e.to_string())?;

    let txt_path = output_dir.join("inp.txt");
    write_config_txt(&txt_path, config).map_err(|e| e.to_string())?;

    save_vibrational_data(&output_dir, modes).map_err(|e| e.to_string())?;
    
    // Save fit.toml if requested or present in state
    save_fit_config(output_dir.join("fit.toml"), fit_config).map_err(|e| e.to_string())?;

    // Save calculated outputs to match Python run_save style.
    let el_reach = config.el_reach;
    let e0 = config.e0;
    let n_time = config.n_time;
    let ts = config.time_step;
    let th = Array1::linspace(0.0, (n_time as f64) * ts / 5.3088, n_time);
    let el = Array1::linspace(e0 - el_reach, e0 + el_reach, 1000);
    let e0_range = Array1::linspace(-el_reach * 0.5, el_reach * 0.5, 501);
    let out_len = el.len().max(e0_range.len()) - el.len().min(e0_range.len()) + 1;
    let conv_el = Array1::linspace(e0 - el_reach * 0.5, e0 + el_reach * 0.5, out_len);

    let rpumps = load_scalar_list(&root.join("rpumps.dat"))?;
    let result = compute_spectra(
        config,
        modes,
        &rpumps,
        &th.view(),
        &el.view(),
        &conv_el.view(),
        &e0_range.view(),
    );

    write_vec(&output_dir.join("Abs.dat"), &result.abs_cross)?;
    write_vec(&output_dir.join("Fl.dat"), &result.fl_cross)?;
    write_vec(&output_dir.join("EL.dat"), &result.conv_el)?;
    write_vec(&output_dir.join("rshift.dat"), &result.rshift)?;

    // Python saves profs.dat as transpose(raman_cross): [energy][mode]
    let profs = transpose_matrix(&result.raman_cross);
    write_matrix(&output_dir.join("profs.dat"), &profs)?;

    if !result.raman_spec.is_empty() {
        // Python saves raman_spec.dat as [rshift][pump]
        let raman_spec_t = transpose_matrix(&result.raman_spec);
        write_matrix(&output_dir.join("raman_spec.dat"), &raman_spec_t)?;
        write_vec(&output_dir.join("rpumps.dat"), &rpumps)?;
    }

    // Save output.toml summary with comments
    let mut summary_content = String::new();
    summary_content.push_str("# ResRAM Output Summary\n");
    summary_content.push_str(&format!("timestamp = \"{}\"\n\n", timestamp));
    summary_content.push_str("[parameters]\n");
    summary_content.push_str(&format!("gamma = {} # homogeneous broadening parameter (cm^-1)\n", config.gamma));
    summary_content.push_str(&format!("theta = {} # static inhomogeneous broadening parameter (cm^-1)\n", config.theta));
    summary_content.push_str(&format!("e0 = {} # 0-0 transition energy (cm^-1)\n", config.e0));
    summary_content.push_str(&format!("kappa = {} # brownian oscillator kappa parameter\n", config.kappa));
    summary_content.push_str(&format!("time_step = {} # integration time step (ps)\n", config.time_step));
    summary_content.push_str(&format!("n_time = {} # number of time steps\n", config.n_time));
    summary_content.push_str(&format!("el_reach = {} # energy grid reach around E0 (cm^-1)\n", config.el_reach));
    summary_content.push_str(&format!("m = {} # transition dipole moment length (Angstroms)\n", config.m));
    summary_content.push_str(&format!("n = {} # refractive index of medium\n", config.n));
    summary_content.push_str(&format!("raman_start = {} # start of raman shift axis (cm^-1)\n", config.raman_start));
    summary_content.push_str(&format!("raman_end = {} # end of raman shift axis (cm^-1)\n", config.raman_end));
    summary_content.push_str(&format!("raman_step = {} # raman shift axis step size (cm^-1)\n", config.raman_step));
    summary_content.push_str(&format!("raman_res = {} # raman peak resolution (cm^-1)\n", config.raman_res));
    summary_content.push_str(&format!("temp = {} # temperature (K)\n", config.temp));
    summary_content.push_str(&format!("convergence = {} # convergence threshold for higher-order sums\n", config.convergence));
    summary_content.push_str(&format!("boltz_toggle = {} # enable boltzmann thermal averaging\n", config.boltz_toggle));

    fs::write(output_dir.join("output.toml"), summary_content).map_err(|e| e.to_string())?;

    // Copy original files from source dir as old if they exist
    let src_toml = root.join("inp.toml");
    if src_toml.exists() {
        fs::copy(src_toml, output_dir.join("inp_old.toml")).ok();
    }
    let src_txt = root.join("inp.txt");
    if src_txt.exists() {
        fs::copy(src_txt, output_dir.join("inp_old.txt")).ok();
    }

    // Copy data files from source dir
    let data_files = [
        "freqs.dat",
        "abs_exp.dat",
        "fl_exp.dat",
        "profs_exp.dat",
        "rpumps.dat",
        "rshift_exp.dat",
    ];

    for file in data_files {
        let src = root.join(file);
        if src.exists() {
            fs::copy(src, output_dir.join(file)).ok();
        }
    }

    Ok(folder_name)
}

#[tauri::command]
fn load_fit_config_cmd(dir: String) -> Result<Option<FitConfig>, String> {
    let root = Path::new(&dir);
    let fit_path = root.join("fit.toml");
    if fit_path.exists() {
        let config = load_fit_config(fit_path).map_err(|e| e.to_string())?;
        Ok(Some(config))
    } else {
        Ok(None)
    }
}

#[tauri::command]
fn load_data(dir: String) -> Result<ResRamConfig, String> {
    let root = Path::new(&dir);
    let toml_path = root.join("inp.toml");
    let txt_path = root.join("inp.txt");

    if toml_path.exists() {
        load_config(toml_path).map_err(|e| e.to_string())
    } else if txt_path.exists() {
        migrate_txt_to_toml(txt_path, toml_path).map_err(|e| e.to_string())
    } else {
        Err("Neither inp.toml nor inp.txt found in the directory".into())
    }
}

#[tauri::command]
fn load_vibrational_data_cmd(dir: String) -> Result<(Vec<VibrationalMode>, Vec<f64>), String> {
    load_vibrational_data(dir).map_err(|e| e.to_string())
}

#[tauri::command]
fn run_calculation(dir: String, config: ResRamConfig, modes: Vec<VibrationalMode>, rpumps: Vec<f64>) -> Result<SimulationResult, String> {
    let root = Path::new(&dir);
    let el_reach = config.el_reach;
    let e0 = config.e0;
    let n_time = config.n_time;
    let ts = config.time_step;
    let th = Array1::linspace(0.0, (n_time as f64) * ts / 5.3088, n_time);
    let el = Array1::linspace(e0 - el_reach, e0 + el_reach, 1000);
    let e0_range = Array1::linspace(-el_reach * 0.5, el_reach * 0.5, 501);
    let out_len = el.len().max(e0_range.len()) - el.len().min(e0_range.len()) + 1;
    let conv_el = Array1::linspace(e0 - el_reach * 0.5, e0 + el_reach * 0.5, out_len);

    let mut result = compute_spectra(
        &config,
        &modes,
        &rpumps,
        &th.view(),
        &el.view(),
        &conv_el.view(),
        &e0_range.view(),
    );

    result.abs_exp = load_exp_interp(&root.join("abs_exp.dat"), &conv_el)?;
    result.fl_exp = load_exp_interp(&root.join("fl_exp.dat"), &conv_el)?;
    result.profs_exp = load_exp_matrix(&root.join("profs_exp.dat"))?;

    Ok(result)
}

#[tauri::command]
async fn run_fit<R: Runtime>(
    app: tauri::AppHandle<R>,
    dir: String,
    config: ResRamConfig,
    modes: Vec<VibrationalMode>,
    fit_indices: Vec<usize>,
    fit_gamma: bool,
    fit_m: bool,
    fit_theta: bool,
    fit_kappa: bool,
    fit_e0: bool,
    algorithm_name: String,
    max_eval: u32,
    refresh_step: u32,
) -> Result<FitResultPayload, String> {
    let root = Path::new(&dir);
    
    // Load experimental data
    let abs_exp_path = root.join("abs_exp.dat");
    let fl_exp_path = root.join("fl_exp.dat");
    let profs_exp_path = root.join("profs_exp.dat");
    let rp_path = root.join("rpumps.dat");

    let profs_exp_content = std::fs::read_to_string(profs_exp_path).map_err(|e| e.to_string())?;
    let profs_rows: Vec<Vec<f64>> = profs_exp_content.lines()
        .map(|l| l.split_whitespace().filter_map(|s| s.parse().ok()).collect())
        .filter(|r: &Vec<f64>| !r.is_empty())
        .collect();
    
    let n_modes = profs_rows.len();
    let n_pumps = if n_modes > 0 { profs_rows[0].len() } else { 0 };
    let mut profs_exp = Array2::zeros((n_modes, n_pumps));
    for (i, row) in profs_rows.into_iter().enumerate() {
        for (j, val) in row.into_iter().enumerate() {
            profs_exp[[i, j]] = val;
        }
    }

    let rp_content = std::fs::read_to_string(rp_path).map_err(|e| e.to_string())?;
    let rpumps: Vec<f64> = rp_content.lines().filter_map(|l| l.trim().parse().ok()).collect();

    // Grids
    let el_reach = config.el_reach;
    let e0 = config.e0;
    let n_time = config.n_time;
    let ts = config.time_step;
    let th = Array1::linspace(0.0, (n_time as f64) * ts / 5.3088, n_time);
    let el = Array1::linspace(e0 - el_reach, e0 + el_reach, 1000);
    let e0_range = Array1::linspace(-el_reach * 0.5, el_reach * 0.5, 501);
    let out_len = el.len().max(e0_range.len()) - el.len().min(e0_range.len()) + 1;
    let conv_el = Array1::linspace(e0 - el_reach * 0.5, e0 + el_reach * 0.5, out_len);

    // Python parity: interpolate experimental spectra onto convEL grid.
    let abs_exp_vec = load_exp_interp(&abs_exp_path, &conv_el)?
        .ok_or_else(|| "abs_exp.dat is required for fitting".to_string())?;
    let _fl_exp_vec = load_exp_interp(&fl_exp_path, &conv_el)?;
    // Python raman_residual uses abs_exp in loss; fl_exp is loaded for plotting only.
    let abs_exp: Array1<f64> = Array1::from(abs_exp_vec);

    // rp indices
    let mut rp_indices = Vec::new();
    for pump in rpumps {
        let mut min_diff = f64::MAX;
        let mut best_idx = 0;
        for (i, &e) in conv_el.iter().enumerate() {
            let diff = (e - pump).abs();
            if diff < min_diff {
                min_diff = diff;
                best_idx = i;
            }
        }
        rp_indices.push(best_idx);
    }

    let k_b_t = 0.695 * config.temp;
    let pre_a = ((5.744e-3) / config.n) * ts;
    let pre_f = pre_a * config.n.powi(2);
    let pre_r = 2.08e-20 * ts.powi(2);

    let context = OptimizationContext {
        config: config.clone(),
        modes: modes.clone(),
        th,
        el,
        conv_el,
        e0_range,
        abs_exp,
        profs_exp,
        rp: rp_indices,
        pre_a,
        pre_f,
        pre_r,
        beta: if config.temp > 0.1 { 1.0 / k_b_t } else { 1e10 },
        dt: ts,
        fit_indices: fit_indices.clone(),
        fit_gamma,
        fit_m,
        fit_theta,
        fit_kappa,
        fit_e0,
        iteration: 0,
        progress_callback: Some(Arc::new(move |iter, loss, params| {
            if iter % refresh_step == 0 {
                let _ = app.emit("fit-progress", ProgressPayload {
                    iteration: iter,
                    loss,
                    parameters: params.to_vec(),
                });
            }
        })),
    };

    let algorithm = parse_algorithm(&algorithm_name);

    // Run in blocking thread
    let (optimized_params, _final_loss) = tauri::async_runtime::spawn_blocking(move || {
        run_optimization(
            algorithm,
            context,
            max_eval,
            1e-8
        )
    }).await.map_err(|e| e.to_string())?.map_err(|e| e)?;

    let mut final_config = config;
    let mut final_modes = modes;
    let mut cursor = 0;

    // Apply optimized deltas back to mode displacements.
    for &idx in &fit_indices {
        final_modes[idx].displacement = optimized_params[cursor];
        cursor += 1;
    }
    if fit_gamma {
        final_config.gamma = optimized_params[cursor];
        cursor += 1;
    }
    if fit_m {
        final_config.m = optimized_params[cursor];
        cursor += 1;
    }
    if fit_theta {
        final_config.theta = optimized_params[cursor];
        cursor += 1;
    }
    if fit_kappa {
        final_config.kappa = optimized_params[cursor];
        cursor += 1;
    }
    if fit_e0 {
        final_config.e0 = optimized_params[cursor];
    }

    let fit_config = FitConfig {
        algorithm: algorithm_name,
        max_eval,
        ftol_rel: 1e-8,
        fit_indices,
        fit_gamma,
        fit_m,
        fit_theta,
        fit_kappa,
        fit_e0,
    };

    let folder_name = save_data_impl(&dir, &final_config, &final_modes, &fit_config)?;

    Ok(FitResultPayload {
        config: final_config,
        modes: final_modes,
        folder_name,
    })
}

#[tauri::command]
fn save_data(
    dir: String, 
    config: ResRamConfig, 
    modes: Vec<VibrationalMode>,
    fit_config: FitConfig
) -> Result<String, String> {
    save_data_impl(&dir, &config, &modes, &fit_config)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            load_data, 
            run_fit, 
            run_calculation, 
            load_vibrational_data_cmd,
            save_data,
            load_fit_config_cmd
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
