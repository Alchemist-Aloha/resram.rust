use resram_rust::config::{load_config, load_vibrational_data, migrate_txt_to_toml, save_config, save_vibrational_data, write_config_txt};
use resram_rust::models::{ResRamConfig, VibrationalMode, SimulationResult};
use resram_rust::core::compute_spectra;
use resram_rust::optimizer::{run_optimization, OptimizationContext, ProgressCallback};
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};
use ndarray::prelude::*;
use std::sync::Arc;
use tauri::{Emitter, Runtime};

#[derive(Clone, Serialize)]
struct ProgressPayload {
    iteration: u32,
    loss: f64,
    parameters: Vec<f64>,
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
fn run_calculation(config: ResRamConfig, modes: Vec<VibrationalMode>, rpumps: Vec<f64>) -> Result<SimulationResult, String> {
    let el_reach = config.el_reach;
    let e0 = config.e0;
    let n_time = config.n_time;
    let ts = config.time_step;
    let th = Array1::linspace(0.0, (n_time as f64) * ts / 5.3088, n_time);
    let el = Array1::linspace(e0 - el_reach, e0 + el_reach, 1000);
    let e0_range = Array1::linspace(-el_reach * 0.5, el_reach * 0.5, 501);
    let out_len = el.len().max(e0_range.len()) - el.len().min(e0_range.len()) + 1;
    let conv_el = Array1::linspace(e0 - el_reach * 0.5, e0 + el_reach * 0.5, out_len);

    Ok(compute_spectra(
        &config,
        &modes,
        &rpumps,
        &th.view(),
        &el.view(),
        &conv_el.view(),
        &e0_range.view(),
    ))
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
) -> Result<ResRamConfig, String> {
    let root = Path::new(&dir);
    
    // Load experimental data
    let abs_exp_path = root.join("abs_exp.dat");
    let profs_exp_path = root.join("profs_exp.dat");
    let rp_path = root.join("rpumps.dat");

    let abs_exp_content = std::fs::read_to_string(abs_exp_path).map_err(|e| e.to_string())?;
    let abs_exp: Array1<f64> = abs_exp_content.lines()
        .filter_map(|l| {
            let parts: Vec<&str> = l.split_whitespace().collect();
            if parts.len() >= 2 { parts[1].parse().ok() } else { l.parse().ok() }
        })
        .collect();

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

    let algorithm = match algorithm_name.as_str() {
        "cobyla" => nlopt::Algorithm::Cobyla,
        _ => nlopt::Algorithm::Bobyqa,
    };

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
    let mut cursor = 0;
    // We update final_config here.
    for _ in &fit_indices {
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

    Ok(final_config)
}

#[tauri::command]
fn save_data(dir: String, config: ResRamConfig, modes: Vec<VibrationalMode>) -> Result<String, String> {
    use std::fs;
    use chrono::Local;

    let root = Path::new(&dir);
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let folder_name = format!("data_{}", timestamp);
    let output_dir = root.join(&folder_name);

    fs::create_dir_all(&output_dir).map_err(|e| format!("Failed to create directory: {}", e))?;

    // 1. Save updated files
    let toml_path = output_dir.join("inp.toml");
    save_config(&toml_path, &config).map_err(|e| e.to_string())?;
    
    let txt_path = output_dir.join("inp.txt");
    write_config_txt(&txt_path, &config).map_err(|e| e.to_string())?;

    save_vibrational_data(&output_dir, &modes).map_err(|e| e.to_string())?;

    // 2. Save output.toml (summary)
    #[derive(Serialize)]
    struct OutputSummary {
        parameters: ResRamConfig,
        timestamp: String,
    }
    let summary = OutputSummary {
        parameters: config.clone(),
        timestamp: timestamp.clone(),
    };
    let summary_content = toml::to_string_pretty(&summary).map_err(|e: toml::ser::Error| e.to_string())?;
    fs::write(output_dir.join("output.toml"), summary_content).map_err(|e| e.to_string())?;

    // 3. Copy original files from source dir as "old" if they exist
    let src_toml = root.join("inp.toml");
    if src_toml.exists() {
        fs::copy(src_toml, output_dir.join("inp_old.toml")).ok();
    }
    let src_txt = root.join("inp.txt");
    if src_txt.exists() {
        fs::copy(src_txt, output_dir.join("inp_old.txt")).ok();
    }

    // 4. Copy other data files from source dir
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            load_data, 
            run_fit, 
            run_calculation, 
            load_vibrational_data_cmd,
            save_data
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
