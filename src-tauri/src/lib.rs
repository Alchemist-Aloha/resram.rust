use resram_rust::config::{load_config, load_vibrational_data, migrate_txt_to_toml};
use resram_rust::models::{ResRamConfig, VibrationalMode, SimulationResult};
use resram_rust::core::compute_spectra;
use resram_rust::optimizer::{run_optimization, OptimizationContext, ProgressCallback};
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};
use ndarray::prelude::*;
use std::sync::{Arc, Mutex};
use tauri::{Emitter, Runtime};

#[derive(Debug, Serialize, Deserialize)]
pub struct AppState {
    pub dir: Option<PathBuf>,
    pub config: Option<ResRamConfig>,
    pub modes: Option<Vec<VibrationalMode>>,
}

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

    let config = if toml_path.exists() {
        load_config(toml_path).map_err(|e| e.to_string())?
    } else if txt_path.exists() {
        migrate_txt_to_toml(txt_path, toml_path).map_err(|e| e.to_string())?
    } else {
        return Err("Neither inp.toml nor inp.txt found in the directory".into());
    };

    Ok(config)
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
    algorithm_name: String,
    max_eval: u32,
) -> Result<ResRamConfig, String> {
    let root = Path::new(&dir);
    
    // In a real app, you would load experimental data once and reuse it.
    // For now, let's load it here.
    let abs_exp_path = root.join("abs_exp.dat");
    let profs_exp_path = root.join("profs_exp.dat");
    let rp_path = root.join("rpumps.dat");

    let abs_exp_content = std::fs::read_to_string(abs_exp_path).map_err(|e| e.to_string())?;
    let abs_exp: Array1<f64> = abs_exp_content.lines().filter_map(|l| l.split('\t').nth(1)?.trim().parse().ok()).collect();

    let profs_exp_content = std::fs::read_to_string(profs_exp_path).map_err(|e| e.to_string())?;
    // This needs proper 2D parsing. Assuming simple whitespace for now.
    // ...
    
    // Progress callback
    let app_handle = app.clone();
    let callback: ProgressCallback = Arc::new(move |iter, loss, params| {
        let payload = ProgressPayload {
            iteration: iter,
            loss,
            parameters: params.to_vec(),
        };
        let _ = app_handle.emit("fit-progress", payload);
    });

    // Run optimization in a separate thread/task
    // For now, return the original config as a placeholder.
    // We'll implement the full optimization context setup here later.
    
    Ok(config)
}

#[tauri::command]
fn load_vibrational_data_cmd(dir: String) -> Result<(Vec<VibrationalMode>, Vec<f64>), String> {
    load_vibrational_data(dir).map_err(|e| e.to_string())
}

#[tauri::command]
fn run_calculation(config: ResRamConfig, modes: Vec<VibrationalMode>) -> Result<SimulationResult, String> {
    // Setup grids (simplified for now, should ideally be passed or calculated properly)
    let el_reach = config.el_reach;
    let e0 = config.e0;
    let n_time = config.n_time;
    let ts = config.time_step;
    let th = Array1::linspace(0.0, (n_time as f64) * ts / 5.3088, n_time);
    let el = Array1::linspace(e0 - el_reach, e0 + el_reach, 1000);
    let e0_range = Array1::linspace(-el_reach * 0.5, el_reach * 0.5, 501);
    let out_len = el.len().max(e0_range.len()) - el.len().min(e0_range.len()) + 1;
    let conv_el = Array1::linspace(e0 - el_reach * 0.5, e0 + el_reach * 0.5, out_len);

    let result = compute_spectra(
        &config,
        &modes,
        &th.view(),
        &el.view(),
        &conv_el.view(),
        &e0_range.view(),
    );

    Ok(result)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![load_data, run_fit, run_calculation, load_vibrational_data_cmd])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
