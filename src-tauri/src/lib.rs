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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![load_data, run_fit])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
