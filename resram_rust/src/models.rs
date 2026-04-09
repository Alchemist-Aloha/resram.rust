use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResRamConfig {
    pub gamma: f64,
    pub theta: f64,
    pub e0: f64,
    pub kappa: f64,
    pub time_step: f64,
    pub n_time: usize,
    pub el_reach: f64,
    pub m: f64,
    pub n: f64,
    pub raman_start: f64,
    pub raman_end: f64,
    pub raman_step: f64,
    pub raman_res: f64,
    pub temp: f64,
    pub convergence: f64,
    pub boltz_toggle: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FitConfig {
    #[serde(default = "default_algorithm")]
    pub algorithm: String,
    #[serde(default = "default_max_eval")]
    pub max_eval: u32,
    #[serde(default = "default_ftol_rel")]
    pub ftol_rel: f64,
    #[serde(default)]
    pub fit_indices: Vec<usize>,
    #[serde(default)]
    pub fit_gamma: bool,
    #[serde(default)]
    pub fit_m: bool,
    #[serde(default)]
    pub fit_theta: bool,
    #[serde(default)]
    pub fit_kappa: bool,
    #[serde(default)]
    pub fit_e0: bool,
}

fn default_algorithm() -> String { "sbplx".to_string() }
fn default_max_eval() -> u32 { 1000 }
fn default_ftol_rel() -> f64 { 1e-6 }

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VibrationalMode {
    pub frequency: f64,
    pub displacement: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SimulationResult {
    pub abs_cross: Vec<f64>,
    pub fl_cross: Vec<f64>,
    pub abs_exp: Option<Vec<f64>>,
    pub fl_exp: Option<Vec<f64>>,
    pub profs_exp: Option<Vec<Vec<f64>>>,
    pub raman_cross: Vec<Vec<f64>>, // [mode_idx][energy_idx]
    pub raman_spec: Vec<Vec<f64>>,  // [pump_idx][rshift_idx]
    pub rp_indices: Vec<usize>,
    pub conv_el: Vec<f64>,
    pub rshift: Vec<f64>,
}
