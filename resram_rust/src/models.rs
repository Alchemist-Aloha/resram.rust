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
pub struct VibrationalMode {
    pub frequency: f64,
    pub displacement: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SimulationResult {
    pub abs_cross: Vec<f64>,
    pub fl_cross: Vec<f64>,
    pub raman_cross: Vec<Vec<f64>>, // [mode_idx][energy_idx]
    pub conv_el: Vec<f64>,
}
