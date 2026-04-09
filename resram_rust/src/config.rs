use std::fs;
use std::path::Path;
use anyhow::{Result, anyhow};
use ndarray::prelude::*;
use crate::models::{ResRamConfig, VibrationalMode, FitConfig};

pub fn load_config<P: AsRef<Path>>(path: P) -> Result<ResRamConfig> {
    let content = fs::read_to_string(path)?;
    let config: ResRamConfig = toml::from_str(&content)?;
    Ok(config)
}

pub fn save_config<P: AsRef<Path>>(path: P, config: &ResRamConfig) -> Result<()> {
    let content = toml::to_string_pretty(config)?;
    fs::write(path, content)?;
    Ok(())
}

pub fn load_fit_config<P: AsRef<Path>>(path: P) -> Result<FitConfig> {
    let content = fs::read_to_string(path)?;
    let config: FitConfig = toml::from_str(&content)?;
    Ok(config)
}

pub fn save_fit_config<P: AsRef<Path>>(path: P, config: &FitConfig) -> Result<()> {
    let content = toml::to_string_pretty(config)?;
    fs::write(path, content)?;
    Ok(())
}

pub fn interp_linear(x: &[f64], y: &[f64], target_x: &Array1<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(target_x.len());
    for &xn in target_x {
        if xn <= x[0] {
            out.push(y[0]);
            continue;
        }
        if xn >= *x.last().unwrap() {
            out.push(*y.last().unwrap());
            continue;
        }
        let mut lo = 0;
        let mut hi = x.len() - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if x[mid] < xn {
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

pub fn parse_spectrum_xy<P: AsRef<Path>>(path: P) -> Result<(Vec<f64>, Vec<f64>)> {
    let content = fs::read_to_string(path)?;
    let mut x = Vec::new();
    let mut y = Vec::new();
    for line in content.lines() {
        let parts: Vec<f64> = line.split_whitespace().filter_map(|s| s.parse().ok()).collect();
        if parts.len() >= 2 {
            x.push(parts[0]);
            y.push(parts[1]);
        }
    }
    
    // Ensure X is monotonically increasing (like in Python logic)
    if x.len() > 1 && x[0] > *x.last().unwrap() {
        x.reverse();
        y.reverse();
    }
    
    Ok((x, y))
}

pub fn load_exp_interp<P: AsRef<Path>>(path: P, conv_el: &Array1<f64>) -> Result<Option<Vec<f64>>> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(None);
    }
    let (x, y) = parse_spectrum_xy(path)?;
    if y.is_empty() {
        return Ok(None);
    }
    Ok(Some(interp_linear(&x, &y, conv_el)))
}

pub fn load_experimental_data<P: AsRef<Path>>(dir: P, conv_el: &Array1<f64>) -> Result<(Option<Vec<f64>>, Option<Vec<Vec<f64>>>)> {
    let dir = dir.as_ref();
    let abs_exp_path = dir.join("abs_exp.dat");
    let profs_exp_path = dir.join("profs_exp.dat");

    let abs_exp = load_exp_interp(abs_exp_path, conv_el)?;

    let profs_exp = if profs_exp_path.exists() {
        let content = fs::read_to_string(profs_exp_path)?;
        let mut rows = Vec::new();
        for line in content.lines() {
            let row: Vec<f64> = line.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if !row.is_empty() {
                rows.push(row);
            }
        }
        Some(rows)
    } else {
        None
    };

    Ok((abs_exp, profs_exp))
}

pub fn migrate_txt_to_toml<P: AsRef<Path>>(txt_path: P, toml_path: P) -> Result<ResRamConfig> {
    let content = fs::read_to_string(txt_path)?;
    let lines: Vec<&str> = content.lines().map(|l| l.split('#').next().unwrap().trim()).collect();
    
    if lines.len() < 16 {
        return Err(anyhow!("Invalid inp.txt format: expected at least 16 lines"));
    }

    let config = ResRamConfig {
        gamma: lines[0].parse()?,
        theta: lines[1].parse()?,
        e0: lines[2].parse()?,
        kappa: lines[3].parse()?,
        time_step: lines[4].parse()?,
        n_time: lines[5].parse::<f64>()? as usize,
        el_reach: lines[6].parse()?,
        m: lines[7].parse()?,
        n: lines[8].parse()?,
        raman_start: lines[9].parse()?,
        raman_end: lines[10].parse()?,
        raman_step: lines[11].parse()?,
        raman_res: lines[12].parse()?,
        temp: lines[13].parse()?,
        convergence: lines[14].parse()?,
        boltz_toggle: lines[15].parse::<i32>()? != 0,
    };

    save_config(toml_path, &config)?;
    Ok(config)
}

pub fn load_vibrational_data<P: AsRef<Path>>(dir: P) -> Result<(Vec<VibrationalMode>, Vec<f64>)> {
    let dir = dir.as_ref();
    let freqs_path = dir.join("freqs.dat");
    let deltas_path = dir.join("deltas.dat");
    let rpumps_path = dir.join("rpumps.dat");

    let freqs_content = fs::read_to_string(freqs_path)?;
    let freqs: Vec<f64> = freqs_content.lines().filter_map(|l| l.trim().parse().ok()).collect();

    let deltas_content = fs::read_to_string(deltas_path)?;
    let deltas: Vec<f64> = deltas_content.lines().filter_map(|l| l.trim().parse().ok()).collect();

    if freqs.len() != deltas.len() {
        return Err(anyhow!("Mismatched lengths in freqs.dat and deltas.dat"));
    }

    let modes = freqs.into_iter().zip(deltas.into_iter())
        .map(|(f, d)| VibrationalMode { frequency: f, displacement: d })
        .collect();

    let rpumps = if rpumps_path.exists() {
        let content = fs::read_to_string(rpumps_path)?;
        content.lines().filter_map(|l| l.trim().parse().ok()).collect()
    } else {
        Vec::new()
    };

    Ok((modes, rpumps))
}

pub fn write_config_toml<P: AsRef<Path>>(path: P, config: &ResRamConfig) -> Result<()> {
    let mut content = String::new();
    content.push_str("# ResRAM Configuration File\n\n");
    content.push_str(&format!("gamma = {} # homogeneous broadening parameter (cm^-1)\n", config.gamma));
    content.push_str(&format!("theta = {} # static inhomogeneous broadening parameter (cm^-1)\n", config.theta));
    content.push_str(&format!("e0 = {} # 0-0 transition energy (cm^-1)\n", config.e0));
    content.push_str(&format!("kappa = {} # brownian oscillator kappa parameter\n", config.kappa));
    content.push_str(&format!("time_step = {} # integration time step (ps)\n", config.time_step));
    content.push_str(&format!("n_time = {} # number of time steps\n", config.n_time));
    content.push_str(&format!("el_reach = {} # energy grid reach around E0 (cm^-1)\n", config.el_reach));
    content.push_str(&format!("m = {} # transition dipole moment length (Angstroms)\n", config.m));
    content.push_str(&format!("n = {} # refractive index of medium\n", config.n));
    content.push_str(&format!("raman_start = {} # start of raman shift axis (cm^-1)\n", config.raman_start));
    content.push_str(&format!("raman_end = {} # end of raman shift axis (cm^-1)\n", config.raman_end));
    content.push_str(&format!("raman_step = {} # raman shift axis step size (cm^-1)\n", config.raman_step));
    content.push_str(&format!("raman_res = {} # raman peak resolution (cm^-1)\n", config.raman_res));
    content.push_str(&format!("temp = {} # temperature (K)\n", config.temp));
    content.push_str(&format!("convergence = {} # convergence threshold for higher-order sums\n", config.convergence));
    content.push_str(&format!("boltz_toggle = {} # enable boltzmann thermal averaging\n", config.boltz_toggle));
    
    fs::write(path, content)?;
    Ok(())
}

pub fn write_config_txt<P: AsRef<Path>>(path: P, config: &ResRamConfig) -> Result<()> {
    let mut content = String::new();
    content.push_str(&format!("{} # gamma linewidth parameter (cm^-1)\n", config.gamma));
    content.push_str(&format!("{} # theta static inhomogeneous linewidth parameter (cm^-1)\n", config.theta));
    content.push_str(&format!("{} # E0 (cm^-1)\n", config.e0));
    content.push_str(&format!("{} # kappa solvent parameter\n", config.kappa));
    content.push_str(&format!("{} # time step (ps)\n", config.time_step));
    content.push_str(&format!("{} # number of time steps\n", config.n_time));
    content.push_str(&format!("{} # range plus and minus E0 to calculate lineshapes\n", config.el_reach));
    content.push_str(&format!("{} # transition length M (Angstroms)\n", config.m));
    content.push_str(&format!("{} # refractive index n\n", config.n));
    content.push_str(&format!("{} # start raman shift axis (cm^-1)\n", config.raman_start));
    content.push_str(&format!("{} # end raman shift axis (cm^-1)\n", config.raman_end));
    content.push_str(&format!("{} # rshift axis step size (cm^-1)\n", config.raman_step));
    content.push_str(&format!("{} # raman spectrum resolution (cm^-1)\n", config.raman_res));
    content.push_str(&format!("{} # Temperature (K)\n", config.temp));
    content.push_str(&format!("{} # convergence for sums\n", config.convergence));
    content.push_str(&format!("{} # Boltz Toggle\n", if config.boltz_toggle { 1 } else { 0 }));
    
    fs::write(path, content)?;
    Ok(())
}

pub fn save_vibrational_data<P: AsRef<Path>>(dir: P, modes: &[VibrationalMode]) -> Result<()> {
    let path = dir.as_ref().join("deltas.dat");
    let content = modes.iter()
        .map(|m| m.displacement.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(path, content)?;
    Ok(())
}

pub fn write_vec<P: AsRef<Path>>(path: P, values: &[f64]) -> Result<()> {
    let content = values
        .iter()
        .map(|v| format!("{:.12e}", v))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(path, content)?;
    Ok(())
}

pub fn write_matrix<P: AsRef<Path>>(path: P, rows: &[Vec<f64>]) -> Result<()> {
    let content = rows
        .iter()
        .map(|row| row.iter().map(|v| format!("{:.12e}", v)).collect::<Vec<_>>().join("\t"))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(path, content)?;
    Ok(())
}

pub fn write_summary_toml<P: AsRef<Path>>(path: P, config: &ResRamConfig, timestamp: &str) -> Result<()> {
    let mut content = String::new();
    content.push_str("# ResRAM Output Summary\n");
    content.push_str(&format!("timestamp = \"{}\"\n\n", timestamp));
    content.push_str("[parameters]\n");
    content.push_str(&format!("gamma = {} # homogeneous broadening parameter (cm^-1)\n", config.gamma));
    content.push_str(&format!("theta = {} # static inhomogeneous broadening parameter (cm^-1)\n", config.theta));
    content.push_str(&format!("e0 = {} # 0-0 transition energy (cm^-1)\n", config.e0));
    content.push_str(&format!("kappa = {} # brownian oscillator kappa parameter\n", config.kappa));
    content.push_str(&format!("time_step = {} # integration time step (ps)\n", config.time_step));
    content.push_str(&format!("n_time = {} # number of time steps\n", config.n_time));
    content.push_str(&format!("el_reach = {} # energy grid reach around E0 (cm^-1)\n", config.el_reach));
    content.push_str(&format!("m = {} # transition dipole moment length (Angstroms)\n", config.m));
    content.push_str(&format!("n = {} # refractive index of medium\n", config.n));
    content.push_str(&format!("raman_start = {} # start of raman shift axis (cm^-1)\n", config.raman_start));
    content.push_str(&format!("raman_end = {} # end of raman shift axis (cm^-1)\n", config.raman_end));
    content.push_str(&format!("raman_step = {} # raman shift axis step size (cm^-1)\n", config.raman_step));
    content.push_str(&format!("raman_res = {} # raman peak resolution (cm^-1)\n", config.raman_res));
    content.push_str(&format!("temp = {} # temperature (K)\n", config.temp));
    content.push_str(&format!("convergence = {} # convergence threshold for higher-order sums\n", config.convergence));
    content.push_str(&format!("boltz_toggle = {} # enable boltzmann thermal averaging\n", config.boltz_toggle));
    
    fs::write(path, content)?;
    Ok(())
}

pub fn transpose_matrix(input: &[Vec<f64>]) -> Vec<Vec<f64>> {
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
