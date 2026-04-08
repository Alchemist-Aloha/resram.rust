use std::fs;
use std::path::Path;
use anyhow::{Result, anyhow};
use crate::models::{ResRamConfig, VibrationalMode};

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
