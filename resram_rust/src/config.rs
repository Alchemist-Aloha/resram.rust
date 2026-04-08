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
