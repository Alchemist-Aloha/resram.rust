use std::path::Path;
use anyhow::Result;
use ndarray::prelude::*;
use resram_rust::config::{migrate_txt_to_toml, load_config, load_vibrational_data};
use resram_rust::core::compute_spectra;

fn main() -> Result<()> {
    let root = Path::new(".");
    let toml_path = root.join("inp.toml");
    let txt_path = root.join("inp.txt");

    let config = if toml_path.exists() {
        println!("Loading existing inp.toml...");
        load_config(toml_path)?
    } else if txt_path.exists() {
        println!("Migrating inp.txt to inp.toml...");
        migrate_txt_to_toml(txt_path, toml_path)?
    } else {
        return Err(anyhow::anyhow!("Neither inp.toml nor inp.txt found in current directory"));
    };

    println!("Loading vibrational data...");
    let (modes, _rpumps) = load_vibrational_data(root)?;

    // Setup grids (simplified for CLI test)
    let el_reach = config.el_reach;
    let e0 = config.e0;
    let n_time = config.n_time;
    let ts = config.time_step;
    let th = Array1::linspace(0.0, (n_time as f64) * ts / 5.3088, n_time); // ps / hbar
    let el = Array1::linspace(e0 - el_reach, e0 + el_reach, 1000);
    let e0_range = Array1::linspace(-el_reach * 0.5, el_reach * 0.5, 501);
    
    // conv_el logic
    let out_len = el.len().max(e0_range.len()) - el.len().min(e0_range.len()) + 1;
    let conv_el = Array1::linspace(e0 - el_reach * 0.5, e0 + el_reach * 0.5, out_len);

    println!("Starting calculation...");
    let result = compute_spectra(
        &config,
        &modes,
        &th.view(),
        &el.view(),
        &conv_el.view(),
        &e0_range.view(),
    );

    println!("Calculation complete.");
    println!("Abs peak value: {}", result.abs_cross.iter().fold(0.0f64, |a, &b| a.max(b)));
    println!("FL peak value: {}", result.fl_cross.iter().fold(0.0f64, |a, &b| a.max(b)));

    Ok(())
}
