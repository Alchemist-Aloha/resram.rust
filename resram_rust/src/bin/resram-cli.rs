use std::path::PathBuf;
use anyhow::{Result, anyhow, Context};
use ndarray::prelude::*;
use resram_rust::config::{
    migrate_txt_to_toml, load_config, load_vibrational_data, 
    load_fit_config, load_experimental_data, write_config_toml, write_config_txt, 
    save_vibrational_data, write_vec, write_matrix, transpose_matrix
};
use resram_rust::core::compute_spectra;
use resram_rust::optimizer::{run_optimization, OptimizationContext};
use nlopt::Algorithm;
use std::sync::Arc;
use chrono::Local;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about = "Resonance Raman (ResRAM) CLI for Bodipy analysis", long_about = None)]
struct Args {
    /// Root directory containing input files (inp.toml/inp.txt, freqs.dat, deltas.dat, etc.)
    #[arg(short, long, default_value = ".")]
    dir: PathBuf,

    /// Trigger optimization (fitting) using fit.toml
    #[arg(short, long)]
    fit: bool,

    /// Custom input TOML file name
    #[arg(short, long, default_value = "inp.toml")]
    input: String,

    /// Custom fitting configuration TOML file name
    #[arg(long, default_value = "fit.toml")]
    fit_config: String,

    /// Skip saving results to a timestamped folder
    #[arg(long)]
    no_save: bool,
}

fn get_algorithm(name: &str) -> Algorithm {
    match name.to_lowercase().as_str() {
        "powell" | "praxis" => Algorithm::Praxis,
        "cobyla" => Algorithm::Cobyla,
        "bobyqa" => Algorithm::Bobyqa,
        "newuoa" => Algorithm::Newuoa,
        "newuoa_bound" | "newuoabound" => Algorithm::NewuoaBound,
        "neldermead" | "nelder-mead" => Algorithm::Neldermead,
        "sbplx" | "subplex" => Algorithm::Sbplx,
        "ln_sbplx" => Algorithm::Sbplx,
        "ln_cobyla" => Algorithm::Cobyla,
        "ln_bobyqa" => Algorithm::Bobyqa,
        "ln_newuoa" => Algorithm::Newuoa,
        "ln_neldermead" => Algorithm::Neldermead,
        "ld_lbfgs" => Algorithm::Lbfgs,
        _ => Algorithm::Sbplx,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = &args.dir;
    
    if !root.exists() {
        return Err(anyhow!("Directory not found: {:?}", root));
    }

    let toml_path = root.join(&args.input);
    let txt_path = root.join("inp.txt");
    let fit_toml_path = root.join(&args.fit_config);

    let mut config = if toml_path.exists() {
        println!("Loading configuration from {:?}...", toml_path);
        load_config(&toml_path)?
    } else if txt_path.exists() {
        println!("Migrating inp.txt to {:?}...", toml_path);
        migrate_txt_to_toml(&txt_path, &toml_path)?
    } else {
        return Err(anyhow!("Neither {} nor inp.txt found in {:?}", args.input, root));
    };

    println!("Loading vibrational data from {:?}...", root);
    let (mut modes, rpumps) = load_vibrational_data(root).context("Failed to load vibrational data (freqs.dat, deltas.dat)")?;

    println!("Simulation Parameters:");
    println!("  Gamma: {:.2} cm^-1", config.gamma);
    println!("  Theta: {:.2} cm^-1", config.theta);
    println!("  E0: {:.2} cm^-1", config.e0);
    println!("  Kappa: {:.3}", config.kappa);
    println!("  M: {:.3} A", config.m);
    println!("  Modes: {}", modes.len());
    if !rpumps.is_empty() {
        println!("  Pump wavelengths: {:?} cm^-1", rpumps);
    }

    // Setup grids
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

    if args.fit {
        if !fit_toml_path.exists() {
            return Err(anyhow!("--fit specified but {:?} not found in {:?}", args.fit_config, root));
        }
        println!("Found fitting configuration at {:?}, starting optimization...", fit_toml_path);
        let fit_config = load_fit_config(&fit_toml_path)?;
        
        println!("Fitting configuration:");
        println!("  Algorithm: {}", fit_config.algorithm);
        println!("  Max Evaluations: {}", fit_config.max_eval);
        println!("  FTOL Relative: {:.2e}", fit_config.ftol_rel);
        println!("  Fitting {} modes: {:?}", fit_config.fit_indices.len(), fit_config.fit_indices);
        if fit_config.fit_gamma { println!("  Fitting Gamma"); }
        if fit_config.fit_m { println!("  Fitting M"); }
        if fit_config.fit_theta { println!("  Fitting Theta"); }
        if fit_config.fit_kappa { println!("  Fitting Kappa"); }
        if fit_config.fit_e0 { println!("  Fitting E0"); }

        let (abs_exp, profs_exp) = load_experimental_data(root, &conv_el)?;

        if let (Some(abs_v), Some(profs_v)) = (abs_exp, profs_exp) {
            let abs_exp_arr = Array1::from_vec(abs_v);
            let n_modes = modes.len();
            
            let n_pumps = profs_v[0].len();
            let mut profs_flat = Vec::new();

            // Handle potential transposition or mismatched rows
            let profs_exp_arr = if profs_v.len() == n_modes {
                for row in profs_v {
                    if row.len() != n_pumps {
                        return Err(anyhow!("Inconsistent number of pumps in profs_exp.dat rows"));
                    }
                    profs_flat.extend(row);
                }
                Array2::from_shape_vec((n_modes, n_pumps), profs_flat)?
            } else if profs_v.len() == n_pumps && profs_v[0].len() == n_modes {
                println!("Detected transposed profs_exp.dat ({} pumps, {} modes). Transposing back...", n_pumps, n_modes);
                let mut data = Array2::zeros((n_pumps, n_modes));
                for (r, row) in profs_v.into_iter().enumerate() {
                    for (c, val) in row.into_iter().enumerate() {
                        data[[r, c]] = val;
                    }
                }
                data.reversed_axes()
            } else {
                return Err(anyhow!("profs_exp.dat size ({}, {}) does not match modes ({}) or pumps ({})", 
                    profs_v.len(), n_pumps, n_modes, rpumps.len()));
            };

            let n = config.n;
            let pre_a = ((5.744e-3) / n) * ts;
            let pre_f = pre_a * n.powi(2);
            let pre_r = 2.08e-20 * ts.powi(2);
            let beta = if config.temp > 0.1 { 1.0 / (0.695 * config.temp) } else { 1.0e10 };

            // rp indices calculation
            let mut rp_indices = Vec::new();
            for &pump in &rpumps {
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

            let context = OptimizationContext {
                config: config.clone(),
                modes: modes.clone(),
                th: th.clone(),
                el: el.clone(),
                conv_el: conv_el.clone(),
                e0_range: e0_range.clone(),
                abs_exp: abs_exp_arr,
                profs_exp: profs_exp_arr,
                rp: rp_indices,
                pre_a,
                pre_f,
                pre_r,
                beta,
                dt: ts,
                fit_indices: fit_config.fit_indices.clone(),
                fit_gamma: fit_config.fit_gamma,
                fit_m: fit_config.fit_m,
                fit_theta: fit_config.fit_theta,
                fit_kappa: fit_config.fit_kappa,
                fit_e0: fit_config.fit_e0,
                iteration: 0,
                progress_callback: Some(Arc::new(|iter, loss, _| {
                    if iter % 100 == 0 {
                        println!("Iteration {}: Loss = {:.4e}", iter, loss);
                    }
                })),
            };

            let algorithm = get_algorithm(&fit_config.algorithm);
            match run_optimization(algorithm, context, fit_config.max_eval, fit_config.ftol_rel) {
                Ok((x, loss)) => {
                    println!("Optimization complete! Final Loss = {:.4e}", loss);
                    let mut cursor = 0;
                    for &idx in &fit_config.fit_indices {
                        modes[idx].displacement = x[cursor];
                        cursor += 1;
                    }
                    if fit_config.fit_gamma {
                        config.gamma = x[cursor];
                        cursor += 1;
                    }
                    if fit_config.fit_m {
                        config.m = x[cursor];
                        cursor += 1;
                    }
                    if fit_config.fit_theta {
                        config.theta = x[cursor];
                        cursor += 1;
                    }
                    if fit_config.fit_kappa {
                        config.kappa = x[cursor];
                        cursor += 1;
                    }
                    if fit_config.fit_e0 {
                        config.e0 = x[cursor];
                    }
                }
                Err(e) => println!("Optimization failed: {}", e),
            }
        } else {
            println!("Experimental data (abs_exp.dat and profs_exp.dat) not found or invalid, skipping optimization.");
        }
    }

    println!("Starting final calculation and saving results...");
    let result = compute_spectra(
        &config,
        &modes,
        &rpumps,
        &th.view(),
        &el.view(),
        &conv_el.view(),
        &e0_range.view(),
    );

    if !args.no_save {
        // Timestamped folder creation logic matching Tauri UI
        let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
        let folder_name = format!("data_{}", timestamp);
        let output_dir = root.join(&folder_name);
        std::fs::create_dir_all(&output_dir)?;

        println!("Saving results to {:?}...", folder_name);
        write_config_toml(output_dir.join("inp.toml"), &config)?;
        write_config_txt(output_dir.join("inp.txt"), &config)?;
        resram_rust::config::write_summary_toml(output_dir.join("output.toml"), &config, &timestamp)?;
        save_vibrational_data(&output_dir, &modes)?;
        
        write_vec(output_dir.join("Abs.dat"), &result.abs_cross)?;
        write_vec(output_dir.join("Fl.dat"), &result.fl_cross)?;
        write_vec(output_dir.join("EL.dat"), &result.conv_el)?;
        write_vec(output_dir.join("rshift.dat"), &result.rshift)?;

        let profs = transpose_matrix(&result.raman_cross);
        write_matrix(output_dir.join("profs.dat"), &profs)?;

        if !result.raman_spec.is_empty() {
            let raman_spec_t = transpose_matrix(&result.raman_spec);
            write_matrix(output_dir.join("raman_spec.dat"), &raman_spec_t)?;
            if !rpumps.is_empty() {
                write_vec(output_dir.join("rpumps.dat"), &rpumps)?;
            }
        }

        // Copy original input and data files as backups
        if toml_path.exists() { std::fs::copy(&toml_path, output_dir.join("inp_old.toml")).ok(); }
        if txt_path.exists() { std::fs::copy(&txt_path, output_dir.join("inp_old.txt")).ok(); }

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
                std::fs::copy(src, output_dir.join(file)).ok();
            }
        }
    }

    println!("Calculation complete.");
    println!("Abs peak value: {:.4e}", result.abs_cross.iter().fold(0.0f64, |a, &b| a.max(b)));
    println!("FL peak value: {:.4e}", result.fl_cross.iter().fold(0.0f64, |a, &b| a.max(b)));

    Ok(())
}
