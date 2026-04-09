import { useState, useEffect, useRef } from "react";
import Plot from "react-plotly.js";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { Settings, Play, Activity, FolderOpen, List, Save } from "lucide-react";
import "./App.css";

interface ResRamConfig {
  gamma: number;
  theta: number;
  e0: number;
  kappa: number;
  m: number;
  n: number;
  temp: number;
  time_step: number;
  n_time: number;
  el_reach: number;
  raman_start: number;
  raman_end: number;
  raman_step: number;
  raman_res: number;
  convergence: number;
  boltz_toggle: boolean;
}

interface FitConfig {
  algorithm: string;
  max_eval: number;
  ftol_rel: number;
  fit_indices: number[];
  fit_gamma: boolean;
  fit_m: boolean;
  fit_theta: boolean;
  fit_kappa: boolean;
  fit_e0: boolean;
}

interface VibrationalMode {
  frequency: number;
  displacement: number;
}

interface SimulationResult {
  abs_cross: number[];
  fl_cross: number[];
  abs_exp?: number[];
  fl_exp?: number[];
  profs_exp?: number[][];
  raman_cross: number[][];
  raman_spec: number[][];
  rp_indices: number[];
  conv_el: number[];
  rshift: number[];
}

interface ProgressPayload {
  iteration: number;
  loss: number;
  parameters: number[];
}

interface FitResultPayload {
  config: ResRamConfig;
  modes: VibrationalMode[];
  folder_name: string;
}

interface FitSession {
  baseConfig: ResRamConfig;
  baseModes: VibrationalMode[];
  fitIndices: number[];
  fitGamma: boolean;
  fitM: boolean;
  fitTheta: boolean;
  fitKappa: boolean;
  fitE0: boolean;
}

function App() {
  const [config, setConfig] = useState<ResRamConfig | null>(null);
  const [modes, setModes] = useState<VibrationalMode[]>([]);
  const [rpumps, setRpumps] = useState<number[]>([]);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [dir, setDir] = useState("");
  const [status, setStatus] = useState("Ready");
  const [maxEval, setMaxEval] = useState(1000);
  const [refreshStep, setRefreshStep] = useState(10);
  const [fitAlgorithm, setFitAlgorithm] = useState("powell");
  const [, setProgress] = useState<ProgressPayload | null>(null);
  const [isFitting, setIsFitting] = useState(false);
  const fitSessionRef = useRef<FitSession | null>(null);

  // Fitting toggles
  const [fitSwitches, setFitSwitches] = useState({
    gamma: true,
    m: false,
    theta: false,
    kappa: false,
    e0: false,
    modes: [] as boolean[]
  });

  useEffect(() => {
    const unlisten = listen<ProgressPayload>("fit-progress", (event) => {
      setProgress(event.payload);
      setStatus(`Fitting: Iteration ${event.payload.iteration}, Loss: ${event.payload.loss.toExponential(4)}`);

      const session = fitSessionRef.current;
      if (!session) return;

      let cursor = 0;
      const nextModes = session.baseModes.map((m) => ({ ...m }));
      const nextConfig: ResRamConfig = { ...session.baseConfig };

      for (const idx of session.fitIndices) {
        if (cursor >= event.payload.parameters.length) return;
        nextModes[idx].displacement = event.payload.parameters[cursor];
        cursor += 1;
      }

      if (session.fitGamma) {
        if (cursor >= event.payload.parameters.length) return;
        nextConfig.gamma = event.payload.parameters[cursor];
        cursor += 1;
      }
      if (session.fitM) {
        if (cursor >= event.payload.parameters.length) return;
        nextConfig.m = event.payload.parameters[cursor];
        cursor += 1;
      }
      if (session.fitTheta) {
        if (cursor >= event.payload.parameters.length) return;
        nextConfig.theta = event.payload.parameters[cursor];
        cursor += 1;
      }
      if (session.fitKappa) {
        if (cursor >= event.payload.parameters.length) return;
        nextConfig.kappa = event.payload.parameters[cursor];
        cursor += 1;
      }
      if (session.fitE0) {
        if (cursor >= event.payload.parameters.length) return;
        nextConfig.e0 = event.payload.parameters[cursor];
      }

      setModes(nextModes);
      setConfig(nextConfig);
    });
    
    // Initial loading from sample_data
    loadFolder("sample_data");

    return () => {
      unlisten.then(f => f());
    };
  }, []);

  useEffect(() => {
    if (!dir || !config || modes.length === 0) return;
    const timer = window.setTimeout(() => {
      runCalcWith(config, modes, rpumps);
    }, 300);

    return () => window.clearTimeout(timer);
  }, [dir, config, modes, rpumps]);

  async function loadFolder(path: string) {
    try {
      const loadedConfig = await invoke<ResRamConfig>("load_data", { dir: path });
      const [loadedModes, loadedRpumps] = await invoke<[VibrationalMode[], number[]]>("load_vibrational_data_cmd", { dir: path });
      const fitConfig = await invoke<FitConfig | null>("load_fit_config_cmd", { dir: path });

      setConfig(loadedConfig);
      setModes(loadedModes);
      setRpumps(loadedRpumps);
      
      if (fitConfig) {
        setFitAlgorithm(fitConfig.algorithm);
        setMaxEval(fitConfig.max_eval);
        setFitSwitches({
          gamma: fitConfig.fit_gamma,
          m: fitConfig.fit_m,
          theta: fitConfig.fit_theta,
          kappa: fitConfig.fit_kappa,
          e0: fitConfig.fit_e0,
          modes: loadedModes.map((_, i) => fitConfig.fit_indices.includes(i))
        });
      } else {
        setFitSwitches({
          gamma: true,
          m: false,
          theta: false,
          kappa: false,
          e0: false,
          modes: new Array(loadedModes.length).fill(true)
        });
      }
      
      setDir(path);
      setStatus(`Loaded data from ${path}`);
    } catch (e) {
      setStatus(`Error: ${e}`);
    }
  }

  async function openFolder() {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        defaultPath: dir || "."
      });
      if (selected && typeof selected === 'string') {
        loadFolder(selected);
      }
    } catch (e) {
      setStatus(`Error opening folder: ${e}`);
    }
  }

  async function runCalcWith(nextConfig: ResRamConfig, nextModes: VibrationalMode[], nextRpumps: number[]) {
    try {
      const res = await invoke<SimulationResult>("run_calculation", {
        dir,
        config: nextConfig,
        modes: nextModes,
        rpumps: nextRpumps,
      });
      setResult(res);
    } catch (e) {
      setStatus(`Error: ${e}`);
    }
  }

  async function runCalc() {
    if (!config || modes.length === 0) return;
    setStatus("Calculating...");
    await runCalcWith(config, modes, rpumps);
  }

  async function startFit() {
    if (!config || modes.length === 0) return;
    setStatus("Starting fit...");
    setIsFitting(true);
    try {
      const fitIndices = modes.map((_, i) => i).filter(i => fitSwitches.modes[i]);
      fitSessionRef.current = {
        baseConfig: { ...config },
        baseModes: modes.map((m) => ({ ...m })),
        fitIndices,
        fitGamma: fitSwitches.gamma,
        fitM: fitSwitches.m,
        fitTheta: fitSwitches.theta,
        fitKappa: fitSwitches.kappa,
        fitE0: fitSwitches.e0,
      };
      const fitResult = await invoke<FitResultPayload>("run_fit", {
        dir,
        config,
        modes,
        fitIndices,
        fitGamma: fitSwitches.gamma,
        fitM: fitSwitches.m,
        fitTheta: fitSwitches.theta,
        fitKappa: fitSwitches.kappa,
        fitE0: fitSwitches.e0,
          algorithmName: fitAlgorithm,
        maxEval: maxEval,
        refreshStep: refreshStep,
      });
      setConfig(fitResult.config);
      setModes(fitResult.modes);
      await runCalcWith(fitResult.config, fitResult.modes, rpumps);
      setStatus(`Fitting complete and saved: ${fitResult.folder_name}`);
    } catch (e) {
      setStatus(`Error: ${e}`);
    } finally {
      fitSessionRef.current = null;
      setIsFitting(false);
    }
  }

  async function handleSave() {
    if (!config || !dir) return;
    setStatus("Saving...");
    try {
      const fit_config: FitConfig = {
        algorithm: fitAlgorithm,
        max_eval: maxEval,
        ftol_rel: 1e-8,
        fit_indices: modes.map((_, i) => i).filter(i => fitSwitches.modes[i]),
        fit_gamma: fitSwitches.gamma,
        fit_m: fitSwitches.m,
        fit_theta: fitSwitches.theta,
        fit_kappa: fitSwitches.kappa,
        fit_e0: fitSwitches.e0,
      };
      const folderName = await invoke<string>("save_data", { dir, config, modes, fitConfig: fit_config });
      setStatus(`Saved to folder: ${folderName}`);
    } catch (e) {
      setStatus(`Error: ${e}`);
    }
  }

  const updateMode = (idx: number, delta: number) => {
    const newModes = [...modes];
    newModes[idx].displacement = delta;
    setModes(newModes);
  };

  const toggleModeFit = (idx: number) => {
    const newSwitches = [...fitSwitches.modes];
    newSwitches[idx] = !newSwitches[idx];
    setFitSwitches({...fitSwitches, modes: newSwitches});
  };

  const modeColor = (i: number) => {
    const hue = (i * 47) % 360;
    return `hsl(${hue}, 70%, 45%)`;
  };

  const normalizedFlExp = (result?: SimulationResult | null): number[] | null => {
    if (!result?.fl_exp || result.fl_exp.length === 0 || result.fl_cross.length === 0) {
      return null;
    }
    const maxCalc = Math.max(...result.fl_cross.map((v) => Math.abs(v)));
    const maxExp = Math.max(...result.fl_exp.map((v) => Math.abs(v)));
    if (!Number.isFinite(maxCalc) || !Number.isFinite(maxExp) || maxExp <= 0) {
      return result.fl_exp;
    }
    return result.fl_exp.map((v) => (maxCalc * v) / maxExp);
  };

  return (
    <div className="container">
      <nav className="sidebar">
        <div className="sidebar-header">
          <h1>ResRAM</h1>
          <span className="status-badge">{status}</span>
        </div>
        
        <div className="sidebar-section scrollable">
          <div className="section-header">
            <h3><Settings size={16} /> Parameters</h3>
          </div>
          {config && (
            <div className="params-grid">
              <div className="param-row">
                <label>Gamma</label>
                <input type="number" value={config.gamma} onChange={e => setConfig({...config, gamma: parseFloat(e.target.value)})} />
                <input type="checkbox" checked={fitSwitches.gamma} onChange={() => setFitSwitches({...fitSwitches, gamma: !fitSwitches.gamma})} title="Fit?" />
              </div>
              <div className="param-row">
                <label>Theta</label>
                <input type="number" value={config.theta} onChange={e => setConfig({...config, theta: parseFloat(e.target.value)})} />
                <input type="checkbox" checked={fitSwitches.theta} onChange={() => setFitSwitches({...fitSwitches, theta: !fitSwitches.theta})} title="Fit?" />
              </div>
              <div className="param-row">
                <label>E0</label>
                <input type="number" value={config.e0} onChange={e => setConfig({...config, e0: parseFloat(e.target.value)})} />
                <input type="checkbox" checked={fitSwitches.e0} onChange={() => setFitSwitches({...fitSwitches, e0: !fitSwitches.e0})} title="Fit?" />
              </div>
              <div className="param-row">
                <label>Kappa</label>
                <input type="number" value={config.kappa} step="0.01" onChange={e => setConfig({...config, kappa: parseFloat(e.target.value)})} />
                <input type="checkbox" checked={fitSwitches.kappa} onChange={() => setFitSwitches({...fitSwitches, kappa: !fitSwitches.kappa})} title="Fit?" />
              </div>
              <div className="param-row">
                <label>Trans. M</label>
                <input type="number" value={config.m} step="0.1" onChange={e => setConfig({...config, m: parseFloat(e.target.value)})} />
                <input type="checkbox" checked={fitSwitches.m} onChange={() => setFitSwitches({...fitSwitches, m: !fitSwitches.m})} title="Fit?" />
              </div>
              <div className="param-row mt-2">
                <label>Max Fit Step</label>
                <input type="number" value={maxEval} onChange={e => setMaxEval(parseInt(e.target.value))} />
              </div>
                <div className="param-row">
                  <label>Algorithm</label>
                  <select value={fitAlgorithm} onChange={e => setFitAlgorithm(e.target.value)}>
                    <option value="powell">Powell (Praxis)</option>
                    <option value="cobyla">COBYLA</option>
                    <option value="bobyqa">BOBYQA</option>
                    <option value="newuoa">NEWUOA</option>
                    <option value="newuoa_bound">NEWUOA Bound</option>
                    <option value="neldermead">Nelder-Mead</option>
                    <option value="sbplx">SBPLX (Subplex)</option>
                  </select>
                </div>
              <div className="param-row">
                <label>UI Refresh Step</label>
                <input type="number" value={refreshStep} onChange={e => setRefreshStep(parseInt(e.target.value))} />
              </div>
            </div>
          )}

          <div className="section-header mt-4">
            <h3><List size={16} /> Vibrational Modes</h3>
          </div>
          <div className="modes-table">
            <table>
              <thead>
                <tr><th>Freq</th><th>Delta</th><th>Fit</th></tr>
              </thead>
              <tbody>
                {modes.map((m, i) => (
                  <tr key={i}>
                    <td>{m.frequency.toFixed(0)}</td>
                    <td>
                      <input 
                        type="number" 
                        step="0.01" 
                        value={m.displacement} 
                        onChange={e => updateMode(i, parseFloat(e.target.value))} 
                      />
                    </td>
                    <td>
                      <input type="checkbox" checked={fitSwitches.modes[i]} onChange={() => toggleModeFit(i)} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="sidebar-actions">
          <div className="action-row">
            <button onClick={openFolder} title="Open Workspace"><FolderOpen size={18} /></button>
            <button onClick={handleSave} disabled={!config} title="Save Changes"><Save size={18} /></button>
          </div>
          <button onClick={runCalc} disabled={!config}><Play size={18} /> Run Calc</button>
          <button className="primary" onClick={startFit} disabled={!config || isFitting}><Activity size={18} /> Start Fit</button>
        </div>
      </nav>

      <main className="content">
        <div className="plot-container">
          <Plot
            data={[
              ...(result ? [
                {
                  x: result.conv_el,
                  y: result.abs_cross,
                  type: 'scatter' as const,
                  name: 'Abs (Calc)',
                  line: { color: 'blue' }
                },
                {
                  x: result.conv_el,
                  y: result.fl_cross,
                  type: 'scatter' as const,
                  name: 'FL (Calc)',
                  line: { color: 'red' }
                },
                ...(result.abs_exp ? [{
                  x: result.conv_el,
                  y: result.abs_exp,
                  type: 'scatter' as const,
                  name: 'Abs (Exp)',
                  line: { color: 'blue', dash: 'dash' as const }
                }] : []),
                ...(result.fl_exp ? [{
                  x: result.conv_el,
                    y: normalizedFlExp(result) ?? result.fl_exp,
                  type: 'scatter' as const,
                  name: 'FL (Exp)',
                  line: { color: 'red', dash: 'dash' as const }
                }] : [])
              ] : [])
            ]}
            layout={{ 
              title: { text: 'Absorption & Fluorescence' }, 
              autosize: true,
              margin: { t: 40, r: 20, b: 40, l: 60 },
              xaxis: { title: { text: 'Wavenumber (cm⁻¹)' } },
                yaxis: { title: { text: 'Cross Section (Å²/Molecule)' } }
            }}
            useResizeHandler={true}
            style={{ width: "100%", height: "100%" }}
          />
        </div>
        
        <div className="plot-row">
          <div className="plot-container small">
            <Plot
              data={[
                  ...(result ? result.raman_cross.map((rc, i) => ({
                    x: result.conv_el,
                    y: rc,
                    type: 'scatter' as const,
                    mode: 'lines' as const,
                    name: `${modes[i].frequency.toFixed(0)} cm⁻¹`,
                    line: { color: modeColor(i) }
                  })) : []),
                  ...(result && result.profs_exp ? result.profs_exp.map((row, j) => {
                      const xPts = result.rp_indices
                        .filter((idx) => idx >= 0 && idx < result.conv_el.length)
                        .map((idx) => result.conv_el[idx]);
                    const yPts = row.slice(0, xPts.length);
                    return {
                      x: xPts.slice(0, yPts.length),
                      y: yPts,
                      type: 'scatter' as const,
                      mode: 'markers' as const,
                      name: `${modes[j]?.frequency?.toFixed(0) ?? j} cm⁻¹ (Exp)`,
                      marker: { color: modeColor(j), size: 7, symbol: 'circle-open' as const },
                      showlegend: false,
                    };
                  }) : [])
              ]}
              layout={{ 
                title: { text: 'REPs' }, 
                autosize: true,
                margin: { t: 40, r: 20, b: 40, l: 60 },
                  xaxis: { title: { text: 'Excitation Wavenumber (cm⁻¹)' } },
                  yaxis: { title: { text: 'Raman Cross Section (10^-14 Å²/Molecule)' } },
                  showlegend: true,
                  legend: { orientation: 'h', y: -0.2 }
              }}
              useResizeHandler={true}
              style={{ width: "100%", height: "100%" }}
            />
          </div>
          <div className="plot-container small">
            <Plot
              data={[
                ...(result ? result.raman_spec.map((rs, i) => ({
                  x: result.rshift,
                  y: rs,
                  type: 'scatter' as const,
                  name: `${rpumps[i]} cm⁻¹`
                })) : [])
              ]}
              layout={{ 
                title: { text: 'Raman Spectra' }, 
                autosize: true,
                margin: { t: 40, r: 20, b: 40, l: 60 },
                xaxis: { title: { text: 'Raman Shift (cm⁻¹)' } },
                  yaxis: { title: { text: 'Raman Cross Section (10^-14 Å²/Molecule)' } },
                showlegend: false
              }}
              useResizeHandler={true}
              style={{ width: "100%", height: "100%" }}
            />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
