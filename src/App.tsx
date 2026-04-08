import { useState, useEffect } from "react";
import Plot from "react-plotly.js";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
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

interface VibrationalMode {
  frequency: number;
  displacement: number;
}

interface SimulationResult {
  abs_cross: number[];
  fl_cross: number[];
  raman_cross: number[][];
  raman_spec: number[][];
  conv_el: number[];
  rshift: number[];
}

interface ProgressPayload {
  iteration: number;
  loss: number;
  parameters: number[];
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
  const [, setProgress] = useState<ProgressPayload | null>(null);

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
    });
    
    // Initial loading from sample_data
    loadFolder("sample_data");

    return () => {
      unlisten.then(f => f());
    };
  }, []);

  async function loadFolder(path: string) {
    try {
      const loadedConfig = await invoke<ResRamConfig>("load_data", { dir: path });
      const [loadedModes, loadedRpumps] = await invoke<[VibrationalMode[], number[]]>("load_vibrational_data_cmd", { dir: path });
      
      setConfig(loadedConfig);
      setModes(loadedModes);
      setRpumps(loadedRpumps);
      setFitSwitches({
        ...fitSwitches,
        modes: new Array(loadedModes.length).fill(true)
      });
      setDir(path);
      setStatus(`Loaded data from ${path}`);
    } catch (e) {
      setStatus(`Error: ${e}`);
    }
  }

  async function openFolder() {
    const path = prompt("Enter data folder path:", dir || ".");
    if (path) {
      loadFolder(path);
    }
  }

  async function runCalc() {
    if (!config || modes.length === 0) return;
    setStatus("Calculating...");
    try {
      const res = await invoke<SimulationResult>("run_calculation", { config, modes, rpumps });
      setResult(res);
      setStatus("Calculation complete");
    } catch (e) {
      setStatus(`Error: ${e}`);
    }
  }

  async function startFit() {
    if (!config || modes.length === 0) return;
    setStatus("Starting fit...");
    try {
      const fitIndices = modes.map((_, i) => i).filter(i => fitSwitches.modes[i]);
      const resConfig = await invoke<ResRamConfig>("run_fit", {
        dir,
        config,
        modes,
        fitIndices,
        fitGamma: fitSwitches.gamma,
        fitM: fitSwitches.m,
        fitTheta: fitSwitches.theta,
        fitKappa: fitSwitches.kappa,
        fitE0: fitSwitches.e0,
        algorithmName: "powell",
        maxEval: maxEval,
        refreshStep: refreshStep,
      });
      setConfig(resConfig);
      setStatus("Fitting complete");
      runCalc();
    } catch (e) {
      setStatus(`Error: ${e}`);
    }
  }

  async function handleSave() {
    if (!config || !dir) return;
    setStatus("Saving...");
    try {
      const folderName = await invoke<string>("save_data", { dir, config, modes });
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
          <button className="primary" onClick={startFit} disabled={!config}><Activity size={18} /> Start Fit</button>
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
                }
              ] : [])
            ]}
            layout={{ 
              title: { text: 'Absorption & Fluorescence' }, 
              autosize: true,
              margin: { t: 40, r: 20, b: 40, l: 60 },
              xaxis: { title: { text: 'Wavenumber (cm⁻¹)' } },
              yaxis: { title: { text: 'Cross Section' } }
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
                  name: `${modes[i].frequency.toFixed(0)}`
                })) : [])
              ]}
              layout={{ 
                title: { text: 'REPs' }, 
                autosize: true,
                margin: { t: 40, r: 20, b: 40, l: 60 },
                xaxis: { title: { text: 'Excitation (cm⁻¹)' } },
                showlegend: false
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
