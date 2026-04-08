import { useState, useEffect } from "react";
import Plot from "react-plotly.js";
import { invoke } from "@tauri-apps/api/core";
import { Settings, Play, Activity, FolderOpen, Save, List } from "lucide-react";
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
}

interface VibrationalMode {
  frequency: number;
  displacement: number;
}

interface SimulationResult {
  abs_cross: number[];
  fl_cross: number[];
  raman_cross: number[][];
  conv_el: number[];
}

function App() {
  const [config, setConfig] = useState<ResRamConfig | null>(null);
  const [modes, setModes] = useState<VibrationalMode[]>([]);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [dir, setDir] = useState("");
  const [status, setStatus] = useState("Ready");

  async function openFolder() {
    const path = prompt("Enter data folder path:", ".");
    if (path) {
      try {
        const loadedConfig = await invoke<ResRamConfig>("load_data", { dir: path });
        // Also need modes
        const [loadedModes, _] = await invoke<[VibrationalMode[], number[]]>("load_vibrational_data_cmd", { dir: path });
        
        setConfig(loadedConfig);
        setModes(loadedModes);
        setDir(path);
        setStatus(`Loaded data from ${path}`);
      } catch (e) {
        setStatus(`Error: ${e}`);
      }
    }
  }

  async function runCalc() {
    if (!config || modes.length === 0) return;
    setStatus("Calculating...");
    try {
      const res = await invoke<SimulationResult>("run_calculation", { config, modes });
      setResult(res);
      setStatus("Calculation complete");
    } catch (e) {
      setStatus(`Error: ${e}`);
    }
  }

  const updateMode = (idx: number, delta: number) => {
    const newModes = [...modes];
    newModes[idx].displacement = delta;
    setModes(newModes);
  };

  return (
    <div className="container">
      <nav className="sidebar">
        <div className="sidebar-header">
          <h1>ResRAM</h1>
          <span className="status-badge">{status}</span>
        </div>
        
        <div className="sidebar-section scrollable">
          <h3><Settings size={16} /> Global Params</h3>
          {config && (
            <div className="params-grid">
              <label>Gamma: <input type="number" value={config.gamma} onChange={e => setConfig({...config, gamma: parseFloat(e.target.value)})} /></label>
              <label>Theta: <input type="number" value={config.theta} onChange={e => setConfig({...config, theta: parseFloat(e.target.value)})} /></label>
              <label>E0: <input type="number" value={config.e0} onChange={e => setConfig({...config, e0: parseFloat(e.target.value)})} /></label>
            </div>
          )}

          <h3><List size={16} /> Vibrational Modes</h3>
          <div className="modes-table">
            <table>
              <thead>
                <tr><th>Freq</th><th>Delta</th></tr>
              </thead>
              <tbody>
                {modes.map((m, i) => (
                  <tr key={i}>
                    <td>{m.frequency.toFixed(1)}</td>
                    <td>
                      <input 
                        type="number" 
                        step="0.01" 
                        value={m.displacement} 
                        onChange={e => updateMode(i, parseFloat(e.target.value))} 
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="sidebar-actions">
          <button onClick={openFolder} title="Open Workspace"><FolderOpen size={18} /> Open Folder</button>
          <button onClick={runCalc} disabled={!config}><Play size={18} /> Run Calc</button>
          <button className="primary" disabled={!config}><Activity size={18} /> Start Fit</button>
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
              title: 'Absorption & Fluorescence', 
              autosize: true,
              margin: { t: 40, r: 20, b: 40, l: 60 }
            }}
            useResizeHandler={true}
            style={{ width: "100%", height: "100%" }}
          />
        </div>
        <div className="plot-container">
          <Plot
            data={[
              ...(result ? result.raman_cross.map((rc, i) => ({
                x: result.conv_el,
                y: rc,
                type: 'scatter' as const,
                name: `${modes[i].frequency.toFixed(0)} cm⁻¹`
              })) : [])
            ]}
            layout={{ 
              title: 'Raman Excitation Profiles', 
              autosize: true,
              margin: { t: 40, r: 20, b: 40, l: 60 }
            }}
            useResizeHandler={true}
            style={{ width: "100%", height: "100%" }}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
