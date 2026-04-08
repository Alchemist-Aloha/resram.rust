import { useState, useEffect } from "react";
import Plot from "react-plotly.js";
import { invoke } from "@tauri-apps/api/core";
import { Settings, Play, Activity, FolderOpen, Save } from "lucide-react";
import "./App.css";

interface ResRamConfig {
  gamma: number;
  theta: number;
  e0: number;
  kappa: number;
  m: number;
  n: number;
  temp: number;
}

function App() {
  const [config, setConfig] = useState<ResRamConfig | null>(null);
  const [dir, setDir] = useState("");
  const [status, setStatus] = useState("Ready");

  async function openFolder() {
    // In a real Tauri app, you'd use the dialog plugin. 
    // For now, we'll manually enter the path or use a placeholder.
    const path = prompt("Enter data folder path:", ".");
    if (path) {
      try {
        const loadedConfig = await invoke<ResRamConfig>("load_data", { dir: path });
        setConfig(loadedConfig);
        setDir(path);
        setStatus(`Loaded data from ${path}`);
      } catch (e) {
        setStatus(`Error: ${e}`);
      }
    }
  }

  return (
    <div className="container">
      <nav className="sidebar">
        <div className="sidebar-header">
          <h1>ResRAM</h1>
          <span className="status-badge">{status}</span>
        </div>
        
        <div className="sidebar-section">
          <h3><Settings size={16} /> Global Params</h3>
          {config && (
            <div className="params-grid">
              <label>Gamma: <input type="number" value={config.gamma} readOnly /></label>
              <label>Theta: <input type="number" value={config.theta} readOnly /></label>
              <label>E0: <input type="number" value={config.e0} readOnly /></label>
              <label>Temp: <input type="number" value={config.temp} readOnly /></label>
            </div>
          )}
        </div>

        <div className="sidebar-actions">
          <button onClick={openFolder}><FolderOpen size={18} /> Open Folder</button>
          <button disabled={!config}><Play size={18} /> Run Calc</button>
          <button className="primary" disabled={!config}><Activity size={18} /> Start Fit</button>
        </div>
      </nav>

      <main className="content">
        <div className="plot-container">
          <Plot
            data={[
              {
                x: [1, 2, 3],
                y: [2, 6, 3],
                type: 'scatter',
                mode: 'lines+markers',
                marker: { color: 'red' },
                name: 'Absorption'
              }
            ]}
            layout={{ title: 'Spectra', autosize: true }}
            useResizeHandler={true}
            style={{ width: "100%", height: "100%" }}
          />
        </div>
        <div className="plot-container">
          <Plot
            data={[]}
            layout={{ title: 'Raman Excitation Profiles', autosize: true }}
            useResizeHandler={true}
            style={{ width: "100%", height: "100%" }}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
