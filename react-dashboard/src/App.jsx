import React, { useState } from 'react';
import { INITIAL_DOCUMENTS, BASELINE_METRICS } from './models/casData';
import { runCASPipeline } from './services/casEngine';
import './App.css';

export default function App() {
  const [documents] = useState(INITIAL_DOCUMENTS);
  const [trustThreshold, setTrustThreshold] = useState(0.40);
  const [processedResults, setProcessedResults] = useState(null);

  const handleExecute = () => {
    const results = runCASPipeline(documents, trustThreshold);
    setProcessedResults(results);
  };

  return (
    <div className="container">
      <h1>Conflict-Aware Synthesis (CAS) Dashboard</h1>
      <p className="subtitle">Pre-Generation Verification Layer for RAG Architecture</p>

      <div className="card">
        <label><strong>Trust Threshold: </strong>{trustThreshold}</label>
        <input 
          type="range" min="0.10" max="0.80" step="0.05" 
          value={trustThreshold} 
          onChange={(e) => setTrustThreshold(parseFloat(e.target.value))}
        />
        <button onClick={handleExecute} className="btn">Run CAS Verification</button>
      </div>

      {processedResults && (
        <div className="section">
          <h2>1. Conflict Graph Adjudication</h2>
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Source</th>
                <th>Content</th>
                <th>Trust Score</th>
                <th>Penalty</th>
                <th>Net Weight</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {processedResults.map((doc) => (
                <tr key={doc.id}>
                  <td>{doc.id}</td>
                  <td>{doc.source}</td>
                  <td>{doc.content}</td>
                  <td>{doc.trustScore.toFixed(2)}</td>
                  <td className="penalty">-{doc.penaltyScore.toFixed(2)}</td>
                  <td><strong>{doc.finalWeight.toFixed(2)}</strong></td>
                  <td className={doc.isFiltered ? "status-rejected" : "status-retained"}>
                    {doc.isFiltered ? 'REJECTED' : 'RETAINED'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="card context-box">
            <h3>Synthesized Reconciled Context (C_verified):</h3>
            <p>{processedResults.filter(d => !d.isFiltered).map(d => `[${d.source}]: "${d.content}"`).join(' | ')}</p>
          </div>
        </div>
      )}

      <div className="section">
        <h2>2. Benchmark Comparisons (CPI Performance)</h2>
        <table className="data-table">
          <thead>
            <tr>
              <th>Framework</th>
              <th>Accuracy (%)</th>
              <th>Hallucination (%)</th>
              <th>Contradiction (%)</th>
              <th>Consistency (%)</th>
              <th>CPI Score</th>
            </tr>
          </thead>
          <tbody>
            {BASELINE_METRICS.map((row) => (
              <tr key={row.name} className={row.name.includes('CAS') ? 'highlight-row' : ''}>
                <td><strong>{row.name}</strong></td>
                <td>{row.accuracy}</td>
                <td>{row.hallucination}</td>
                <td>{row.contradiction}</td>
                <td>{row.consistency}</td>
                <td><strong>{row.cpi}</strong></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
