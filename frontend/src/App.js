import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './App.css';

const API_URL = 'http://127.0.0.1:5000';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [chartLayout, setChartLayout] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const onDrop = useCallback(acceptedFiles => {
    setFile(acceptedFiles[0]);
    setResult(null);
    setChartData(null);
    setError('');
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
  });

  const handleCsvSubmit = async () => {
    if (!file) {
      setError('Please upload a CSV file first.');
      return;
    }

    setIsLoading(true);
    setError('');
    setResult(null);
    setChartData(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/predict_csv`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      setResult(response.data.summary);
      setChartData(response.data.chart_data);
      setChartLayout(response.data.chart_layout);
    } catch (err) {
      const errorMessage =
        err.response?.data?.error || 'An unknown error occurred during prediction.';
      setError(errorMessage);
      console.error('Backend error:', err.response?.data || err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Exoplanet Detection</h1>
        <p>Using your custom-trained XGBoost Model</p>
      </header>

      <main>
        <div className="card upload-card">
          <h2>Upload KOI Data File</h2>
          <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
            <input {...getInputProps()} />
            {file ? (
              <p>{file.name}</p>
            ) : (
              <p>Drag & drop your 'KOI_cumulative...' CSV file here</p>
            )}
          </div>
          <button onClick={handleCsvSubmit} disabled={isLoading || !file}>
            {isLoading ? 'Analyzing...' : 'Analyze Data'}
          </button>
        </div>

        {error && (
          <div className="card error-card">
            <p className="error-message">Error: {error}</p>
          </div>
        )}

        {result && (
          <div className="card results-card">
            <h2>Analysis Results</h2>
            <p className="summary">{result}</p>
            {chartData && chartLayout && (
              <div className="chart-container">
                <Plot
                  data={[chartData]}
                  layout={{
                    ...chartLayout,
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: '#1e1e3c',
                    font: { color: 'white' },
                  }}
                  config={{ responsive: true }}
                  className="plotly-chart"
                />
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
