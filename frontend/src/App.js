import React, { useState } from 'react';
import AccidentReportForm from './components/AccidentReportForm';
import PredictionResult from './components/PredictionResult';
import styles from './styles.js';

function App() {
  const [predictionResult, setPredictionResult] = useState(null);

  return (
    <div style={styles.appContainer}>
      <h1 style={styles.mainTitle}>Sistema de Análisis de Siniestros</h1>
      <div style={styles.contentContainer}>
        <AccidentReportForm setPredictionResult={setPredictionResult} />
        {predictionResult && <PredictionResult data={predictionResult} />}
      </div>
    </div>
  );
}

export default App;
