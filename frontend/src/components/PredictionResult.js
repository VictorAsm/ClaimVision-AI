import React from 'react';
import styles from './../styles.js';

function PredictionResult({ data }) {

  return (
    <div style={styles.resultContainer}>
      <h2 style={styles.title}>Resultados del Análisis</h2>
      <div style={styles.jsonContainer}>
        <h3 style={styles.subtitle}>Datos de Predicción</h3>
        <h3>{data.msg}</h3>
        <pre style={styles.pre}>
          <ul>
            <li>Damaged: {parseFloat(data.data.damaged_scores.damaged.toFixed(2))}%</li>
            <li>Good: {parseFloat(data.data.damaged_scores.good.toFixed(2))}%</li>
            <li>Unknown: {data.data.damaged_scores.unknown ? parseFloat(data.data.damaged_scores.unknown.toFixed(2)) : ''}%</li>
          </ul>
        </pre>
        <pre style={styles.pre}>
          {data.data.detected_objects.map(item => {
            return <ul>
              <li>Label: {item.label}</li>
              <li>Score: {parseFloat(item.score.toFixed(2))}</li>
              <li>Box: {JSON.stringify(item.box)}</li>
            </ul>
          })}
        </pre>
      </div>
    </div>
  );
}

export default PredictionResult;
