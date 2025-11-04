import React from 'react';

const ResultCard = ({ result }) => {
  return (
    <div style={{
      marginTop: '20px',
      padding: '15px',
      border: '1px solid #ccc',
      borderRadius: '8px'
    }}>
      <h3>Prediction Result</h3>
      <p><strong>Prediction:</strong> {result.prediction}</p>
      <p><strong>Probability:</strong> {(result.probability * 100).toFixed(2)}%</p>
    </div>
  );
};

export default ResultCard;
