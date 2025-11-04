import React, { useState } from 'react';
import { predictFraud } from '../services/api';
import ResultCard from './ResultCard';

const TransactionForm = () => {
  const [formData, setFormData] = useState({
    step: '',
    type: '',
    amount: '',
    oldbalanceOrg: '',
    newbalanceOrig: '',
    oldbalanceDest: '',
    newbalanceDest: ''
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const data = await predictFraud(formData);
      setResult(data);
    } catch (err) {
      alert('Backend not reachable. Make sure FastAPI is running.');
    }
  };

  return (
    <div style={{ width: '400px', margin: 'auto', textAlign: 'center' }}>
      <h2>💳 Fraud Detection</h2>
      <form onSubmit={handleSubmit}>
        {Object.keys(formData).map((key) => (
          <div key={key}>
            <label>{key}:</label><br />
            <input
              type="text"
              name={key}
              value={formData[key]}
              onChange={handleChange}
              required
            />
            <br /><br />
          </div>
        ))}
        <button type="submit">Predict</button>
      </form>

      {result && <ResultCard result={result} />}
    </div>
  );
};

export default TransactionForm;
