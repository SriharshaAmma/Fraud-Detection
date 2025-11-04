import axios from 'axios';

const API_URL = "http://localhost:8000/api/v1/predict";

export const predictFraud = async (transaction) => {
  try {
    const response = await axios.post(API_URL, transaction);
    return response.data;
  } catch (error) {
    console.error('Error connecting to backend:', error);
    throw error;
  }
};
