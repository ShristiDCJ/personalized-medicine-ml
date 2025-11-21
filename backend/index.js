require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { PythonShell } = require('python-shell');
const path = require('path');

const app = express();
const port = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok' });
});

// Prediction endpoint
app.post('/api/predict', async (req, res) => {
  try {
    const { geneticData, clinicalData } = req.body;

    // Configure Python Shell options
    let options = {
      mode: 'text',
      pythonPath: process.env.PYTHON_PATH || 'python',
      scriptPath: path.join(__dirname, 'model'),
      args: [JSON.stringify({ geneticData, clinicalData })]
    };

    // Run the prediction
    PythonShell.run('predict.py', options).then(results => {
      const prediction = JSON.parse(results[0]);
      res.json(prediction);
    }).catch(err => {
      console.error('Prediction error:', err);
      res.status(500).json({ error: 'Error making prediction' });
    });
  } catch (error) {
    console.error('Server error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
}).on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`Port ${port} is already in use. Please try a different port.`);
    process.exit(1);
  } else {
    console.error('Server error:', err);
  }
});