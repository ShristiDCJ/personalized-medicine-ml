# Deployment Guide

This guide explains how to deploy the Personalized Medicine Classification system, which consists of a FastAPI backend and a Next.js frontend.

## Backend Deployment (Render)

1. Create a new Web Service on Render
   - Connect your GitHub repository
   - Select the backend directory
   - Use the following settings:
     - Environment: Python 3.9
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

2. Set Environment Variables
   - `DEBUG`: false
   - `API_KEY`: [Generate a secure API key]
   - `ALLOWED_ORIGINS`: [Your frontend domain]
   - `MODEL_DIR`: model
   - `LOG_LEVEL`: INFO

3. Deploy Model Files
   - Ensure the following files are in the `model` directory:
     - `model.pth`
     - `text_processor.pkl`
     - `le_gene.pkl`
     - `le_variation.pkl`

## Frontend Deployment (Vercel)

1. Create a new project on Vercel
   - Connect your GitHub repository
   - Select the frontend directory
   - Framework Preset: Next.js

2. Configure Environment Variables
   - `NEXT_PUBLIC_API_URL`: [Your backend URL]
   - `NEXT_PUBLIC_API_KEY`: [Same API key as backend]

3. Deploy Settings
   - Build Command: `npm run build`
   - Output Directory: `.next`
   - Install Command: `npm install`

## Local Development Setup

### Backend
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create `.env` file:
   ```
   DEBUG=True
   API_KEY=your-development-key
   ALLOWED_ORIGINS=http://localhost:3000
   MODEL_DIR=model
   LOG_LEVEL=DEBUG
   ```

4. Run the server:
   ```bash
   uvicorn app:app --reload
   ```

### Frontend
1. Install dependencies:
   ```bash
   npm install
   ```

2. Create `.env.local` file:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   NEXT_PUBLIC_API_KEY=your-development-key
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

## API Documentation

The API provides the following endpoints:

- POST `/predict`
  - Headers:
    - `X-API-Key`: API key for authentication
  - Body:
    ```json
    {
      "gene": "string",
      "variation": "string",
      "text": "string"
    }
    ```
  - Response:
    ```json
    {
      "predicted_class": number,
      "class_probabilities": {
        "Class_1": number,
        ...
        "Class_9": number
      },
      "gene": "string",
      "variation": "string"
    }
    ```

- GET `/health`
  - Response: `{"status": "healthy"}`

## Security Considerations

1. Always use HTTPS in production
2. Keep API keys secure and rotate them periodically
3. Configure CORS to allow only your frontend domain
4. Monitor logs for suspicious activity
5. Regularly update dependencies

## Monitoring

1. Use Render's built-in monitoring for backend
2. Use Vercel Analytics for frontend
3. Check application logs in `app.log`
4. Monitor API response times and error rates

## Troubleshooting

1. If predictions fail:
   - Check model files are present in the correct directory
   - Verify API key is correctly set
   - Check application logs for errors

2. If frontend can't connect:
   - Verify API URL is correct
   - Check CORS configuration
   - Verify API key is being sent correctly

3. Performance issues:
   - Check resource usage on Render
   - Monitor memory usage for model loading
   - Consider caching frequent predictions