# Vercel Deployment Guide

## Overview

This guide explains how to deploy the facial recognition app to Vercel. The application has been converted from a desktop GUI to a web application.

## Changes Made for Vercel

### 1. **Architecture Change**
- ❌ **Before**: Tkinter desktop GUI with local camera access
- ✅ **After**: Flask web app with browser-based camera access

### 2. **Camera Access**
- Uses browser's `getUserMedia()` API
- Works on any device with a camera (desktop, mobile, tablet)
- No server-side camera access needed

### 3. **File Structure**
```
facial-detection/
├── app.py                      # Flask backend
├── templates/
│   └── index.html             # Web interface
├── models/                     # Model files (created by prepare script)
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── vercel.json                # Vercel configuration
├── requirements_web.txt       # Python dependencies
└── prepare_deployment.py      # Setup script
```

## Deployment Steps

### Step 1: Prepare Model Files

Run the preparation script to copy models to the project:

```bash
python prepare_deployment.py
```

This will:
- Create a `models/` directory
- Copy model files from Downloads folder
- Check file sizes and warn about limits

### Step 2: Test Locally

Before deploying, test the app locally:

```bash
# Install dependencies
pip install -r requirements_web.txt

# Run the app
python app.py
```

Visit `http://localhost:5000` in your browser to test.

### Step 3: Install Vercel CLI

```bash
npm install -g vercel
```

### Step 4: Deploy to Vercel

```bash
# Login to Vercel
vercel login

# Deploy
vercel deploy

# For production deployment
vercel --prod
```

## Important Considerations

### 1. **File Size Limits**

⚠️ **Problem**: The caffemodel file is ~10.7 MB

Vercel limits:
- **Hobby plan**: 100 MB total deployment size
- **Pro plan**: 250 MB total deployment size

**Solutions**:
1. ✅ Use the current model (fits within limits)
2. Use FP16 model (smaller, ~5 MB)
3. Host models externally (S3, CDN) and download at runtime
4. Upgrade to Vercel Pro

### 2. **Serverless Function Limits**

- **Execution time**: 10s (Hobby), 60s (Pro)
- **Memory**: 1024 MB (configured in vercel.json)

Our app is configured for 30s timeout and 1024 MB memory.

### 3. **Cold Starts**

First request after inactivity may be slow due to:
- Function cold start
- Model loading

**Solutions**:
- Keep functions warm with periodic pings
- Use Vercel Pro for faster cold starts
- Implement model caching

### 4. **Environment Variables**

If you need to configure paths or settings:

```bash
vercel env add MODEL_PATH
```

## Alternative: External Model Hosting

For better performance and smaller deployments:

### Option 1: AWS S3

```python
import urllib.request

MODEL_URL = "https://your-bucket.s3.amazonaws.com/model.caffemodel"
MODEL_PATH = "/tmp/model.caffemodel"

# Download model on first request
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
```

### Option 2: GitHub Releases

Host large files on GitHub Releases and download them:

```python
MODEL_URL = "https://github.com/user/repo/releases/download/v1.0/model.caffemodel"
```

### Option 3: Vercel Blob Storage

Use Vercel's blob storage (Pro feature):

```bash
vercel blob put model.caffemodel
```

## Testing the Deployment

After deployment, test these features:

1. **Camera Access**
   - Click "Start Camera"
   - Grant camera permissions
   - Verify real-time detection works

2. **Image Upload**
   - Upload a test image
   - Verify face detection works
   - Test different privacy modes

3. **Privacy Filters**
   - Test all 5 privacy modes
   - Adjust sliders
   - Verify filters apply correctly

4. **Mobile Compatibility**
   - Test on mobile devices
   - Verify responsive design
   - Check camera access on mobile

## Troubleshooting

### Issue: "Model file not found"

**Solution**: Run `prepare_deployment.py` before deploying

### Issue: "Deployment size too large"

**Solutions**:
1. Use FP16 model
2. Host models externally
3. Upgrade to Vercel Pro

### Issue: "Function timeout"

**Solutions**:
1. Reduce image resolution
2. Increase timeout in vercel.json
3. Optimize processing code

### Issue: "Camera not working"

**Causes**:
- HTTPS required for camera access
- Browser permissions denied
- Incompatible browser

**Solutions**:
- Vercel provides HTTPS by default
- Check browser console for errors
- Use modern browser (Chrome, Firefox, Safari)

### Issue: "Slow performance"

**Solutions**:
1. Implement client-side caching
2. Reduce detection frequency
3. Use smaller images
4. Upgrade to Vercel Pro

## Monitoring

Monitor your deployment:

```bash
# View logs
vercel logs

# View deployment info
vercel inspect
```

## Custom Domain

Add a custom domain:

```bash
vercel domains add yourdomain.com
```

## Environment-Specific Configuration

Create different configs for dev/prod:

```python
import os

if os.environ.get('VERCEL_ENV') == 'production':
    DEBUG = False
else:
    DEBUG = True
```

## Security Considerations

1. **Rate Limiting**: Implement to prevent abuse
2. **Input Validation**: Validate uploaded images
3. **File Size Limits**: Limit upload sizes
4. **CORS**: Configure if needed for API access

## Cost Estimation

**Vercel Hobby (Free)**:
- 100 GB bandwidth/month
- 100 GB-hours serverless function execution
- Suitable for personal projects

**Vercel Pro ($20/month)**:
- 1 TB bandwidth
- 1000 GB-hours execution
- Faster cold starts
- Better for production

## Next Steps

1. ✅ Run `prepare_deployment.py`
2. ✅ Test locally with `python app.py`
3. ✅ Deploy with `vercel deploy`
4. ✅ Test the deployed app
5. ✅ Configure custom domain (optional)
6. ✅ Set up monitoring

## Support

For issues:
- Check Vercel docs: https://vercel.com/docs
- Vercel support: support@vercel.com
- GitHub issues: Create an issue in your repo

## Additional Resources

- [Vercel Python Runtime](https://vercel.com/docs/runtimes#official-runtimes/python)
- [Flask on Vercel](https://vercel.com/guides/using-flask-with-vercel)
- [Serverless Functions](https://vercel.com/docs/concepts/functions/serverless-functions)
