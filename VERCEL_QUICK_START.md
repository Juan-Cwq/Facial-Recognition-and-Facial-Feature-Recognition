# 🚀 Vercel Quick Start

## TL;DR - Deploy in 3 Steps

```bash
# 1. Prepare models
python prepare_deployment.py

# 2. Test locally
python app.py

# 3. Deploy
vercel deploy --prod
```

## What Changed?

### ❌ Old (Desktop App)
- Tkinter GUI
- Local camera via OpenCV
- Can't deploy to web

### ✅ New (Web App)
- Flask web interface
- Browser camera via getUserMedia
- Deploys to Vercel

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Flask backend with face detection |
| `templates/index.html` | Web interface with camera access |
| `vercel.json` | Vercel configuration |
| `requirements_web.txt` | Python dependencies |
| `prepare_deployment.py` | Copies models to project |
| `DEPLOYMENT.md` | Full deployment guide |

## Important Notes

### ✅ What Works
- Real-time camera detection in browser
- Image upload and processing
- All 5 privacy filters
- Mobile compatible
- Responsive design

### ⚠️ Limitations
- Model file is ~10.7 MB (within Vercel limits)
- Cold starts may be slow (~2-3 seconds)
- Free tier: 100 GB bandwidth/month
- Function timeout: 30 seconds (configured)

### 💡 Tips
1. **Test locally first** - Always test with `python app.py` before deploying
2. **Check file sizes** - Run `prepare_deployment.py` to see warnings
3. **Monitor usage** - Check Vercel dashboard for bandwidth/execution time
4. **Use HTTPS** - Camera requires HTTPS (Vercel provides this automatically)

## Troubleshooting

### "Model not found"
```bash
python prepare_deployment.py
```

### "Deployment too large"
- Use FP16 model (smaller)
- Or host models externally (S3, CDN)

### "Camera not working"
- Check browser permissions
- Ensure HTTPS (Vercel provides)
- Try different browser

## Next Steps

1. ✅ Run `prepare_deployment.py`
2. ✅ Test at `http://localhost:5000`
3. ✅ Deploy with `vercel deploy`
4. ✅ Share your app URL!

## Need Help?

- 📖 Full guide: `DEPLOYMENT.md`
- 🐛 Issues: Check browser console
- 💬 Vercel docs: https://vercel.com/docs

## Estimated Costs

**Free Tier** (Hobby):
- ✅ Perfect for personal projects
- ✅ 100 GB bandwidth
- ✅ 100 GB-hours execution
- ✅ Unlimited deployments

**Pro** ($20/month):
- 🚀 10x more bandwidth
- 🚀 Faster cold starts
- 🚀 Better for production

Most users will be fine with the free tier!
