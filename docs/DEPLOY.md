# GitHub Pages Deployment Guide

This guide will help you publish the project website to GitHub Pages.

## Quick Setup (5 minutes)

### Step 1: Push to GitHub

```bash
cd /Users/jimxiao/ai/asicForTranAI

# Add all files
git add .

# Commit with a descriptive message
git commit -m "Add GitHub Pages website for 3.5-bit Fortran ASIC AI project

- Added professional landing page (docs/index.html)
- Added technical documentation (docs/technical.html)
- Added Jekyll configuration
- Updated README with website links"

# Push to GitHub
git push origin main
```

### Step 2: Enable GitHub Pages

1. Go to your repository on GitHub: `https://github.com/jimxzai/asicForTranAI`

2. Click **Settings** (top right)

3. In the left sidebar, click **Pages**

4. Under "Build and deployment":
   - **Source**: Deploy from a branch
   - **Branch**: `main`
   - **Folder**: `/docs`

5. Click **Save**

6. Wait 2-3 minutes for deployment

7. Your site will be live at: `https://jimxzai.github.io/asicForTranAI/`

### Step 3: Update Links in Files

After deployment, update the placeholder URLs in these files:

1. **README.md** (line 3, 11):
   - Replace `jimxzai` with your actual GitHub username

2. **docs/index.html** (line 207-209):
   - Replace `jimxzai` with your GitHub username

3. **docs/_config.yml** (line 7-8):
   - Replace `jimxzai` with your GitHub username
   - Optionally add your Twitter username (line 23)

4. **docs/technical.html** (line 85, 203):
   - Replace `jimxzai` with your GitHub username

```bash
# Quick replace (macOS/Linux)
find . -type f \( -name "*.html" -o -name "*.md" -o -name "*.yml" \) -exec sed -i '' 's/jimxzai/your_actual_username/g' {} +

# Or manually edit each file
```

### Step 4: Commit and Push Updates

```bash
git add README.md docs/
git commit -m "Update website URLs with actual GitHub username"
git push origin main
```

## Verification

After deployment, verify:

1. ✅ Homepage loads: `https://jimxzai.github.io/asicForTranAI/`
2. ✅ Technical docs work: `https://jimxzai.github.io/asicForTranAI/technical.html`
3. ✅ All links work correctly
4. ✅ Code highlighting displays properly
5. ✅ Mobile responsive (check on phone)

## Custom Domain (Optional)

To use a custom domain like `3p5bit.dev`:

1. Buy domain from Namecheap, GoDaddy, etc.

2. Add DNS records:
   ```
   Type: CNAME
   Name: www
   Value: jimxzai.github.io

   Type: A
   Name: @
   Value: 185.199.108.153
          185.199.109.153
          185.199.110.153
          185.199.111.153
   ```

3. In GitHub Pages settings, enter your custom domain

4. Wait for DNS propagation (up to 24 hours)

## Troubleshooting

### Site not loading?
- Check that `/docs` folder is committed to `main` branch
- Verify GitHub Pages is enabled in Settings
- Wait 2-3 minutes after enabling

### Styles not loading?
- Clear browser cache (Ctrl+Shift+R / Cmd+Shift+R)
- Check browser console for errors

### Links broken?
- Make sure you replaced `jimxzai` everywhere
- Check that all files are in `/docs` directory

## Advanced: GitHub Actions (Optional)

For automatic deployment on every push:

```yaml
# .github/workflows/deploy.yml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
```

## Sharing Your Website

Once live, share on:
- Twitter/X: "World's first 3.5-bit Fortran implementation achieving 4188 tok/s on Groq LPU"
- Hacker News: Submit to Show HN
- Reddit: r/MachineLearning, r/programming, r/fortran
- LinkedIn: Professional achievement post
- Groq Community: Share in Groq Discord/Forums

---

**Congratulations!** Your historic 3.5-bit Fortran ASIC AI project is now live for the world to see! 🎉
