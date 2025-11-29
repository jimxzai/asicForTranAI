# Website Documentation

This directory contains the GitHub Pages website for the world's first 3.5-bit Fortran ASIC AI implementation.

## Preview Locally

Open these files in your browser to preview before publishing:

```bash
# Homepage
open docs/index.html

# Technical documentation
open docs/technical.html
```

Or use Python's built-in HTTP server:

```bash
cd docs
python3 -m http.server 8000
# Then visit: http://localhost:8000
```

## Publish to GitHub Pages

See `DEPLOY.md` for complete instructions.

Quick version:
1. Push to GitHub
2. Go to Settings → Pages
3. Select branch: `main`, folder: `/docs`
4. Save and wait 2-3 minutes

Your site will be live at: `https://jimxzai.github.io/asicForTranAI/`

## Files

- `index.html` - Main landing page with performance metrics
- `technical.html` - Comprehensive technical documentation
- `_config.yml` - Jekyll configuration
- `.nojekyll` - Tells GitHub Pages to serve raw HTML
- `DEPLOY.md` - Deployment guide

## Customization

Before publishing, replace `jimxzai` with your GitHub username in:
- All HTML files (search for "jimxzai")
- `_config.yml`
- `../README.md`

Quick find & replace:
```bash
cd ..
find . -type f \( -name "*.html" -o -name "*.md" -o -name "*.yml" \) -exec sed -i '' 's/jimxzai/your_username/g' {} +
```

---

Created 2025-11-28 by Jim Xiao & Claude Code (Anthropic)
