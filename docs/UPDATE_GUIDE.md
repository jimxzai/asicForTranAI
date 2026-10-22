# Website Update Guide

> **Archived / unaudited (2026-10-22).** Figures in this file (70B in 19 GB, 4,188 tok/s, 247 SPARK proofs, 46% smaller than INT4) were not reproducible; see [CLAIMS_AUDIT_2026-10.md](CLAIMS_AUDIT_2026-10.md). Do not quote them.

This guide shows you how to update your website after the initial deployment.

## Quick Updates (Most Common)

### Update Homepage or Technical Docs

```bash
# 1. Edit the files
# Edit docs/index.html or docs/technical.html in your editor

# 2. Commit and push
git add docs/
git commit -m "Update website content"
git push

# 3. Wait 1-2 minutes for GitHub Pages to rebuild
```

Your changes will be live at https://jimxzai.github.io/asicForTranAI/

## Update Workflow (Using Token)

### Option 1: Quick Token Push (Recommended)

If you updated your token with `workflow` scope:

```bash
# 1. Make your changes to any files
git add .
git commit -m "Your commit message"

# 2. Push with your token
git push https://YOUR_TOKEN@github.com/jimxzai/asicForTranAI.git main
```

### Option 2: Set Up Token Credentials (One-time setup)

Store your token so you don't have to paste it every time:

```bash
# macOS/Linux: Store token in git credential manager
git config --global credential.helper osxkeychain  # macOS
# or
git config --global credential.helper store        # Linux

# First push will ask for credentials:
# Username: jimxzai
# Password: [paste your token]

# After this, future pushes work with just:
git push origin main
```

### Option 3: Switch to SSH (One-time setup, best long-term)

```bash
# 1. Verify SSH key is added to GitHub
ssh -T git@github.com

# 2. Change remote to SSH
git remote set-url origin git@github.com:jimxzai/asicForTranAI.git

# 3. Now all pushes use SSH:
git push origin main
```

## Common Update Tasks

### 1. Update Performance Numbers

Edit `docs/index.html`:
```html
<!-- Find this section around line 50 -->
<div class="stat-card">
    <span class="stat-value">4188</span>  <!-- Change this number -->
    <div class="stat-label">Tokens/sec</div>
</div>
```

### 2. Add New Section to Technical Docs

Edit `docs/technical.html`:
```html
<!-- Add before the footer -->
<div class="doc-section">
    <h2 id="newsection">7. Your New Section</h2>
    <p>Your content here...</p>
</div>
```

### 3. Update README

Edit `README.md`:
```bash
# Edit the file
vim README.md  # or use your preferred editor

# Commit and push
git add README.md
git commit -m "Update README with latest info"
git push
```

### 4. Add New Code Examples

```bash
# Add new Fortran files
cp my_new_code.f90 2025-3.5bit-groq-mvp/

# Commit and push
git add 2025-3.5bit-groq-mvp/
git commit -m "Add new code example: my_new_code.f90"
git push
```

## Add Workflow Files Back

If you want to add the GitHub Actions workflow files back:

### Method 1: Update Token and Push

```bash
# 1. Update your token to include 'workflow' scope:
#    https://github.com/settings/tokens
#    - Edit your token
#    - Check: ✓ workflow
#    - Click "Update token"

# 2. Restore workflow files
git checkout HEAD~1 -- .github/workflows/

# 3. Commit and push
git add .github/workflows/
git commit -m "Add GitHub Actions workflows"
git push https://YOUR_TOKEN@github.com/jimxzai/asicForTranAI.git main
```

### Method 2: Recreate Workflow Files

Create `.github/workflows/pages.yml`:

```yaml
name: Deploy GitHub Pages

on:
  push:
    branches: [ main ]
    paths:
      - 'docs/**'

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

Then commit and push with workflow-enabled token.

## Testing Changes Locally

Before pushing, preview your website locally:

```bash
# Option 1: Open directly
open docs/index.html

# Option 2: Run local server (better for testing links)
cd docs
python3 -m http.server 8000
# Visit: http://localhost:8000
```

## Workflow for Major Updates

```bash
# 1. Create a new branch
git checkout -b feature/update-homepage

# 2. Make your changes
# Edit files...

# 3. Test locally
open docs/index.html

# 4. Commit
git add .
git commit -m "Major homepage redesign"

# 5. Push branch
git push origin feature/update-homepage

# 6. Create Pull Request on GitHub
# Review changes, then merge to main

# 7. Pull the merged changes
git checkout main
git pull

# Website updates automatically!
```

## Quick Reference

### Most Common Commands

```bash
# Quick update cycle
git add .
git commit -m "Update content"
git push

# Check website status
# Visit: https://github.com/jimxzai/asicForTranAI/deployments

# Verify website is live
curl -I https://jimxzai.github.io/asicForTranAI/

# See what changed
git status
git diff
```

### GitHub Pages Build Time

- **Typical**: 30-90 seconds
- **First deployment**: 2-3 minutes
- **Check deployment status**: https://github.com/jimxzai/asicForTranAI/actions

### Troubleshooting

**Website not updating?**
```bash
# 1. Check if push succeeded
git log origin/main --oneline -3

# 2. Check GitHub Pages deployment
# Visit: https://github.com/jimxzai/asicForTranAI/settings/pages

# 3. Hard refresh browser
# Chrome/Edge: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)

# 4. Check GitHub Actions
# Visit: https://github.com/jimxzai/asicForTranAI/actions
```

**Push failed?**
```bash
# Token expired or lacks permissions
# Go to: https://github.com/settings/tokens
# Edit your token and extend expiration or add scopes

# Then push again
git push https://YOUR_TOKEN@github.com/jimxzai/asicForTranAI.git main
```

## Best Practices

1. **Always test locally first**: `open docs/index.html`
2. **Use meaningful commit messages**: Describe what you changed
3. **Update incrementally**: Small, focused commits are better
4. **Check deployment**: Wait for GitHub Pages to rebuild
5. **Clear browser cache**: Hard refresh to see changes

---

**Questions?** Check:
- GitHub Pages docs: https://docs.github.com/pages
- Repository settings: https://github.com/jimxzai/asicForTranAI/settings

Your website is live at: **https://jimxzai.github.io/asicForTranAI/**
