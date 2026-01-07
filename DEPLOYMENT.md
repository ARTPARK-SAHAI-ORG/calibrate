# Deploying Documentation to Netlify

This guide explains how to deploy the Pense documentation to Netlify.

## Prerequisites

- A Netlify account (free tier works)
- Your repository pushed to GitHub (can be private)

## Deployment Steps

### 1. Connect Repository to Netlify

1. Go to [Netlify](https://www.netlify.com) and sign in
2. Click "Add new site" → "Import an existing project"
3. Connect to GitHub and select your repository
4. Netlify will auto-detect the build settings from `netlify.toml`

### 2. Verify Build Settings

Netlify should automatically detect:

- **Build command**: `pip install -r requirements-docs.txt && mkdocs build`
- **Publish directory**: `site`
- **Python version**: 3.10

If not auto-detected, manually configure:

- Build command: `pip install -r requirements-docs.txt && mkdocs build`
- Publish directory: `site`
- Python version: 3.10

### 3. Deploy

1. Click "Deploy site"
2. Netlify will build and deploy your documentation
3. Your site will be available at `https://<random-name>.netlify.app`

### 4. Customize Site Name (Optional)

1. Go to Site settings → General
2. Change "Site name" to something like `pense-docs`
3. Your site will be available at `https://pense-docs.netlify.app`

### 5. Update mkdocs.yml

Update the `site_url` in `mkdocs.yml` to match your Netlify site URL:

```yaml
site_url: https://your-site-name.netlify.app
```

Also update `repo_url` and `repo_name` if you want GitHub links:

```yaml
repo_url: https://github.com/your-username/agentloop
repo_name: pense
```

## Automatic Deployments

Netlify automatically deploys:

- **Production**: Every push to your main/master branch
- **Preview**: Every pull request gets a preview deployment

## Local Testing

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements-docs.txt

# Serve locally
mkdocs serve

# Build locally
mkdocs build
```

Visit `http://127.0.0.1:8000` to preview your documentation.

## Troubleshooting

### Build Fails

- Check Netlify build logs for errors
- Verify Python version is 3.10+
- Ensure `requirements-docs.txt` is in the repository

### Site Not Updating

- Check that changes are pushed to the main branch
- Verify Netlify build completed successfully
- Clear browser cache

### Missing Files

- Ensure all markdown files are in the `docs/` directory
- Check that `mkdocs.yml` navigation matches your file structure

## Next Steps

- Customize the theme in `mkdocs.yml`
- Add more content following the documentation plan
- Set up a custom domain (optional)
