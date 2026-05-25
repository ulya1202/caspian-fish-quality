# Deploy — canlı demo

## GitHub Pages (layihə səhifəsi)

**URL:** https://ulya1202.github.io/caspian-fish-quality/

Enabled via repo **Settings → Pages → Deploy from branch `main`, folder `/docs`**.

The workflow [`.github/workflows/pages.yml`](.github/workflows/pages.yml) is optional; if Actions deploy fails, legacy `/docs` hosting still works after the setting above is enabled.

Push to `main` updates the site automatically.

## Streamlit Community Cloud (interaktiv demo)

1. Open https://share.streamlit.io and sign in with GitHub.
2. **New app** → repository `ulya1202/caspian-fish-quality`, branch `main`.
3. **Main file path:** `app/streamlit_app.py`
4. **Requirements file:** `requirements.txt` (includes `-e .` for this package)
5. Deploy — URL will be like `https://caspian-fish-quality.streamlit.app`

After deploy, update the Streamlit URL in [`docs/index.html`](docs/index.html) if it differs.

## Lokal test

```bash
pip install -e ".[demo,test]"
streamlit run app/streamlit_app.py
```
