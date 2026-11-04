# Visual article

Open `index.html` directly in a browser, or serve the repository root with:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
# http://127.0.0.1:8765/blog/
```

This is a private repository artifact. No public hosting or deployment is configured. All chart data and image assets are local, and the page works offline.

Rebuild from the repository root with `python3 scripts/build_blog.py`. The builder reads only frozen results and images; it performs no model inference, selection or API calls. Edit narrative in `scripts/templates/blog.html`, design in `style.css`, and interaction code in `app.js`. The full machine-readable results remain in `../expanded_results/`.
