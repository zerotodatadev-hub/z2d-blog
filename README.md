# Z2D Blog

This repo builds and publishes the blog posts for Z2D.

## Installation and build commands

Setup the environment:

```bash
python -m venv nikola-env
source nikola-env/bin/activate
pip install -r requirements.txt
```

For building and deployment:

```bash
cd mysite
nikola build -a
nikola serve -b # local preview

nikola deploy # for deployment to github pages
```

---

## License

- Source code in this repository is licensed under the [MIT License](./LICENSE).
- Blog posts, articles, and media files are licensed under
  [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
