document.addEventListener('DOMContentLoaded', () => {
  const toggleButton = document.createElement('button');
  toggleButton.id = 'code-toggle';
  toggleButton.innerText = '🌙 Dark Code';
  toggleButton.style.position = 'fixed';
  toggleButton.style.bottom = '20px';
  toggleButton.style.right = '20px';
  toggleButton.style.zIndex = '1000';
  toggleButton.style.padding = '0.5em 1em';
  toggleButton.style.borderRadius = '6px';
  toggleButton.style.border = 'none';
  toggleButton.style.cursor = 'pointer';
  toggleButton.style.background = '#2563EB';
  toggleButton.style.color = '#fff';
  toggleButton.style.fontFamily = 'Inter, sans-serif';
  document.body.appendChild(toggleButton);

  // Reuse existing link or create one
  let linkEl = document.getElementById('code-theme');
  if (!linkEl) {
    linkEl = document.createElement('link');
    linkEl.rel = 'stylesheet';
    linkEl.id = 'code-theme';
    document.head.appendChild(linkEl);
  }

  // Derive base from current href (preserves correct relative prefix)
  const current = linkEl.getAttribute('href') || '';
  const slash = current.lastIndexOf('/');
  const base = slash >= 0 ? current.slice(0, slash + 1) : '';

  // Initial state and button label
  let isDark = /(^|\/)code\.css(\?|$)/.test(current);
  linkEl.setAttribute('href', base + (isDark ? 'code.css' : 'code-light.css'));
  toggleButton.innerText = isDark ? '☀️ Light Code' : '🌙 Dark Code';

  toggleButton.addEventListener('click', () => {
    isDark = !isDark;
    linkEl.setAttribute('href', base + (isDark ? 'code.css' : 'code-light.css'));
    toggleButton.innerText = isDark ? '☀️ Light Code' : '🌙 Dark Code';
  });
});
