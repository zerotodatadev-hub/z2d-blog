document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("div.code").forEach((container) => {
        if (container.querySelector(".copy-button")) return;
        const pre = container.querySelector("pre");
        if (!pre) return;

        // Anchor the button
        if (!container.style.position) container.style.position = "relative";

        const btn = document.createElement("button");
        btn.className = "copy-button";
        btn.type = "button";
        btn.title = "Copy code";
        btn.setAttribute("aria-label", "Copy code to clipboard");
        btn.textContent = "Copy";

        btn.addEventListener("click", async () => {
            const text = pre.innerText;
            try {
                await navigator.clipboard.writeText(text);
            } catch {
                const ta = document.createElement("textarea");
                ta.value = text;
                ta.style.position = "fixed";
                ta.style.opacity = "0";
                document.body.appendChild(ta);
                ta.select();
                document.execCommand("copy");
                document.body.removeChild(ta);
            }
            const old = btn.textContent;
            btn.textContent = "Copied!";
            setTimeout(() => (btn.textContent = old), 1200);
        });

        container.appendChild(btn);
    });
});