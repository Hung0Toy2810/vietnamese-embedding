// Translate static interface strings without mixing locale data into application logic.
fetch("/static/locales/vi.json")
  .then((response) => response.json())
  .then((labels) => {
    const walker = document.createTreeWalker(
      document.body,
      NodeFilter.SHOW_TEXT,
    );
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      const key = node.textContent.trim();
      if (labels[key])
        node.textContent = node.textContent.replace(key, labels[key]);
    }
    for (const element of document.querySelectorAll(
      "[title], [aria-label], [placeholder]",
    )) {
      for (const attribute of ["title", "aria-label", "placeholder"]) {
        const value = element.getAttribute(attribute);
        if (labels[value]) element.setAttribute(attribute, labels[value]);
      }
    }
    document.documentElement.lang = "vi";
  })
  .catch(() => {});
