// Drag a divider to resize its adjacent panels.
(() => {
  const workspace = document.querySelector(".workspace");
  const panels = [...workspace.querySelectorAll(".panel")];
  const dividers = [...workspace.querySelectorAll(".column-resizer")];
  const storageKey = "legal-rag-column-widths";
  const defaults = [1, 2, 2];
  const minimums = [160, 220, 220];
  let proportions = [...defaults];

  try {
    const saved = JSON.parse(localStorage.getItem(storageKey));
    if (
      Array.isArray(saved) &&
      saved.length === 3 &&
      saved.every((n) => Number.isFinite(n) && n > 0)
    ) {
      proportions = saved;
    }
  } catch {
    /* Browser storage may be unavailable. */
  }

  function save() {
    try {
      localStorage.setItem(storageKey, JSON.stringify(proportions));
    } catch {
      /* Resizing still works without persistence. */
    }
  }

  function applyWidths() {
    if (window.innerWidth <= 720) {
      workspace.style.removeProperty("grid-template-columns");
      return;
    }
    const available = workspace.clientWidth - 2; // Account for two 1px borders.
    const remaining = available - minimums.reduce((sum, n) => sum + n, 0);
    const total = proportions.reduce((sum, n) => sum + n, 0);
    let widths = proportions.map((n) => (available * n) / total);
    if (widths.some((n, i) => n < minimums[i])) {
      const extra = widths.map((n, i) => Math.max(0, n - minimums[i]));
      const extraTotal = extra.reduce((sum, n) => sum + n, 0);
      widths = minimums.map((n, i) => n + (remaining * extra[i]) / extraTotal);
    }
    workspace.style.gridTemplateColumns = widths.map((n) => `${n}px`).join(" ");
    dividers.forEach((divider, i) => {
      const position =
        widths.slice(0, i + 1).reduce((sum, n) => sum + n, 0) + i;
      const pair = widths[i] + widths[i + 1];
      divider.style.left = `${position}px`;
      divider.setAttribute(
        "aria-valuemin",
        Math.round((100 * minimums[i]) / pair),
      );
      divider.setAttribute(
        "aria-valuemax",
        Math.round((100 * (pair - minimums[i + 1])) / pair),
      );
      divider.setAttribute(
        "aria-valuenow",
        Math.round((100 * widths[i]) / pair),
      );
    });
  }

  function moveDivider(index, delta, initial) {
    const pair = initial[index] + initial[index + 1];
    const left = Math.max(
      minimums[index],
      Math.min(pair - minimums[index + 1], initial[index] + delta),
    );
    proportions = [...initial];
    proportions[index] = left;
    proportions[index + 1] = pair - left;
    applyWidths();
  }

  dividers.forEach((divider, index) => {
    let drag = null;
    divider.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      divider.focus();
      drag = {
        x: event.clientX,
        widths: panels.map((panel) => panel.getBoundingClientRect().width),
      };
      divider.setPointerCapture(event.pointerId);
      divider.classList.add("active");
      document.body.classList.add("resizing");
    });
    divider.addEventListener("pointermove", (event) => {
      if (drag) moveDivider(index, event.clientX - drag.x, drag.widths);
    });
    divider.addEventListener("pointerup", (event) => {
      if (divider.hasPointerCapture(event.pointerId))
        divider.releasePointerCapture(event.pointerId);
    });
    divider.addEventListener("lostpointercapture", () => {
      drag = null;
      divider.classList.remove("active");
      document.body.classList.remove("resizing");
      save();
    });
    divider.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
      event.preventDefault();
      const step = event.shiftKey ? 50 : 10;
      moveDivider(
        index,
        event.key === "ArrowLeft" ? -step : step,
        panels.map((panel) => panel.getBoundingClientRect().width),
      );
      save();
    });
    divider.addEventListener("dblclick", () => {
      proportions = [...defaults];
      applyWidths();
      save();
    });
  });

  new ResizeObserver(applyWidths).observe(workspace);
  applyWidths();
})();
