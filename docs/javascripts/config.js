window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// The current course and lecturer-resource pages share the NVIDIA-inspired
// presentation. Archived cohorts retain their original visual treatment.
document.documentElement.classList.toggle(
  "nvidia-theme",
  !window.location.pathname.includes("/archive/")
);

document$.subscribe(() => {
  MathJax.typesetPromise();
});
