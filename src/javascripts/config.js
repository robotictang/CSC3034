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

const lecturerFolder = "E Lecturer’s PowerPoint slides & lecture notes or SIM/";

function isLecturerPdf(file) {
  return file.startsWith(lecturerFolder)
    && file.toLowerCase().endsWith(".pdf")
    && !/[\\\u0000-\u001f\u007f]/.test(file)
    && file.split("/").every((part) => part && part !== "." && part !== "..");
}

function routeLecturerPdfsThroughViewer() {
  document.querySelectorAll("a[href*='/blob/main/'][href$='.pdf']").forEach((link) => {
    const source = new URL(link.href);
    const marker = "/blob/main/";
    const markerIndex = source.pathname.indexOf(marker);
    if (source.hostname !== "github.com" || markerIndex === -1) return;

    let file;
    try {
      file = decodeURIComponent(source.pathname.slice(markerIndex + marker.length));
    } catch {
      return;
    }
    if (!isLecturerPdf(file)) return;

    const cardTitle = link.closest(".resource-card")?.querySelector("h3")?.textContent.trim();
    const viewer = new URL("pdf-viewer.html", window.location.href);
    viewer.searchParams.set("file", file);
    viewer.searchParams.set(
      "title",
      cardTitle ? `${cardTitle} — ${link.textContent.trim()}` : link.textContent.trim()
    );
    link.href = viewer.href;
    link.target = "_self";
  });
}

function configurePdfViewer() {
  const frame = document.querySelector("#course-pdf-frame");
  if (!frame) return;

  const status = document.querySelector("#course-pdf-status");
  const heading = document.querySelector("#course-pdf-title");
  const externalLink = document.querySelector("#course-pdf-external");
  const parameters = new URLSearchParams(window.location.search);
  const file = parameters.get("file") || "";
  const title = parameters.get("title") || "Course PDF";

  if (!isLecturerPdf(file)) {
    frame.removeAttribute("src");
    frame.hidden = true;
    status.hidden = false;
    externalLink.hidden = true;
    return;
  }

  frame.hidden = false;
  status.hidden = true;
  externalLink.hidden = false;
  const encodedPath = file.split("/").map(encodeURIComponent).join("/");
  const rawUrl = `https://raw.githubusercontent.com/robotictang/CSC3034/main/${encodedPath}`;
  document.title = `${title} - CSC3034 CI Labs`;
  heading.textContent = title;
  frame.title = title;
  frame.src = rawUrl;
  externalLink.href = rawUrl;
}

routeLecturerPdfsThroughViewer();
configurePdfViewer();

function initializeCoursePage() {
  routeLecturerPdfsThroughViewer();
  configurePdfViewer();
  if (typeof window.MathJax?.typesetPromise === "function") {
    window.MathJax.typesetPromise().catch((error) => {
      console.warn("Could not typeset course equations:", error);
    });
  }
}

if (typeof document$ !== "undefined") {
  document$.subscribe(initializeCoursePage);
} else {
  initializeCoursePage();
}
