const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../src/javascripts/config.js'), 'utf8');
const folder = 'E Lecturer’s PowerPoint slides & lecture notes or SIM/';

function viewer(file = '', links = []) {
  const elements = Object.fromEntries(['frame', 'status', 'title', 'external'].map(name => [
    `#course-pdf-${name}`, { hidden: false, removeAttribute(name) { delete this[name]; } }
  ]));
  const url = new URL('https://example.com/course/pdf-viewer.html');
  url.searchParams.set('file', file);
  url.searchParams.set('title', '<img src=x onerror=alert(1)>');
  const context = vm.createContext({
    URL, URLSearchParams, console,
    window: { location: url },
    document: {
      querySelector: selector => elements[selector],
      querySelectorAll: () => links
    },
    // Simulate Material initialising before the optional MathJax CDN is ready.
    document$: { subscribe: callback => callback() }
  });
  vm.runInContext(source, context);
  return { context, elements };
}

test('valid nested PDFs are encoded and titles are assigned as plain text', () => {
  const { elements } = viewer(`${folder}Tutorials/Tutorial Q.pdf`);
  assert.equal(elements['#course-pdf-frame'].hidden, false);
  assert.equal(elements['#course-pdf-status'].hidden, true);
  assert.equal(elements['#course-pdf-title'].textContent, '<img src=x onerror=alert(1)>');
  assert.match(elements['#course-pdf-frame'].src, /Tutorials\/Tutorial%20Q\.pdf$/);
});

test('missing, traversing and non-PDF paths hide both frame and external link', () => {
  for (const file of ['', `${folder}../private.pdf`, `${folder}./a.pdf`,
    `${folder}\\a.pdf`, `${folder}a\u0000.pdf`, `${folder}/a.pdf`, `${folder}a.html`]) {
    const { elements } = viewer(file);
    assert.equal(elements['#course-pdf-frame'].hidden, true, file);
    assert.equal(elements['#course-pdf-external'].hidden, true, file);
    assert.equal(elements['#course-pdf-status'].hidden, false, file);
    assert.equal(elements['#course-pdf-frame'].src, undefined, file);
  }
});

test('malformed link encoding does not stop other PDF links being routed', () => {
  const bad = { href: 'https://github.com/robotictang/CSC3034/blob/main/%ZZ.pdf' };
  const good = {
    href: `https://github.com/robotictang/CSC3034/blob/main/${encodeURI(folder)}00%20Briefing.pdf`,
    textContent: 'Open PDF', closest: () => null
  };
  viewer('', [bad, good]);
  assert.match(good.href, /^https:\/\/example.com\/course\/pdf-viewer.html\?file=/);
  assert.equal(good.target, '_self');
});

test('a viewer can recover from a missing PDF on a later page initialisation', () => {
  const { context, elements } = viewer();
  context.window.location.search = `?file=${encodeURIComponent(folder + '00 Briefing.pdf')}`;
  vm.runInContext('configurePdfViewer()', context);
  assert.equal(elements['#course-pdf-frame'].hidden, false);
  assert.equal(elements['#course-pdf-external'].hidden, false);
  assert.equal(elements['#course-pdf-status'].hidden, true);
});
