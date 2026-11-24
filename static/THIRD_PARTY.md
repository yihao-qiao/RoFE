# Third-party assets

Everything the page needs is served from this folder; nothing is loaded from external services.
`css/page.css` and `css/fonts.css` belong to this page; the other stylesheets, the fonts and MathJax are third-party
files under the licences below.

| Asset | Files | Licence |
|---|---|---|
| Nerfies project-page template | `../index.html` layout, `css/index.css` | CC BY-SA 4.0 (credit kept in the page footer) |
| Bulma | `css/bulma.min.css` | MIT |
| Charis SIL (subset to Latin, Greek and common symbols) | `fonts/CharisSIL-*.woff2` | OFL 1.1, `fonts/CharisSIL-OFL.txt` |
| Latin Modern Mono (subset) | `fonts/LMMono10-Regular.woff2` | GUST Font License, `fonts/LatinModern-GUST-FONT-LICENSE.txt` |
| MathJax 4.1.3 with the mathjax-newcm (New Computer Modern) font, speech worker and English/Nemeth maps | `mathjax/` | Apache 2.0, `mathjax/LICENSE` |
| Academicons 1.9.4, outline of the arXiv glyph (U+E974), Paper button | inline SVG in `../index.html` | OFL 1.1, `fonts/Academicons-OFL.txt` |
| Font Awesome Free 5.15.1, `github` icon, Code button | inline SVG in `../index.html` | CC BY 4.0 (https://fontawesome.com/license/free) |
| Photos of the floral-bouquet task of MulTaBench (Kaggle), Step 0 of the overview figure | `images/overview.svg` | Apache 2.0 (licence text as in `mathjax/LICENSE`); credited here and in the caption of the paper's Fig. 1 |

The figures in `images/` are the paper's own figures, converted to SVG. Their text is drawn as glyph outlines, so
they need no font files.

Iowan Old Style, the text face, is a commercial Bitstream font. It is named first in the font stack
(`css/fonts.css`) so that systems that ship it (macOS, iOS) use it, but it is not served from here.
The two button icons are drawn as inline SVG paths; the icon fonts and the Font Awesome script themselves are not
included, so the page loads no icon font.
