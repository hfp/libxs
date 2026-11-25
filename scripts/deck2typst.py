#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render a reveal.js Markdown deck (as mkslides builds it) to a landscape PDF
via Pandoc and Typst.

Screenshotting the built deck in a headless browser is pixel-exact but gives a
PDF of images: no selectable text, several MB, and a minute or more for a deck.
This keeps the slides as text and vector art (a fraction of the size, under a
second) at the cost of exact fidelity -- Typst paginates a flow where reveal
fits content to a fixed viewport, so a dense slide is scaled down to fit rather
than overflowing to a second page.

Reveal-only constructs are translated rather than dropped: a
"data-background-image" slide becomes a figure (full-bleed without text), an
image on a line of its own (Markdown or an <img> tag) takes the height the
text leaves, span.when becomes a small gray run beside the title, and
span.dim becomes a smaller gray run. The layout follows reveal's black theme
and the decks' custom.css (h2 at 1.8em, and the classes dim, when, refs, and
center where a deck defines them).

Typst is used through its Python wheel ("pip install typst"), which ships a
self-contained binary, so no Rust toolchain is needed. Where pip is refused,
an unpacked wheel in TYPST_HOME (default ~/.local/lib/typst-wheel) is found
too, and --typst points at a typst executable instead.

Usage:
    python3 deck2typst.py index.md out.pdf [--assets DIR]
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

# 16:9 at reveal's 1920x1080, in points, so figures keep their proportions.
PAGE_W_PT = 960.0
PAGE_H_PT = 540.0
# Reveal's own content inset at 1920x1080 is ~95px, i.e. half that in points.
MARGIN_PT = 47.0

# The layout is a translation of the reveal.js theme rather than a guess, so
# the two renderings agree. Typst has no CSS, but every value below has a
# counterpart in black.css (its --r-* custom properties) or in custom.css, and
# is expressed as a ratio so it survives a change of BODY_PT.
#
#   --r-main-font-size: 42px        body text
#   h2 font-size: 1.8em            custom.css, overrides the theme's 1.6em
#   --r-heading3-size: 1.3em
#   --r-heading-margin: 0 0 20px 0  space below a heading
#   --r-block-margin: 20px          space around a paragraph or list
#   --r-heading-line-height: 1.2
#   .reveal p line-height: 1.3
#   .dim: 50% size (custom.css)
REVEAL_BASE_PX = 42.0
# The page is 960pt for reveal's 1920px, so a CSS pixel is half a point and the
# body size follows directly. Getting this wrong makes every ratio below wrong
# too: at 19pt the titles wrapped where reveal keeps them on one line.
BODY_PT = REVEAL_BASE_PX / 2.0
H2_EM = 1.8
H3_EM = 1.3
BLOCK_EM = 20.0 / REVEAL_BASE_PX
HEAD_LINE = 1.2
HEAD_WEIGHT = 600  # --r-heading-font-weight; "bold" (700) sets titles wider
BODY_LINE = 1.3
DIM_EM = 0.5
REFS_EM = 0.7  # custom.css .refs ul
WHEN_EM = 0.45  # custom.css .when, relative to the heading

# CSS line-height is the entire line advance. Typst's leading is only the gap
# added between lines, so the per-line extent has to come off. Measured, not
# assumed: two lines at 21pt occupy 36.9pt at leading 0.30em and 43.2pt at
# 0.60em, which puts the extent at (36.9 - 0.30*21) / 2 = 0.729em and confirms
# advance = extent + leading. Subtracting a full em instead (an easy guess)
# renders the text visibly tighter than the deck.
CAP_EM = 0.729
BODY_LEADING = BODY_LINE - CAP_EM
HEAD_LEADING = HEAD_LINE - CAP_EM

# Block separation needs the same correction as the leading, and for the same
# reason. A CSS margin sits between LINE BOXES, which already include the
# leading; a Typst block is tight to the glyph extents. Passing the 20px margin
# straight through therefore made a paragraph break (24.0pt baseline to
# baseline) TIGHTER than an ordinary line break (25.9pt) -- a new paragraph read
# as a wrapped line. Adding the margin on top of the leading reproduces what
# CSS actually does: a block boundary is one line advance plus the margin.
BLOCK_SPACING = BODY_LEADING + BLOCK_EM

# List items carry no margin in reveal (.reveal li), so item to item should read
# exactly like a wrapped line: the leading, with no block spacing added.
ITEM_SPACING = BODY_LEADING

# Same reason, for the space below a heading: the block ends at the baseline of
# the last title line, so the descender room that CSS gets from the line box has
# to be added to the theme's 20px margin or the title collides with the text.
HEAD_BELOW = BLOCK_EM + (HEAD_LINE - CAP_EM) * H2_EM * 0.5

# One title line at full size, so the header box is the same height on every
# slide even when a long title has been scaled down to fit.
HEAD_HEIGHT = CAP_EM * H2_EM

# A figure beside the text is as tall as the body area under the header: the
# frame less the margins and the header box with its space below.
SIDE_FIG_PT = PAGE_H_PT - 2 * MARGIN_PT - (HEAD_HEIGHT + HEAD_BELOW) * BODY_PT

# reveal asks for Source Sans Pro; "Source Sans 3" is its successor and is what
# Debian ships (fonts-adobe-sourcesans3). The fallbacks keep the render working
# on a box without it, at the cost of wider titles that wrap sooner.
FONTS = ("Source Sans 3", "Source Sans Pro", "TeX Gyre Heros", "DejaVu Sans")

BG = "#191919"
FG = "#ffffff"
ACCENT = "#42affa"
DIM = "#898781"

# Tried in order until the slide fits; mirrors what a presenter would do by
# hand. The floor is a readability limit, not a fitting one -- below this the
# slide needs editing, and the script says so.
SCALES = (1.0, 0.94, 0.88, 0.82, 0.76, 0.7, 0.64, 0.58, 0.52)

TEMPLATE = r"""
%(doc)s
#set page(width: %(pw).1fpt, height: %(ph).1fpt, margin: %(pm).1fpt,
          fill: rgb("%(bg)s"))
#set text(font: %(ff)s, size: %(bs).1fpt,
          fill: rgb("%(fg)s"))
#set par(justify: false, leading: %(bl).2fem, spacing: %(bsp).2fem)
#set block(spacing: %(bsp).2fem)
#set list(spacing: %(isp).2fem, indent: 0.6em)
#set enum(spacing: %(isp).2fem, indent: 0.6em)

// A heading keeps its own tighter leading (CSS 1.2 against the body's 1.3),
// and the space below it is the theme's 20px expressed against the body size,
// so a two-line title does not sit on top of the first paragraph.
#show heading: it => block(width: 100%%, below: %(hb).2fem, above: 0em)[
  #set par(leading: %(hl).2fem)
  #text(size: %(h2).2fem, weight: %(hw)d, fill: rgb("%(ac)s"))[#upper(it.body)]
]
#show heading.where(level: 3): it => block(width: 100%%, below: %(h3b).2fem)[
  #set par(leading: %(hl).2fem)
  #text(size: %(h3).2fem, weight: %(hw)d, fill: rgb("%(ac)s"))[#it.body]
]
#show link: set text(fill: rgb("%(ac)s"))
#set table(stroke: 0.5pt + rgb("%(dm)s"))
#show table.cell.where(y: 0): set text(weight: "bold")
#let dim(body) = text(size: %(de).2fem, fill: rgb("%(dm)s"))[#body]

// A dated title. The date stays outside upper() and at .when's 45 percent; run
// through the ordinary heading rule it would be set as full-size uppercase
// title text, which pushes the title onto a second line.
//
// The header is a FIXED-HEIGHT box on every slide, so titles -- and therefore
// the bodies beneath them -- always start at the same place. Three simpler
// arrangements all failed: inline, the date drops to a line of its own and makes
// a dated header taller; in a grid column it steals width and the title wraps;
// placed over the title it collides with a full-width one. So the title is
// scaled to fit one line in the width the date leaves. Text width is linear in
// font size, so the ratio is exact and needs no search.
#let slidetitle(t, w) = context {
  let full = %(h2).2fem
  let dw = if w == none { 0pt } else {
    measure(text(size: %(we).2fem)[#w]).width + 0.6 * %(bs).1fpt
  }
  let avail = %(pw).1fpt - 2 * %(pm).1fpt - dw
  let nat = measure(text(size: full, weight: %(hw)d)[#upper(t)]).width
  let s = calc.min(1.0, avail / nat)
  block(width: 100%%, height: %(hh).2fem, above: 0em, below: %(hb).2fem)[
    #grid(columns: (1fr, auto), align: bottom,
      text(size: full * s, weight: %(hw)d, fill: rgb("%(ac)s"))[#upper(t)],
      if w != none {
        text(size: %(we).2fem, fill: rgb("%(dm)s"))[#w]
      },
    )
  ]
}
#let fullbleed(path) = block(width: 100%%, height: 100%%)[
  #align(center + horizon)[#image(path, height: 100%%, fit: "contain")]
]
// A figure beside the text: reveal's background at "right" or "left" with text
// on the slide. Fixed to the body area's height rather than 1fr, since a grid
// row has no height to divide; the width of its column bounds it too.
#let figside(path) = image(path, width: 100%%, height: %(fh).1fpt, fit: "contain")
// A figure under text takes the height the text did not use: 1fr resolves
// against the slide's fixed-height block, so the image is never cropped and
// never overlaps the caption.
#let figrest(path) = block(width: 100%%, height: 1fr)[
  #align(center + horizon)[#image(path, height: 100%%, fit: "contain")]
]

// One slide per page, scaled down until it fits. measure() reports the height
// the content would take, so the choice is made before anything is laid out;
// without it a dense slide silently continues onto a second page.
//
// The title sits at the top of the frame on every slide, at the same height,
// rather than floating with the content: flipping through the PDF should not
// move the heading around. This is a deliberate departure from reveal's
// "center: true", which centres the whole slide and therefore puts the title
// somewhere different on each one. Only the body takes the room the header
// leaves (height: 1fr), and only the body is ever scaled.
#let slide(title: [], when: none, centred: false, refs: false,
          listy: false, body: []) = context {
  let aw = %(pw).1fpt - 2 * %(pm).1fpt
  let ah = %(ph).1fpt - 2 * %(pm).1fpt
  let th = measure(block(width: aw, title)).height
  // A reference list is scaled so that no entry wraps -- the point of the
  // slide is one publication per line. custom.css uses a fixed 0.7em, but the
  // size that achieves it depends on the font, so it is derived here instead:
  // laid out at an absurd width nothing can wrap, giving the height the body
  // would have with every entry on one line. That height is linear in the font
  // size, so comparing it against the real, wrapped height at each candidate
  // scale detects wrapping exactly, without counting entries.
  let base = if refs { %(rf).2f * %(bs).1fpt } else { %(bs).1fpt }
  let flat = if refs {
    measure(block(width: 100000pt, text(size: base, body))).height
  } else { 0pt }
  let chosen = %(minscale).2f
  for s in (%(scales)s) {
    let bh = measure(block(width: aw, text(size: base * s, body))).height
    // 1.02 absorbs rounding; a single wrapped line costs far more than 2%%.
    let unwrapped = not refs or bh <= flat * s * 1.02
    if th + bh <= ah and unwrapped {
      chosen = s
      break
    }
  }
  // A cover or sign-off is centred on both axes: the pinned header is for the
  // pages of detail in between, where a moving heading is what distracts.
  if centred {
    // The BLOCK is centred, not the text inside it: centring the text leaves
    // bullet markers stranded at the left margin while their text floats.
    // Reveal does the same thing with "ul { display: inline-block }". The box
    // is only as wide as the content needs, clamped to the frame.
    // measure() on a block reports the CONSTRAINT, not the content: measuring
    // the content itself is what gives the natural (unwrapped) width.
    // Only a list needs the narrow box. Prose is centred line by line anyway,
    // and a narrow box wraps a heading in the body -- the role line on a cover
    // -- at the width of the widest body line, not at the width of the frame.
    let nat = measure(text(size: base * chosen, body)).width
    block(width: aw, height: ah)[
      #align(center + horizon)[
        #align(center)[#title]
        #box(width: if listy { calc.min(nat, aw) } else { aw })[
          // A list is ranged left inside the centred box, so its markers stay
          // against their text; prose is centred, as a cover or sign-off wants.
          #align(if listy { left } else { center })[
            #text(size: base * chosen)[#body]
          ]
        ]
      ]
    ]
  } else {
    // The title is pinned to the top, and the body is centred vertically in
    // whatever room it leaves: a sparse slide reads as a block in the frame
    // rather than as content that fell to the top edge. Only the vertical axis
    // is centred -- text stays ranged left.
    block(width: aw, height: ah)[
      #title
      #block(width: 100%%, height: 1fr)[
        #align(horizon)[#text(size: base * chosen)[#body]]
      ]
    ]
  }
  [#metadata(chosen)<fit>]
}

$body$
"""


def slide_source(text, assets, root):
    """Split one slide into (title, body), with reveal constructs translated.

    The title is returned separately so the template can pin it to the top of
    every frame; a slide without a heading (a bare figure) has an empty title.
    Image paths stay relative to root: Typst resolves them against its --root,
    so an absolute path would be appended to it and not found.
    """
    # Speaker notes start at a "Note:" line and run to the end of the slide
    # (reveal's notesSeparator, case-insensitive); they are not slide content.
    text = re.split(r"^[ \t]*notes?:", text, maxsplit=1, flags=re.M | re.I)[0]
    bg = re.search(r'data-background-image="([^"]+)"', text)
    m_cls = re.search(r'<!--\s*\.slide:[^>]*class="([^"]+)"', text)
    classes = m_cls.group(1).split() if m_cls else []
    lines = [
        ln for ln in text.split("\n") if not ln.strip().startswith("<!--")
    ]
    body = "\n".join(lines).strip()

    # span.when is a date on the heading. Rendered through the ordinary heading
    # rule it would be uppercased at full title size; slidetitle() keeps it
    # small and gray the way custom.css does.
    title = ""
    when = None
    # Match to the first newline explicitly: "$" without re.M anchors at the end
    # of the whole slide, so this silently found no title at all.
    m_head = re.match(r"^(#{1,3})[ \t]+([^\n]*)", body)
    if m_head:
        head = m_head.group(2)
        body = body[m_head.end() :].lstrip("\n")
        m_when = re.search(r'\s*<span\s+class="when">(.*?)</span>\s*$', head)
        if m_when:
            title = '#slidetitle("%s", "%s")' % (
                _q(head[: m_when.start()]),
                _q(m_when.group(1)),
            )
            when = _q(m_when.group(1))
        else:
            title = '#slidetitle("%s", none)' % _q(head)
    body = re.sub(r'\s*<span\s+class="when">(.*?)</span>', r" (\1)", body)
    # span.dim recedes via opacity in reveal, which PDF text cannot do; the
    # template's dim() approximates it with size and gray. The format has to be
    # exactly "typst": Pandoc passes a raw inline of that format through
    # verbatim, but SILENTLY DROPS one of any other format -- an earlier
    # "{=typst-dim}" marker lost every dim line without a warning. The span is
    # opened and closed by two raw inlines rather than passed as one string, so
    # Pandoc still converts the Markdown between them: a string argument printed
    # "**bold**" and "[label](url)" literally. DOTALL because a dim note may
    # wrap in the source; without it such a span never matched and rendered as
    # body text once its tags were stripped below.
    body = re.sub(
        r'<span\s+class="dim">(.*?)</span>',
        lambda m: "`#dim[`{=typst}%s`]`{=typst}" % m.group(1).strip(),
        body,
        flags=re.S,
    )
    body = re.sub(r"</?span[^>]*>", "", body)
    # A line break written as HTML (Pandoc's Typst writer drops raw HTML)
    # becomes a Markdown hard break.
    body = re.sub(r"<br\s*/?>", "\\\n", body, flags=re.I)

    # An image on a line of its own, Markdown or an <img> tag (which Pandoc's
    # Typst writer drops), takes the height the text leaves rather than its
    # natural size: figures are drawn ~2500px wide, which blows past the
    # fitting floor, and the caption above or below must stay on the slide.
    def figure(m):
        path = m.group(1)
        if assets:
            cand = os.path.join(assets, os.path.basename(path))
            if os.path.isfile(cand):
                path = os.path.relpath(os.path.abspath(cand), root)
        # blank lines around it: a fence inside a paragraph is not a block
        return '\n```{=typst}\n#figrest("%s")\n```\n' % path

    body = re.sub(
        r"^[ \t]*<img\b[^>]*\bsrc=\"([^\"]+)\"[^>]*>[ \t]*$",
        figure,
        body,
        flags=re.M | re.I,
    )
    body = re.sub(
        r"^[ \t]*!\[[^\]]*\]\(([^)]+)\)[ \t]*$", figure, body, flags=re.M
    )

    if bg:
        path = bg.group(1)
        if assets:
            cand = os.path.join(assets, os.path.basename(path))
            if os.path.isfile(cand):
                path = os.path.relpath(os.path.abspath(cand), root)
        # Most background slides carry no text at all: the image is the slide.
        if not body:
            return (
                title,
                when,
                '```{=typst}\n#fullbleed("%s")\n```' % path,
                classes,
            )
        # Positioned at a side, the background shares the slide with the text
        # (reveal: the figure right or left, the text ranged against the other
        # side). A grid gives each half; the body stays Markdown between the
        # two raw blocks, so Pandoc still converts it.
        m_pos = re.search(r'data-background-position="([^"]+)"', text)
        side = m_pos.group(1).split()[0] if m_pos else ""
        # A horizontal percentage counts as the side it leans to.
        m_pct = re.fullmatch(r"([0-9.]+)%", side)
        if m_pct:
            side = "right" if 50 <= float(m_pct.group(1)) else "left"
        if side in ("right", "left"):
            fig = 'figside("%s")' % path
            text_open = "[\n```\n\n%s\n\n```{=typst}\n]" % body
            # The text's share follows its longest line (markup and the small
            # dim runs aside; <br> breaks a line): short points leave the
            # figure room, up to more than half of the width, long ones take
            # up to the ~60% reveal gives them, since a wrapped line is worse
            # than a smaller figure. ~28 characters per fr of text.
            plain = re.sub(r"`#dim\[`\{=typst\}.*?`\]`\{=typst\}", "", body)
            plain = re.sub(r"\*\*|__|`|\\$", "", plain, flags=re.M)
            longest = max(
                (
                    len(ln.strip().lstrip("-*0123456789. "))
                    for ln in re.split(r"\n|<br\s*/?>", plain)
                ),
                default=0,
            )
            share = min(1.8, max(1.0, longest / 28.0))
            cols = (
                "(%.2ffr, 1fr)" % share
                if "right" == side
                else "(1fr, %.2ffr)" % share
            )
            cells = (text_open, fig) if "right" == side else (fig, text_open)
            return (
                title,
                when,
                (
                    "```{=typst}\n#grid(columns: %s, column-gutter:"
                    " 1.2em, align: horizon, %s, %s)\n```" % ((cols,) + cells)
                ),
                classes,
            )
        # Otherwise reveal layers the text over the background, which a flowed
        # page cannot do. Put the text first and give the image the remaining
        # height, so the two never collide.
        return (
            title,
            when,
            ('%s\n\n```{=typst}\n#figrest("%s")\n```' % (body, path)),
            classes,
        )
    return title, when, body, classes


def _q(text):
    """Quote for a Typst string literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"').strip()


def convert(src, assets, root):
    """Split the stitched deck and emit the markdown Pandoc will translate."""
    with open(src, encoding="utf8") as f:
        deck = f.read()
    slides = [s for s in (t.strip() for t in deck.split("\n---\n")) if s]
    out = []
    for text in slides:
        title, when, body, classes = slide_source(text, assets, root)
        # Raw blocks are passed through by Pandoc's typst writer, so the slide
        # is handed to the template as two separate arguments: the title it
        # pins to the top, and the body it centres in the rest.
        whenarg = '"%s"' % when if when else "none"
        out.append(
            "```{=typst}\n#slide(\n  title: [%s],\n  when: %s,"
            "\n  centred: %s,\n  refs: %s,\n  listy: %s,"
            "\n  body: [\n```\n\n%s\n\n```{=typst}\n  ],\n)\n```"
            % (
                title,
                whenarg,
                "true" if "center" in classes else "false",
                "true" if "refs" in classes else "false",
                "true" if re.search(r"^[-*] ", body, re.M) else "false",
                body,
            )
        )
    return "\n\n".join(out) + "\n", len(slides)


def docinfo(title, author, keywords):
    """PDF metadata. File previews and viewers show it -- a slide PDF without
    it is titled by its file name."""
    fields = []
    if title:
        fields.append('title: "%s"' % _q(title))
    if author:
        fields.append('author: "%s"' % _q(author))
    words = [w.strip() for w in (keywords or "").split(",") if w.strip()]
    if words:
        fields.append(
            "keywords: (%s,)" % ", ".join('"%s"' % _q(w) for w in words)
        )
    return "#set document(%s)" % ", ".join(fields) if fields else ""


def template(doc=""):
    return TEMPLATE % {
        "doc": doc,
        "pw": PAGE_W_PT,
        "ph": PAGE_H_PT,
        "pm": MARGIN_PT,
        "bs": BODY_PT,
        "bg": BG,
        "fg": FG,
        "ac": ACCENT,
        "dm": DIM,
        "scales": ", ".join("%.2f" % s for s in SCALES),
        "minscale": SCALES[-1],
        "bl": BODY_LEADING,
        "hl": HEAD_LEADING,
        "bm": BLOCK_EM,
        "hb": HEAD_BELOW,
        "h3b": HEAD_BELOW * 0.6,
        "h2": H2_EM,
        "h3": H3_EM,
        "de": DIM_EM,
        "we": WHEN_EM * H2_EM,
        "hw": HEAD_WEIGHT,
        "hh": HEAD_HEIGHT,
        "rf": REFS_EM,
        "bsp": BLOCK_SPACING,
        "isp": ITEM_SPACING,
        "fh": SIDE_FIG_PT,
        "ff": "(" + ", ".join('"%s"' % f for f in FONTS) + ")",
    }


def find_wheel():
    """Make typst importable, whether pip-installed or an unpacked wheel.

    "pip install typst" is the normal route and is left alone. Where pip is
    refused (PEP 668, an externally managed Python), the same wheel unpacked
    into TYPST_HOME or ~/.local/lib/typst-wheel is used instead; at ~78 MB it
    does not belong in a repository. Appended, not prepended, so an installed
    module keeps precedence.
    """
    try:
        import typst  # noqa: F401
    except ImportError:
        for cand in (
            os.environ.get("TYPST_HOME"),
            os.path.expanduser("~/.local/lib/typst-wheel"),
        ):
            if cand and os.path.isdir(os.path.join(cand, "typst")):
                sys.path.append(cand)
                break


def compile_pdf(typ, pdf, root, typst_bin):
    """Compile Typst source, preferring the wheel over an executable."""
    result = 0
    if not typst_bin:
        find_wheel()
        try:
            import typst

            typst.compile(typ, output=pdf, root=root)
        except ImportError:
            typst_bin = shutil.which("typst")
            if not typst_bin:
                print(
                    "no typst: pip install typst, set TYPST_HOME, or pass --typst",
                    file=sys.stderr,
                )
                result = 1
        except Exception as exc:
            print("typst failed: %s" % exc, file=sys.stderr)
            result = 1
    if 0 == result and typst_bin:
        run = subprocess.run(
            [typst_bin, "compile", "--root", root, typ, pdf],
            capture_output=True,
            text=True,
            check=False,
        )
        if 0 != run.returncode:
            print(run.stderr.strip(), file=sys.stderr)
            result = 1
    return result


def fits(typ, root, typst_bin):
    """Per-slide scale factors, so an over-dense slide can be reported."""
    scales = []
    if not typst_bin:
        find_wheel()
        try:
            import json
            import typst

            scales = [
                x["value"]
                for x in json.loads(typst.query(typ, "<fit>", root=root))
            ]
        except Exception:
            scales = []
    return scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="stitched deck markdown")
    ap.add_argument("pdf")
    ap.add_argument(
        "--assets",
        default=None,
        help="directory holding the figures (default: beside src)",
    )
    ap.add_argument(
        "--typst",
        default=None,
        help="typst executable, if the Python wheel is unavailable",
    )
    ap.add_argument(
        "--keep",
        action="store_true",
        help="keep the generated .typ beside the PDF",
    )
    ap.add_argument("--title", default=None, help="PDF document title")
    ap.add_argument("--author", default=None, help="PDF document author")
    ap.add_argument(
        "--keywords", default=None, help="PDF keywords, comma-separated"
    )
    args = ap.parse_args()
    result = 0

    if not os.path.isfile(args.src):
        print("no such file: %s" % args.src, file=sys.stderr)
        result = 1

    if 0 == result and not shutil.which("pandoc"):
        print("pandoc is required", file=sys.stderr)
        result = 1

    if 0 == result:
        src = os.path.abspath(args.src)
        root = os.path.dirname(src)
        assets = args.assets or os.path.join(root, "assets")
        body, nslides = convert(
            src, assets if os.path.isdir(assets) else None, root
        )

        # The .typ must sit in root, not a subdirectory of it: Typst resolves
        # relative image paths against the file's own directory.
        tmp = tempfile.mkdtemp(prefix="deck2typst-", dir=root)
        md = os.path.join(tmp, "flat.md")
        tpl = os.path.join(tmp, "template.typ")
        typ = os.path.join(root, ".deck2typst.typ")
        with open(md, "w", encoding="utf8") as f:
            f.write(body)
        with open(tpl, "w", encoding="utf8") as f:
            f.write(template(docinfo(args.title, args.author, args.keywords)))

        run = subprocess.run(
            ["pandoc", md, "-o", typ, "-t", "typst", "--template", tpl],
            capture_output=True,
            text=True,
            check=False,
        )
        if 0 != run.returncode:
            print(run.stderr.strip(), file=sys.stderr)
            result = 1

        if 0 == result:
            result = compile_pdf(
                typ, os.path.abspath(args.pdf), root, args.typst
            )

        if 0 == result:
            scales = fits(typ, root, args.typst)
            tight = [i + 1 for i, s in enumerate(scales) if s < 1.0]
            floor = [i + 1 for i, s in enumerate(scales) if s <= SCALES[-1]]
            print("wrote %s (%d slides)" % (args.pdf, nslides))
            if tight:
                print(
                    "  scaled to fit: slide(s) %s"
                    % ", ".join(str(i) for i in tight)
                )
            if floor:
                print(
                    "  at the %.2f floor, so probably too dense: %s"
                    % (SCALES[-1], ", ".join(str(i) for i in floor)),
                    file=sys.stderr,
                )

        if args.keep and 0 == result:
            shutil.copy(typ, os.path.splitext(args.pdf)[0] + ".typ")
        shutil.rmtree(tmp, ignore_errors=True)
        if os.path.isfile(typ):
            os.remove(typ)

    return result


if __name__ == "__main__":
    sys.exit(main())
