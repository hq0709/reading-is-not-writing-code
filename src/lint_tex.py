"""Structural checks on the LaTeX source that the compiler does not make.

Both of the faults this catches were real and neither raised a warning: a caption body left behind as
plain text inside a figure environment after its \\caption was replaced, which typeset as an unattributed
paragraph under the figure; and a figure environment deleted along with the section it sat in, leaving
\\ref calls with no target.

    python src/lint_tex.py paper/iclr/main.tex
"""
import re, sys

import os as _os
_main = sys.argv[1] if len(sys.argv) > 1 else "paper/iclr/main.tex"
src = open(_main).read()
# follow \input, or a label defined in an included file reads as a dangling \ref
for _inc in re.findall(r"\\input\{([^}]+)\}", src):
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(_main)), _inc)
    for _c in (_p, _p + ".tex"):
        if _os.path.exists(_c):
            src += "\n" + open(_c).read()
            break
bad = 0

def strip_braced(text, macro):
    """Remove every \\macro{...} from text, matching braces so nested groups do not truncate it."""
    out, i = [], 0
    while True:
        j = text.find("\\" + macro + "{", i)
        if j < 0:
            out.append(text[i:]); return "".join(out)
        out.append(text[i:j])
        k, depth = j + len(macro) + 2, 1
        while k < len(text) and depth:
            depth += (text[k] == "{") - (text[k] == "}")
            k += 1
        i = k


# 1. loose prose inside a float. A caption body left behind after its \caption was replaced typesets as
#    an unattributed paragraph and raises no warning, which is how one shipped into the last build.
for env in ("figure", "table"):
    for m in re.finditer(rf"\\begin{{{env}}}.*?\\end{{{env}}}", src, re.S):
        rest = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", m.group(0), flags=re.S)
        rest = strip_braced(rest, "caption")
        # drop every remaining command and its bracket/brace arguments, leaving only bare prose
        rest = re.sub(r"\\[a-zA-Z@]+\s*(\[[^\]]*\])?(\{[^{}]*\})*", " ", rest)
        rest = re.sub(r"[^A-Za-z ]", " ", rest)
        words = [w for w in rest.split() if len(w) > 2 and w.isalpha()]
        if len(words) > 8:
            line = src[:m.start()].count("\n") + 1
            print(f"FAIL {env} at line {line}: {len(words)} words of loose text outside the caption:\n"
                  f"      {' '.join(words[:14])!r}")
            bad += 1

# 2. every \label in a float is referenced, and every \ref resolves
labels = set(re.findall(r"\\label\{((?:fig|tab|sec|app):[^}]+)\}", src))
refs = set(re.findall(r"\\(?:ref|autoref)\{([^}]+)\}", src))
refs |= {"fig:" + x for x in re.findall(r"\\fig\{([^}]+)\}", src)}
refs.discard("fig:#1")
for r in sorted(refs - labels):
    print(f"FAIL \\ref{{{r}}} has no \\label"); bad += 1
for l in sorted(labels - refs):
    if l.startswith(("fig:", "tab:")):
        print(f"FAIL \\label{{{l}}} is never referenced"); bad += 1

# 3. every included graphic exists
import os
root = os.path.dirname(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "paper/iclr/main.tex"))
for g in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", src):
    if not os.path.exists(os.path.join(root, g)):
        print(f"FAIL missing image {g}"); bad += 1

print(("FAILED: %d structural problem(s)" % bad) if bad else "tex structure ok")
sys.exit(1 if bad else 0)
