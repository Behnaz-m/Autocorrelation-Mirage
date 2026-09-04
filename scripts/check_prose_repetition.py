#!/usr/bin/env python3
"""Report repeated prose phrases in a LaTeX manuscript; it never edits text."""
from collections import Counter
from pathlib import Path
import re
text=Path('main2_iberamia.tex').read_text()
text=re.sub(r'\\(?:cite|ref|label|input|includegraphics)\{[^}]*\}',' ',text)
text=re.sub(r'\$.*?\$|\\\[.*?\\\]',' ',text,flags=re.S)
text=re.sub(r'\\[A-Za-z]+(?:\[[^]]*\])?(?:\{[^}]*\})?',' ',text)
words=re.findall(r"[A-Za-z][A-Za-z-]+",text.lower())
allow={'row-wise cross-validation','grouped cross-validation','episode memorization','preprocessing leakage','explicit leakage'}
for n in range(2,6):
 c=Counter(' '.join(words[i:i+n]) for i in range(len(words)-n+1))
 print(f'\n{n}-grams:',[(p,k) for p,k in c.most_common(20) if k>1 and p not in allow][:12])
sentences=re.split(r'[.!?]+',text); starts=Counter(' '.join(re.findall(r'[A-Za-z]+',s.lower())[:3]) for s in sentences)
print('\nsentence openings:',[(x,n) for x,n in starts.most_common(20) if n>1 and x])
print('\ncontent words:',Counter(w for w in words if len(w)>5).most_common(25))
