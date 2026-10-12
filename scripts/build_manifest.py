"""Explicitly reseal the current public snapshot after reviewing changes."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent.parent
def included(p):
    r=p.relative_to(ROOT)
    return not (any(x in r.parts for x in ['.git','__pycache__','.venv','.tmp','sandbox']) or
                r.parts[0] in ['runs','kb','pdfs','models','qa','outputs','dev'] or
                p.name in ['.env','manifest.json'] or p.suffix in ['.pyc','.log'])
if __name__=='__main__':
    rows={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(ROOT.rglob('*')) if p.is_file() and included(p)}
    (ROOT/'data/manifest.json').write_text(json.dumps({'sha256':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Manifest generated for',len(rows),'files. Review changes before publishing.')
