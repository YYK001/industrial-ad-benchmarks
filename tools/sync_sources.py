"""Copy the INP-Former BTAD entry and its local Python imports from a research workspace.

Explicit source-only export: no recursive copying of datasets, weights or outputs.
Usage: python tools/sync_sources.py /path/to/experiment
"""
import ast
from pathlib import Path
import shutil
import sys


def main():
    source = Path(sys.argv[1]).resolve()
    destination = Path(__file__).resolve().parents[1]
    pending = list((source/'external_baselines/inpformer_btad').glob('*.py'))
    pending += list((source/'external_baselines/inpformer_external').glob('*.py'))
    pending += [source/'DINOv2/nvs/conditional_nvs/metrics.py']
    seen = set()

    def resolve(name):
        parts = name.split('.')
        base = source.joinpath(*parts)
        if parts[0] == 'nvs': base = source/'DINOv2'/Path(*parts)
        for path in (base.with_suffix('.py'), base/'__init__.py'):
            if path.is_file(): pending.append(path)

    while pending:
        path = pending.pop().resolve()
        if path in seen: continue
        seen.add(path)
        relative = path.relative_to(source)
        if any(p in relative.parts for p in ('outputs','downloads','validation','source_snapshot')):
            raise ValueError('non-source dependency: '+str(relative))
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        package = list(relative.with_suffix('').parts[:-1])
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                for alias in node.names: resolve(alias.name)
            elif isinstance(node,ast.ImportFrom):
                name = node.module or ''
                if node.level:
                    prefix = package[:len(package)-node.level+1]
                    name = '.'.join(prefix+([name] if name else []))
                resolve(name)
                for alias in node.names: resolve(name+'.'+alias.name)
        parent = path.parent
        while parent != source:
            if (parent/'__init__.py').is_file(): pending.append(parent/'__init__.py')
            parent = parent.parent
    for path in sorted(seen):
        target = destination/path.relative_to(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    for name in ('inpformer_btad','inpformer_external'):
        shutil.copyfile(source/'external_baselines'/name/'README.md',destination/'external_baselines'/name/'README.md')
    print('Exported',len(seen),'Python source files')
    for p in sorted(seen): print(p.relative_to(source).as_posix())


if __name__ == '__main__': main()
