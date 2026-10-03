"""Read-only source dependency inventory; never imports Torch or acquires HPU.

Reachability is conservative (all branches). Non-reachable files are candidates
for review, not proof of dead code: standalone APIs and native bindings also exist.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVING_ROOTS = ('gaudi_kernels.serving.startup.sitecustomize',
                 'gaudi_kernels.serving.bootstrap', 'gaudi_kernels.serving.launch',
                 'gaudi_kernels.serving.worker')
GRAPH_ROOTS = ('gaudi_kernels.graph',)


def inventory(root=ROOT):
    package=root/'python'
    files={}
    for path in sorted((package/'gaudi_kernels').rglob('*.py')):
        parts=list(path.relative_to(package).with_suffix('').parts)
        if parts[-1]=='__init__':parts.pop()
        files['.'.join(parts)]=path
    edges={name:set() for name in files}
    imports={name:set() for name in files}
    dynamic=[]
    for name,path in files.items():
        parent=name if path.name=='__init__.py' else name.rpartition('.')[0]
        # Importing a submodule executes its package initializers too.
        parts=name.split('.')
        edges[name].update('.'.join(parts[:n]) for n in range(1,len(parts))
                           if '.'.join(parts[:n]) in files)
        for node in ast.walk(ast.parse(path.read_text(),filename=str(path))):
            targets=[]
            if isinstance(node,ast.Import):
                targets=[alias.name for alias in node.names]
            elif isinstance(node,ast.ImportFrom):
                base=node.module or ''
                if node.level:
                    base='.'.join(parent.split('.')[:len(parent.split('.'))-node.level+1]+([base] if base else []))
                targets=[base]+[base+'.'+alias.name for alias in node.names]
            elif isinstance(node,ast.Call) and ast.unparse(node.func) in ('importlib.import_module','__import__'):
                if node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):
                    targets=[node.args[0].value]
                else:
                    dynamic.append({'module':name,'line':node.lineno,'call':ast.unparse(node)})
            imports[name].update(targets)
            edges[name].update(target for target in targets if target in files)
    def closure(roots):
        seen=set();pending=list(roots)
        while pending:
            node=pending.pop()
            if node in seen:continue
            if node not in files:raise ValueError('Missing declared entry: '+node)
            seen.add(node);pending.extend(edges[node]-seen)
        return sorted(seen)
    serving=closure(SERVING_ROOTS);graph=closure(GRAPH_ROOTS)
    violations=[{'module':name,'import':target} for name in serving
                for target in sorted(imports[name])
                if target.split('.')[0] in ('benchmarks','tools','reference')]
    return {
        'scope':'static source reachability; not execution coverage or deletion authorization',
        'roots':{'serving':list(SERVING_ROOTS),'owned_graph':list(GRAPH_ROOTS)},
        'serving_modules':serving,'owned_graph_modules':graph,
        'outside_these_roots':sorted(set(files)-set(serving)-set(graph)),
        'dynamic_imports_requiring_review':dynamic,'boundary_violations':violations,
        'files':{name:{'path':str(path.relative_to(root)),
                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                       'imports':sorted(edges[name])} for name,path in files.items()},
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();report=inventory()
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'modules':len(report['files']),'serving':len(report['serving_modules']),
                      'owned_graph':len(report['owned_graph_modules']),
                      'outside_roots':len(report['outside_these_roots']),
                      'boundary_violations':report['boundary_violations']}))
    if report['boundary_violations']:raise SystemExit(1)


if __name__=='__main__':main()
