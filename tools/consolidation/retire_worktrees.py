#!/usr/bin/env python3
"""Archive inactive clean experiment worktrees, retaining Git branches/assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def live_references(directory):
    root=str(directory.resolve())
    found=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:
            if process.stat().st_uid!=os.getuid():continue
            candidates=[process/'cwd',*(process/'fd').iterdir()]
            for item in candidates:
                try:target=os.readlink(item)
                except OSError:continue
                if target==root or target.startswith(root+'/'):
                    found.append({'pid':int(process.name),'reference':str(item),'target':target})
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:
            helper="import json,os,sys;from pathlib import Path;p=Path('/proc')/sys.argv[1];r=sys.argv[2];out=[];items=[p/'cwd',*(p/'fd').iterdir()];"\
                   "[(out.append({'pid':int(sys.argv[1]),'reference':str(i),'target':os.readlink(i)}) if os.readlink(i)==r or os.readlink(i).startswith(r+'/') else None) for i in items if i.is_symlink()];print(json.dumps(out))"
            try:found.extend(json.loads(subprocess.check_output(['sudo','-n','python3','-c',helper,process.name,root],text=True)))
            except subprocess.CalledProcessError:raise RuntimeError('Cannot audit own process references: '+process.name)
    return found


def retire(repository, destination, keep, apply=False):
    records=[]
    listed=subprocess.check_output(['git','worktree','list','--porcelain'],cwd=repository,text=True)
    destination.mkdir(parents=True,exist_ok=True)
    index=destination/'index.json'
    for part in listed.strip().split('\n\n'):
        row=dict(line.split(' ',1) for line in part.splitlines() if ' ' in line)
        directory=Path(row['worktree']).resolve()
        if directory.name in keep or directory==repository.resolve():
            row['disposition']='retain-active-or-designated';records.append(row);continue
        status=subprocess.check_output(['git','status','--porcelain'],cwd=directory,text=True)
        references=live_references(directory)
        row.update(clean=not status,live_references=references)
        if status or references:
            row['disposition']='retain-dirty-or-active';records.append(row);continue
        # User explicitly requested retirement of these historical experiment
        # worktrees. Canonical serving no longer imports paths under them.
        row['disposition']='archive-candidate'
        if apply:
            archive=destination/(directory.name+'-'+row['HEAD'][:12]+'.tar')
            if archive.exists():raise FileExistsError(archive)
            with tarfile.open(archive,'w',dereference=False) as output:
                for entry in sorted(directory.iterdir()):
                    if entry.name=='.git':continue
                    output.add(entry,arcname=directory.name+'/'+entry.name,recursive=True)
            # Validate the archive can be read completely BEFORE retiring. This
            # includes ignored binaries/output assets, not just tracked files.
            with tarfile.open(archive) as saved:
                members=saved.getmembers()
                for member in members:
                    if member.isfile():
                        original=directory.parent/member.name
                        content=saved.extractfile(member)
                        digest=hashlib.sha256()
                        for chunk in iter(lambda:content.read(1<<20),b''):digest.update(chunk)
                        original_digest=hashlib.sha256()
                        with original.open('rb') as stream:
                            for chunk in iter(lambda:stream.read(1<<20),b''):original_digest.update(chunk)
                        if digest.digest()!=original_digest.digest():raise RuntimeError('Archive verification failed: '+member.name)
            digest=hashlib.sha256()
            with archive.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
            row.update(archive=str(archive),archive_sha256=digest.hexdigest(),archive_bytes=archive.stat().st_size,
                       archived_entries=len(members),git_branch_retained=True)
            # A final activity/status check closes the archive-time race. No
            # --force: any untracked/dirty or newly active tree is retained.
            if live_references(directory) or subprocess.check_output(['git','status','--porcelain'],cwd=directory,text=True):
                row['disposition']='archived-but-retained-changed'
            else:
                subprocess.run(['git','worktree','remove',str(directory)],cwd=repository,check=True)
                row['disposition']='archived-and-retired'
        records.append(row)
        index.write_text(json.dumps({'worktrees_before':len(listed.strip().split('\n\n')),'records':records},indent=2)+'\n')
    final=subprocess.check_output(['git','worktree','list','--porcelain'],cwd=repository,text=True)
    result={'worktrees_before':len(listed.strip().split('\n\n')),'worktrees_after':len(final.strip().split('\n\n')),
            'scope':'Clean inactive historical experiment trees only; branches retained; ignored assets verified before removal',
            'records':records}
    index.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repository',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--keep',required=True)
    p.add_argument('--apply',action='store_true');a=p.parse_args()
    r=retire(a.repository,a.archive,set(a.keep.split(',')),a.apply)
    print('Worktrees:',r['worktrees_before'],'->',r['worktrees_after'])
