"""Read pinned local prompt tokens without re-rendering or exposing answer checks."""
import hashlib
import json


def load(path, *, needed, maximum_length):
    raw=path.read_bytes()
    rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows)<needed:
        raise ValueError('Insufficient distinct corpus requests')
    selected=rows[:needed]
    if len({r['id'] for r in selected})!=needed:
        raise ValueError('Duplicate corpus sample IDs')
    prompts=[];identities=[]
    for row in selected:
        ids=row['prompt_token_ids']
        if not isinstance(ids,list) or not ids or any(type(v) is not int or v<0 for v in ids):
            raise ValueError('Corpus requires original integer token IDs')
        if len(ids)>maximum_length:
            raise ValueError('Context exceeds configured model capacity; refusing silent truncation')
        digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        if digest!=row['prompt_token_ids_sha256']:
            raise ValueError('Prompt token identity mismatch')
        prompts.append(dict(prompt_token_ids=ids))
        identities.append(dict(id=row['id'],length=len(ids),prompt_token_ids_sha256=digest,
                               messages_sha256=row.get('messages_sha256')))
    return prompts,dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest(),requests=identities)
