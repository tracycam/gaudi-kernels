"""Measure actual OpenAI streaming responses, preserving IDs and arrival times."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen


def completion(base, model, prompt, tokens, barrier=None, draft_token_limit=None):
    body = dict(model=model, prompt=prompt, temperature=0, max_tokens=tokens,
                stream=True, stream_options={'include_usage': True}, return_token_ids=True)
    if draft_token_limit is not None:
        body['vllm_xargs'] = {'draft_token_limit': draft_token_limit}
    if barrier:
        barrier.wait()
    begin = time.perf_counter()
    events, token_ids, text, usage, finish = [], [], [], None, None
    request = Request(base + '/v1/completions', json.dumps(body).encode(),
                      {'Content-Type': 'application/json'})
    with urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                break
            chunk = json.loads(data)
            if 'error' in chunk:
                raise RuntimeError(chunk['error'])
            usage = chunk.get('usage') or usage
            for choice in chunk.get('choices', []):
                ids = choice.get('token_ids') or []
                token_ids.extend(ids)
                text.append(choice.get('text', ''))
                if ids:
                    events.append({'seconds': time.perf_counter() - begin, 'ids': ids})
                finish = choice.get('finish_reason') or finish
    elapsed = time.perf_counter() - begin
    if not events or not usage or usage['completion_tokens'] != len(token_ids):
        raise RuntimeError('Stream ID/usage accounting mismatch or missing token arrivals')
    interval = events[-1]['seconds'] - events[0]['seconds']
    return dict(ids=token_ids, text=''.join(text), usage=usage, finish_reason=finish,
                started_monotonic=begin,
                events=events, wall_seconds=elapsed, ttft_seconds=events[0]['seconds'],
                request_tps=len(token_ids) / elapsed,
                after_first_arrival_tps=((len(token_ids) - len(events[0]['ids'])) / interval
                                         if interval else None))


def concurrent_interval(rows):
    """Count arrivals only while every request is producing tokens.

    Strictly exclude the left endpoint burst, just as single-request decode
    accounting excludes its first burst. No tokens emitted during another
    request's prefill, or after its last token, count toward this interval.
    """
    start = max(row['started_monotonic'] + row['events'][0]['seconds'] for row in rows)
    end = min(row['started_monotonic'] + row['events'][-1]['seconds'] for row in rows)
    if end <= start:
        return dict(seconds=0, tokens_per_request=[0] * len(rows), tps_per_request=None)
    counts = [sum(len(event['ids']) for event in row['events']
                  if start < row['started_monotonic'] + event['seconds'] <= end)
              for row in rows]
    return dict(seconds=end - start, tokens_per_request=counts,
                tps_per_request=[count / (end - start) for count in counts])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='mimo-dflash')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--prompt-file', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--batches', default='1,2,3')
    parser.add_argument('--draft-token-limit', type=int)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if args.reference and args.prompt_file:
        parser.error('--reference and --prompt-file are mutually exclusive')
    prompt = args.prompt_file.read_text() if args.prompt_file else reference['prompt_ids'] if reference else (
        'Record: the verification code for the copper lantern is 7391.\n'
        'Question: What is the verification code? State the code and explain the evidence.\nAnswer:')
    result = dict(status='RUNNING', scope='Actual HTTP token arrivals; prefill included in request TPS', runs=[])

    def save():
        args.out.write_text(json.dumps(result, indent=2) + '\n')

    save()
    try:
        for batch in map(int, args.batches.split(',')):
            barrier = threading.Barrier(batch)
            with ThreadPoolExecutor(max_workers=batch) as pool:
                rows = list(pool.map(lambda _: completion(args.base, args.model, prompt, args.tokens, barrier,
                                                          args.draft_token_limit),
                                     range(batch)))
            if reference:
                expected = reference['runs'][0]['outputs'][0]['ids']
                for row in rows:
                    row['reference_prefix_match'] = row['ids'] == expected[:len(row['ids'])]
            result['runs'].append(dict(batch=batch, requests=rows,
                                      concurrent_decode=concurrent_interval(rows)))
            save()
        result['status'] = 'HTTP_ACCOUNTING_PASS'
    except BaseException as error:
        result.update(status='FAIL', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
