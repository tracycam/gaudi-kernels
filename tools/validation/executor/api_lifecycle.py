"""Exercise public serving admission, EOS, cancellation and slot reuse."""
import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='mimo-dflash')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = dict(status='RUNNING', cases=[])

    def save():
        args.out.write_text(json.dumps(result, indent=2) + '\n')

    def post(path, body):
        try:
            with urlopen(Request(args.base + path, json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'}), timeout=120) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    save()
    try:
        for options in (dict(temperature=0.7), dict(temperature=0, logprobs=0),
                        dict(temperature=0, presence_penalty=1)):
            status, body = post('/v1/completions', dict(model=args.model, prompt='Hello', max_tokens=2, **options))
            assert status == 400, (options, status, body)
            result['cases'].append(dict(kind='reject_unsupported_sampling', options=options, status=status))
            assert urlopen(args.base + '/health', timeout=10).status == 200
            save()
        status, body = post('/collective_rpc', {})
        assert status == 404, (status, body)
        result['cases'].append(dict(kind='no_debug_rpc', status=status))

        def chat():
            status, body = post('/v1/chat/completions', dict(
                model=args.model, messages=[dict(role='user', content='Reply exactly OK.')],
                temperature=0, max_tokens=16, chat_template_kwargs=dict(enable_thinking=False)))
            assert status == 200, body
            choice = body['choices'][0]
            assert choice['message']['content'].strip() == 'OK', body
            assert choice['finish_reason'] == 'stop', body
            result['cases'].append(dict(kind='chat_eos_and_slot_reuse', response=body))
            save()

        chat()
        request = Request(args.base + '/v1/completions', json.dumps(dict(
            model=args.model, prompt='Write a detailed essay of 3000 words about measurement.',
            temperature=0, max_tokens=1024, stream=True, return_token_ids=True)).encode(),
            {'Content-Type': 'application/json'})
        with urlopen(request, timeout=120) as response:
            for line in response:
                if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                    continue
                event = json.loads(line[6:])
                assert 'error' not in event, event
                if any(choice.get('token_ids') for choice in event.get('choices', [])):
                    result['cases'].append(dict(kind='cancel_after_first_token', event=event))
                    break
            else:
                raise RuntimeError('No token arrived before cancellation')
        save()
        # Endpoint behavior verifies released ownership without production RPC.
        time.sleep(0.5)
        chat()
        status, body = post('/v1/completions', dict(model=args.model, prompt='The next number after 1 is',
                                                 temperature=0, max_tokens=1))
        assert status == 200 and body['usage']['completion_tokens'] == 1, body
        assert body['choices'][0]['finish_reason'] == 'length', body
        result['cases'].append(dict(kind='one_token_limit', response=body))
        result['status'] = 'PUBLIC_API_LIFECYCLE_PASS'
    except BaseException as error:
        result.update(status='FAIL', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
