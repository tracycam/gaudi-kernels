"""HTTP accounting must distinguish an accepted-token burst from one token."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest

from tools.validation.executor.api_smoke import completion, concurrent_interval


class ApiArrivalTests(unittest.TestCase):
    def serve(self, completion_tokens):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                assert body['stream'] and body['return_token_ids']
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                chunks = [dict(choices=[dict(token_ids=[11, 12], text='ab')]),
                          dict(choices=[dict(token_ids=[13], text='c', finish_reason='length')]),
                          dict(choices=[], usage=dict(completion_tokens=completion_tokens))]
                for chunk in chunks:
                    self.wfile.write(b'data: ' + json.dumps(chunk).encode() + b'\n\n')
                    self.wfile.flush()
                self.wfile.write(b'data: [DONE]\n\n')

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return 'http://127.0.0.1:' + str(server.server_port)

    def test_token_bursts_and_terminal_usage(self):
        result = completion(self.serve(3), 'model', 'prompt', 3)
        self.assertEqual(result['ids'], [11, 12, 13])
        self.assertEqual([len(event['ids']) for event in result['events']], [2, 1])
        self.assertEqual(result['text'], 'abc')
        self.assertEqual(result['finish_reason'], 'length')
        self.assertGreater(result['request_tps'], 0)

    def test_missing_stream_token_cannot_be_silently_counted(self):
        with self.assertRaisesRegex(RuntimeError, 'accounting mismatch'):
            completion(self.serve(4), 'model', 'prompt', 3)

    def test_concurrent_window_excludes_prefill_and_finished_requests(self):
        def row(begin, events):
            return dict(started_monotonic=begin,
                        events=[dict(seconds=seconds, ids=list(range(count))) for seconds, count in events])
        result = concurrent_interval([row(10, [(1, 2), (2, 3), (3, 4), (4, 5)]),
                                      row(11, [(1, 6), (2, 7), (4, 8)])])
        self.assertEqual(result['seconds'], 2)
        self.assertEqual(result['tokens_per_request'], [9, 7])
        self.assertEqual(result['tps_per_request'], [4.5, 3.5])
        self.assertIsNone(concurrent_interval([row(0, [(1, 2), (2, 3)]),
                                               row(3, [(1, 2), (2, 3)])])['tps_per_request'])


if __name__ == '__main__':
    unittest.main()
