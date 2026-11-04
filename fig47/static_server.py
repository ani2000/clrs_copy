"""Plain static file server for fig47/ (viewer_fig47.html + pre-generated PNG
figures and JSON data files). Zero compute at request time -- no model, no
JAX -- just http.server.SimpleHTTPRequestHandler rooted at this directory.
"""
import os
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial

PORT = 8767

if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    handler = partial(SimpleHTTPRequestHandler, directory=here)
    print(f'serving fig47/ static files on http://0.0.0.0:{PORT}/viewer_fig47.html', flush=True)
    ThreadingHTTPServer(('0.0.0.0', PORT), handler).serve_forever()
