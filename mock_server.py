from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class SimpleServer(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps([0.0]*7).encode('utf-8')) # Just array instead of {"action":...} since loads() parses this directly

def run(server_class=HTTPServer, handler_class=SimpleServer, port=10093):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd on port {port}...')
    httpd.serve_forever()

if __name__ == "__main__":
    run()
