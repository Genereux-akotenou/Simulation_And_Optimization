import socket
import numpy as np
import threading
import json
import matplotlib.pyplot as plt
from http.server import BaseHTTPRequestHandler, HTTPServer
import time

class RealTimeVisualizer:
    def __init__(self, server_host='', server_port=8080):
        self.server_host = server_host
        self.server_port = server_port
        self.data = []
        self.quit_flag = False
        self.server_thread = threading.Thread(target=self.run_server)
        self.server_thread.daemon = True
        self.server_thread.start()

    def init_communication(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(('localhost', 12345))  # Example port and address, replace with actual values

    def send_data(self, data):
        json_data = json.dumps(data.tolist())
        self.sock.sendall(json_data.encode())

    def receive_data(self):
        while not self.quit_flag:
            try:
                received_data = self.sock.recv(1024)
                if not received_data:
                    break
                self.data.append(json.loads(received_data.decode()))
                self.plot_data()
            except Exception as e:
                print("Error receiving data:", e)
                break

    def plot_data(self):
        plt.clf()
        # Assuming data is a list of arrays
        for array in self.data:
            plt.plot(array)
        plt.draw()
        plt.pause(0.01)

    def run_server(self):
        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Real Time Visualization</title>
                    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.0/chart.min.js"></script>
                    <script>
                        var socket = new WebSocket("ws://%s:%d/");
                        socket.onmessage = function(event) {
                            var data = JSON.parse(event.data);
                            var ctx = document.getElementById('chart').getContext('2d');
                            var chart = new Chart(ctx, {
                                type: 'line',
                                data: {
                                    labels: Array.from({length: data.length}, (_, i) => i),
                                    datasets: [{
                                        label: 'Data',
                                        data: data,
                                        borderColor: 'rgb(75, 192, 192)',
                                        tension: 0.1
                                    }]
                                },
                                options: {
                                    scales: {
                                        x: {
                                            type: 'linear',
                                            position: 'bottom'
                                        }
                                    }
                                }
                            });
                        };
                    </script>
                </head>
                <body>
                    <canvas id="chart"></canvas>
                </body>
                </html>
                """ % (self.server_host, self.server_port)
                self.wfile.write(html.encode())

        server_address = (self.server_host, self.server_port)
        httpd = HTTPServer(server_address, RequestHandler)
        httpd.serve_forever()

    def quit(self):
        self.quit_flag = True
        self.sock.close()

if __name__ == "__main__":
    visualizer = RealTimeVisualizer()
    visualizer.init_communication()

    # Example sending data
    for i in range(10):
        data = np.random.rand(100)
        visualizer.send_data(data)
        time.sleep(1)

    visualizer.quit()
