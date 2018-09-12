import argparse
import datetime as dt

from pythonosc import osc_server
from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher

from kalman_mapper import KalmanMapper
import kalman_mapper


class OSCComm(object):
    def __init__(self, ip, port, remote_ip, remote_port):
        self.client = udp_client.SimpleUDPClient(remote_ip, remote_port)
        self.process_note_on = None
        self.process_note_off = None
        self.ip = ip
        self.port = port

    def run_server_forever(self):
        dispatcher = Dispatcher()
        dispatcher.map('/note_on', self._note_on_handler)
        dispatcher.map('/note_off', self._note_off_handler)
        server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), dispatcher)
        print('Serving on {}'.format(server.server_address))
        server.serve_forever()

    def _note_on_handler(self, unused_addr, *args):
        pitch = self._get_input(args)
        if pitch == -1: return
        if self.process_note_on is not None:
            self.process_note_on(pitch)

    def _get_input(self, args):
        if len(args) != 1: return -1
        try:
            val = int(args[0])
            return val
        except ValueError:
            return -1

    def _note_off_handler(self, unused_addr, *args):
        pitch = self._get_input(args)
        if pitch == -1: return
        if self.process_note_off is not None:
            self.process_note_off(pitch)

    def send_hannds_msg(self, is_left_hand, pitch):
        if is_left_hand:
            self.client.send_message('/left_hand', pitch)
        else:
            self.client.send_message('/right_hand', pitch)


class KalmanConnector(object):
    def __init__(self, osc_comm):
        self.mapper = KalmanMapper()
        self.start_t = dt.datetime.now()
        self.osc_comm = osc_comm
        self.osc_comm.process_note_on = self.process_note_on
        self.osc_comm.process_note_off = self.process_note_off

    def process_note_on(self, pitch):
        now = dt.datetime.now()
        delta = (now - self.start_t).seconds
        event = kalman_mapper.MidiEvent(pitch, is_note_on=True, when=delta, is_left=None)
        self.mapper.midi_event(event)
        self.osc_comm.send_hannds_msg(self.mapper.last_was_left_hand, pitch)

    def process_note_off(self, pitch):
        now = dt.datetime.now()
        delta = (now - self.start_t).seconds
        event = kalman_mapper.MidiEvent(pitch, is_note_on=False, when=delta, is_left=None)
        self.mapper.midi_event(event)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote_ip', default='127.0.0.1', help='The IP of the computer running Max/MSP')
    parser.add_argument('--remote_port', type=int, default=5005, help='The port Max/MSP listens on')
    parser.add_argument('--ip', default='127.0.0.1', help='The IP to listen on')
    parser.add_argument('--port', type=int, default=5006, help='The port to listen on')
    args = parser.parse_args()
    osc_comm = OSCComm(args.ip, args.port, args.remote_ip, args.remote_port)
    KalmanConnector(osc_comm)
    osc_comm.run_server_forever()


if __name__ == '__main__':
    main()
