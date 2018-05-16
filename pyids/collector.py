import pyshark


protocols = {'tcp':6, 'udp':17, 'icmp':1}

print("Hello")
capture = pyshark.LiveCapture(interface='veth1')

for packet in capture.sniff_continuously(packet_count=5):

    print("highest layer: {}".format(packet.highest_layer))

    if packet.ip.proto is protocols['tcp']:
        print("tcp")
    if packet.ip.proto is protocols['udp']:
        print("udp")
    if packet.ip.proto is protocols['icmp']:
        print("icmp")

    print("protocol:{}".format(packet.ip.proto))
    print("ip len:{}".format(packet.ip.len))
    print("src_addr:{}".format(packet.ip.src))
    print("dst_addr:{}".format(packet.ip.dst))
