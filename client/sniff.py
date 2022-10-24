from scapy.all import sniff
from ssp import SspHeader

registered = []


def handle_pkt(p):
    p.getlayer(SspHeader).show2()
    registered.append(p.hashret())
    print(f"{p.hashret()} Response to prev: {p.hashret() == registered[-1]}")


sniff(prn=handle_pkt, filter="udp")
