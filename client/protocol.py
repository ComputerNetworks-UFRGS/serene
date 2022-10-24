from scapy.all import (
    BitField,
    BitEnumField,
    Packet,
    SignedIntField,
    bind_layers,
    UDP,
    Ether,
    IP,
    get_if_addr,
    conf,
)
import torch

from constants import GRADS_PER_PKT, SCALING_FACTOR, MAX_INT, MIN_INT

actions = {
    "read_row": 0,
    "inc": 1,
    "clock": 2,
}


def quantize(fp):
    return (fp * SCALING_FACTOR).type(torch.IntTensor)


def unquantize(v):
    return v / SCALING_FACTOR


# Protocol definitions
class SspHeader(Packet):
    fields_desc = [
        BitField("worker_id", 0, 16),
        BitField("worker_clock", 0, 16),
        BitField("grad_segment", 0, 16),
        BitEnumField("action", 0, 8, actions),
    ]

    def answers(self, other):
        if not isinstance(other, SspHeader):
            return 0
        return self.grad_segment == other.grad_segment and self.action == other.action


class Gradient(Packet):
    fields_desc = [SignedIntField(f"grad_{i}", 0) for i in range(GRADS_PER_PKT)]

    def get_grads(self):
        return [
            self.grad_0,
            self.grad_1,
            self.grad_2,
            self.grad_3,
            self.grad_4,
            self.grad_5,
            self.grad_6,
            self.grad_7,
            self.grad_8,
            self.grad_9,
            self.grad_10,
            self.grad_11,
            self.grad_12,
            self.grad_13,
            self.grad_14,
            self.grad_15,
            self.grad_16,
            self.grad_17,
            self.grad_18,
            self.grad_19,
            self.grad_20,
            self.grad_21,
            self.grad_22,
            self.grad_23,
            self.grad_24,
            self.grad_25,
            self.grad_26,
            self.grad_27,
            self.grad_28,
            self.grad_29,
            self.grad_30,
            self.grad_31,
        ]


bind_layers(UDP, SspHeader, sport=8000, dport=8000)
bind_layers(SspHeader, Gradient, action=0)

BASE_PKT = (
    Ether(dst="ff:ff:ff:ff:ff:ff")
    / IP(dst="10.0.0.1", src=get_if_addr(conf.iface))
    / UDP(sport=8000, dport=8000)
)


def assemble_pkt(worker_id, clock, segment, action="read_row", grads=None):
    """Assemble an packet with the defined protocol"""
    pkt = BASE_PKT / SspHeader(
        worker_id=worker_id, worker_clock=clock, grad_segment=segment, action=action
    )
    if grads is None:
        size = 0
    else:
        size = min(torch.numel(grads), GRADS_PER_PKT)
        grads = quantize(grads)
    pkt = pkt / Gradient(**{f"grad_{i}": grads[i] for i in range(size)})

    return pkt
