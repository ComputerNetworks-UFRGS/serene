from scapy.all import sendp, srp, srp1, conf, Ether, IP, UDP
import torch
import time
from protocol import Gradient, SspHeader, assemble_pkt, unquantize

CHUNK_SIZE = 500
conf.use_pcap = True
conf.verb = 0
conf.layers.filter([Ether, IP, UDP, SspHeader, Gradient])


def chunkify(iterable, size=CHUNK_SIZE):
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def read_rows(n, worker_id, worker_clock):
    """Read the first n rows"""
    pkts = [assemble_pkt(worker_id, worker_clock, x, "read_row") for x in range(n)]
    responses = []

    for chunk in chunkify(pkts):
        res, nres = srp(chunk, timeout=3, retry=-1000, filter="udp")
        answers = [answer.answer for answer in res]
        if len(answers) != len(chunk):
            raise RuntimeError("Deu ruim! Pacotes sem resposta: {}".format(nres))
        responses.extend(answers)


    responses = sorted(responses, key=lambda obj: obj.grad_segment)
    grads = [res.getlayer(Gradient).get_grads() for res in responses]
    tensors = [torch.tensor(grad) for grad in grads]

    # Concatenating all tensors into one
    return unquantize(torch.cat(tensors))


def inc_rows(n, values, worker_id, worker_clock):
    """Increment the value of the first n rows"""
    pkts = [
        assemble_pkt(worker_id, worker_clock, x, "inc", value)
        for x, value in zip(range(n), values)
    ]

    sendp(pkts)


def clock(worker_id, worker_clock):
    """Inform the switch that the worker has completed one clock.
    Only return when the worker can proceed training"""

    pkt = assemble_pkt(worker_id, worker_clock, 0, "clock")
    response = srp1(pkt)
    if response is None:
        raise RuntimeError(
            f"Clock response {worker_clock} for worker {worker_id} timed out"
        )
