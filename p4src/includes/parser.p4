#include "headers.p4"

#define PORT 8000
#define SSP_ACTION_READ 0
#define SSP_ACTION_INC 1
#define SSP_ACTION_CLOCK 2

struct metadata {
    bit<1> flushQueue;
}

/*************************************************************************
*********************** P A R S E R  ***********************************
*************************************************************************/

parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {
    state start {
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            0x800: parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            17: parse_udp;
            default: accept;
        }
    }
    
    state parse_udp {
        packet.extract(hdr.udp);
        transition select(hdr.udp.dstPort) {
            PORT: parse_ssp;
            default: accept;
        }
    }

    state parse_ssp {
        packet.extract(hdr.ssp);
        transition select(hdr.ssp.actionCode) {
            SSP_ACTION_INC: parse_data;
            default: accept;
        }
    }

    state parse_data {
        packet.extract(hdr.data);
        transition accept;
    }
}