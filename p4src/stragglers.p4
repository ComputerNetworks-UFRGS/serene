/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>

#include "includes/constants.p4"
#include "includes/parser.p4"
#include "includes/queue.p4"

/*************************************************************************
************   C H E C K S U M    V E R I F I C A T I O N   *************
*************************************************************************/

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {
    apply {}
}

/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/
register< bit<32> >(VALUES_PER_ROW * AGGR_ROWS) sspTable;


control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {

    // ======== Synchronization Barriers =======
    register< bit<MAX_WORKERS> >(BUFFER_SIZE) buffer;
    register< bit<16> >(1) lowClockReg;
    register< bit<16> >(1) highClockReg;
    bit<16> window = 0;

    action set_window(bit<16> currentWindow) {
        window = currentWindow;
    }

    table window_table {
        key = {}
        actions = { set_window; }
    }


    action bounce_pkt() {
        standard_metadata.egress_spec = standard_metadata.ingress_port;

        bit<48> tmpEth = hdr.ethernet.dstAddr;
        hdr.ethernet.dstAddr = hdr.ethernet.srcAddr;
        hdr.ethernet.srcAddr = tmpEth;

        bit<32> tmpIp = hdr.ipv4.dstAddr;
        hdr.ipv4.dstAddr = hdr.ipv4.srcAddr;
        hdr.ipv4.srcAddr = tmpIp;

        bit<16> tmpPort = hdr.udp.dstPort;
        hdr.udp.dstPort = hdr.udp.srcPort;
        hdr.udp.srcPort = tmpPort;
    }

    // This included file contains the 'readRow' and 'increment' actions. These actions 
    // are generated automatically during build, according to the aggregation settings. 
    // They were extracted to their own file to make the generation process easier.
    #include "includes/ssp.p4"

    apply {
        if (hdr.ssp.isValid()) {
            if (hdr.ssp.actionCode == SSP_ACTION_INC) {
                increment();
                mark_to_drop(standard_metadata);
            } else if (hdr.ssp.actionCode == SSP_ACTION_READ) {
                readRow();
                bounce_pkt();
            } else if (hdr.ssp.actionCode == SSP_ACTION_CLOCK) {
                // ===== Buffer Update =====
                bit<16> workerClock = hdr.ssp.workerClock;
                bit<16> workerId = hdr.ssp.workerId;

                // Cast to bit<4> because BMv2 can only shift 8 bits at once 
                bit<MAX_WORKERS> workerMask = (bit<MAX_WORKERS>) 1 << (bit<4>) workerId;

                bit<32> updateSlot = (bit<32>) workerClock & (BUFFER_SIZE - 1);

                // Bitmap Update
                bit<MAX_WORKERS> bitmap;
                buffer.read(bitmap, updateSlot);
                bitmap = bitmap | workerMask;

                // If all workers already passed this iteration, reset the slot
                if (~bitmap == 0)
                    bitmap = 0;
                buffer.write(updateSlot, bitmap);

                // ===== Barriers Update and Clock Control =====
                bit<16> lowClock;
                lowClockReg.read(lowClock, 0);
                bit<16> highClock;
                highClockReg.read(highClock, 0);

                window_table.apply();

                if (bitmap == 0) {
                    lowClock = workerClock;
                    lowClockReg.write(0, lowClock);

                    if (highClock - lowClock <= window) {
                        meta.flushQueue = 1;
                    }
                } else if (workerClock > highClock) {
                    highClock = workerClock;
                    highClockReg.write(0, highClock);
                }

                if (workerClock - lowClock > window) {
                    enqueue(workerId);
                    mark_to_drop(standard_metadata);
                } else {
                    bounce_pkt();
                }
            }
        }
    }
}

/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {
    action set_worker_data(bit<48> macAddr, bit<32> ipAddr, bit<16> workerId) {
        hdr.ethernet.dstAddr = macAddr;
        hdr.ipv4.dstAddr = ipAddr;
        hdr.ssp.workerId = workerId;
    }
    
    table worker_data {
        key = {
            standard_metadata.egress_port: exact;
        }
        actions = {
            set_worker_data;
            NoAction;
        }
    }
    apply {
        // Set to 0 to disable udp checksum validation
        hdr.udp.checksum = 0;

        // Identifying packets that should flush the queue
        if (meta.flushQueue == 1 || standard_metadata.instance_type == 2) {
            bit<32> remaining;
            queueElements.read(remaining, 0);

            // While
            if (remaining != 0) {
                bit<16> workerId;
                dequeue(workerId);
                clone(CloneType.E2E, (bit<32>) workerId);
            }
        }
        // Fix headers before sending
        worker_data.apply();
    }
}

/*************************************************************************
*************   C H E C K S U M    C O M P U T A T I O N   **************
*************************************************************************/

control MyComputeChecksum(inout headers hdr, inout metadata meta) {
    apply {
        update_checksum(
            hdr.ipv4.isValid(),
            { hdr.ipv4.version,
              hdr.ipv4.ihl,
              hdr.ipv4.diffserv,
              hdr.ipv4.totalLen,
              hdr.ipv4.identification,
              hdr.ipv4.flags,
              hdr.ipv4.fragOffset,
              hdr.ipv4.ttl,
              hdr.ipv4.protocol,
              hdr.ipv4.srcAddr,
              hdr.ipv4.dstAddr },
            hdr.ipv4.hdrChecksum,
            HashAlgorithm.csum16);
    }
}

/*************************************************************************
***********************  D E P A R S E R  *******************************
*************************************************************************/

control MyDeparser(packet_out packet, in headers hdr) {
    apply { 
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.udp);
        packet.emit(hdr.ssp);
        packet.emit(hdr.data);
    }
}

/*************************************************************************
***********************  S W I T C H  *******************************
*************************************************************************/

V1Switch(
    MyParser(),
    MyVerifyChecksum(),
    MyIngress(),
    MyEgress(),
    MyComputeChecksum(),
    MyDeparser()
) main;
