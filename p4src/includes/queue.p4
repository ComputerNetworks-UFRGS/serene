register< bit<16> >(MAX_WORKERS * 2) waitingQueue;
register< bit<32> >(1) queueElements;

action enqueue(bit<16> workerId) {
    bit<32> queueEnd;
    queueElements.read(queueEnd, 0);
    waitingQueue.write(queueEnd, workerId);
    queueElements.write(0, queueEnd + 1);
}

action dequeue(out bit<16> workerId) {
    bit<32> queueEnd;
    queueElements.read(queueEnd, 0);
    waitingQueue.read(workerId, queueEnd - 1);
    queueElements.write(0, queueEnd - 1);
}