action aggregate(bit<16> baseIndex, bit<32> element, bit<32> value) {
    bit<32> currentValue;
    bit<32> index = (bit<32>) baseIndex * VALUES_PER_ROW + element;
    sspTable.read(currentValue, index);
    sspTable.write(index, currentValue + value);
}


action readRow() {
    // Updating lengths
    hdr.ipv4.totalLen = hdr.ipv4.totalLen + 128;
    hdr.udp.length = hdr.udp.length + 128;

    // Read a table row
    hdr.data.setValid();
    bit<32> index = (bit<32>) hdr.ssp.gradSegment * VALUES_PER_ROW;
    sspTable.read(hdr.data.value00, index + 0);
    sspTable.read(hdr.data.value01, index + 1);
    sspTable.read(hdr.data.value02, index + 2);
    sspTable.read(hdr.data.value03, index + 3);
    sspTable.read(hdr.data.value04, index + 4);
    sspTable.read(hdr.data.value05, index + 5);
    sspTable.read(hdr.data.value06, index + 6);
    sspTable.read(hdr.data.value07, index + 7);
    sspTable.read(hdr.data.value08, index + 8);
    sspTable.read(hdr.data.value09, index + 9);
    sspTable.read(hdr.data.value10, index + 10);
    sspTable.read(hdr.data.value11, index + 11);
    sspTable.read(hdr.data.value12, index + 12);
    sspTable.read(hdr.data.value13, index + 13);
    sspTable.read(hdr.data.value14, index + 14);
    sspTable.read(hdr.data.value15, index + 15);
    sspTable.read(hdr.data.value16, index + 16);
    sspTable.read(hdr.data.value17, index + 17);
    sspTable.read(hdr.data.value18, index + 18);
    sspTable.read(hdr.data.value19, index + 19);
    sspTable.read(hdr.data.value20, index + 20);
    sspTable.read(hdr.data.value21, index + 21);
    sspTable.read(hdr.data.value22, index + 22);
    sspTable.read(hdr.data.value23, index + 23);
    sspTable.read(hdr.data.value24, index + 24);
    sspTable.read(hdr.data.value25, index + 25);
    sspTable.read(hdr.data.value26, index + 26);
    sspTable.read(hdr.data.value27, index + 27);
    sspTable.read(hdr.data.value28, index + 28);
    sspTable.read(hdr.data.value29, index + 29);
    sspTable.read(hdr.data.value30, index + 30);
    sspTable.read(hdr.data.value31, index + 31);
}

action increment() {
    aggregate(hdr.ssp.gradSegment, 0, hdr.data.value00);
    aggregate(hdr.ssp.gradSegment, 1, hdr.data.value01);
    aggregate(hdr.ssp.gradSegment, 2, hdr.data.value02);
    aggregate(hdr.ssp.gradSegment, 3, hdr.data.value03);
    aggregate(hdr.ssp.gradSegment, 4, hdr.data.value04);
    aggregate(hdr.ssp.gradSegment, 5, hdr.data.value05);
    aggregate(hdr.ssp.gradSegment, 6, hdr.data.value06);
    aggregate(hdr.ssp.gradSegment, 7, hdr.data.value07);
    aggregate(hdr.ssp.gradSegment, 8, hdr.data.value08);
    aggregate(hdr.ssp.gradSegment, 9, hdr.data.value09);
    aggregate(hdr.ssp.gradSegment, 10, hdr.data.value10);
    aggregate(hdr.ssp.gradSegment, 11, hdr.data.value11);
    aggregate(hdr.ssp.gradSegment, 12, hdr.data.value12);
    aggregate(hdr.ssp.gradSegment, 13, hdr.data.value13);
    aggregate(hdr.ssp.gradSegment, 14, hdr.data.value14);
    aggregate(hdr.ssp.gradSegment, 15, hdr.data.value15);
    aggregate(hdr.ssp.gradSegment, 16, hdr.data.value16);
    aggregate(hdr.ssp.gradSegment, 17, hdr.data.value17);
    aggregate(hdr.ssp.gradSegment, 18, hdr.data.value18);
    aggregate(hdr.ssp.gradSegment, 19, hdr.data.value19);
    aggregate(hdr.ssp.gradSegment, 20, hdr.data.value20);
    aggregate(hdr.ssp.gradSegment, 21, hdr.data.value21);
    aggregate(hdr.ssp.gradSegment, 22, hdr.data.value22);
    aggregate(hdr.ssp.gradSegment, 23, hdr.data.value23);
    aggregate(hdr.ssp.gradSegment, 24, hdr.data.value24);
    aggregate(hdr.ssp.gradSegment, 25, hdr.data.value25);
    aggregate(hdr.ssp.gradSegment, 26, hdr.data.value26);
    aggregate(hdr.ssp.gradSegment, 27, hdr.data.value27);
    aggregate(hdr.ssp.gradSegment, 28, hdr.data.value28);
    aggregate(hdr.ssp.gradSegment, 29, hdr.data.value29);
    aggregate(hdr.ssp.gradSegment, 30, hdr.data.value30);
    aggregate(hdr.ssp.gradSegment, 31, hdr.data.value31);
}