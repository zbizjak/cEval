"""Robust VTP polyline reader/writer for the metric-sensitivity study.

``metrics_lib.load_vtp`` uses ``vtkXMLPolyDataReader``, which on VTK 9.1 fails to
parse the base64-appended, zlib-compressed VTP files produced for TopCoW CT and
AortaSeg24 (it aborts with an XML parse error at the ``AppendedData`` marker).
TopCoW MR is plain ASCII and reads fine. Rather than pin a VTK version or touch
``metrics_lib``, this module parses the container directly.

Supports the subset of the VTP format actually present in this data:
  - ``format="ascii"``
  - ``format="appended"`` with ``encoding="base64"``, raw or zlib-compressed,
    with UInt32 or UInt64 headers
  - ``format="binary"`` (inline base64), raw or zlib-compressed

Returns the same contract as ``metrics_lib.load_vtp``:
    points (N,3) float64, edges (E,2) int64 undirected, deduplicated, sorted.
"""
from __future__ import annotations

import base64
import re
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np

_VTK_DTYPES = {
    "Int8": np.int8, "UInt8": np.uint8,
    "Int16": np.int16, "UInt16": np.uint16,
    "Int32": np.int32, "UInt32": np.uint32,
    "Int64": np.int64, "UInt64": np.uint64,
    "Float32": np.float32, "Float64": np.float64,
}


def _decode_appended_block(raw: bytes, dtype, header_dtype, compressed):
    """Decode one base64 block from the AppendedData payload.

    Uncompressed layout:  [header: nbytes][data]
    Compressed layout:    [header: nblocks, blocksize, last_blocksize,
                           csize_1..csize_nblocks][zlib blocks]
    Both header and data are base64-encoded, and VTK encodes them *separately*,
    so the header must be decoded first to learn how many bytes to take next.
    """
    hsize = np.dtype(header_dtype).itemsize

    if not compressed:
        # base64 of an hsize-byte header occupies ceil(hsize/3)*4 chars
        nchars = ((hsize + 2) // 3) * 4
        nbytes = int(np.frombuffer(
            base64.b64decode(raw[:nchars]), dtype=header_dtype, count=1)[0])
        data = base64.b64decode(raw[nchars:])[:nbytes]
        return np.frombuffer(data, dtype=dtype)

    # Compressed: read the 3-word prefix to learn nblocks, then the full header.
    nchars3 = ((3 * hsize + 2) // 3) * 4
    head3 = np.frombuffer(
        base64.b64decode(raw[:nchars3]), dtype=header_dtype, count=3)
    nblocks = int(head3[0])
    hbytes = (3 + nblocks) * hsize
    nchars_h = ((hbytes + 2) // 3) * 4
    header = np.frombuffer(
        base64.b64decode(raw[:nchars_h]), dtype=header_dtype, count=3 + nblocks)
    csizes = header[3:].astype(np.int64)

    payload = base64.b64decode(raw[nchars_h:])
    out, off = [], 0
    for csize in csizes:
        out.append(zlib.decompress(payload[off:off + int(csize)]))
        off += int(csize)
    return np.frombuffer(b"".join(out), dtype=dtype)


def _decode_inline(text: str, dtype, header_dtype, compressed, fmt):
    if fmt == "ascii":
        return np.fromstring(text, sep=" ", dtype=np.float64).astype(dtype)
    return _decode_appended_block(
        text.strip().encode("ascii"), dtype, header_dtype, compressed)


def read_vtp_polylines(path):
    """Read a VTP file containing polylines -> (points (N,3), edges (E,2))."""
    path = Path(path)
    data = path.read_bytes()

    # Split off AppendedData before XML parsing: its payload is not valid XML,
    # which is exactly what makes vtkXMLPolyDataReader abort on these files.
    appended = b""
    m = re.search(rb"<AppendedData[^>]*>", data)
    if m:
        start = data.index(b"_", m.end()) + 1
        end = data.find(b"</AppendedData>", start)
        appended = data[start:end].strip()
        # Drop the AppendedData element entirely; whatever tags followed it in
        # the original (typically just </VTKFile>) are preserved.
        data = data[:m.start()] + data[end + len(b"</AppendedData>"):]

    root = ET.fromstring(data.decode("utf-8", errors="replace"))
    header_dtype = _VTK_DTYPES[root.get("header_type", "UInt32")]
    compressed = root.get("compressor") is not None

    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"{path.name}: no <Piece>")

    def array(parent_tag, name=None, index=0):
        parent = piece.find(parent_tag)
        if parent is None:
            return None
        arrays = parent.findall("DataArray")
        if name is not None:
            arrays = [a for a in arrays if a.get("Name") == name]
        if index >= len(arrays):
            return None
        da = arrays[index]
        dtype = _VTK_DTYPES[da.get("type")]
        fmt = da.get("format")
        if fmt == "appended":
            off = int(da.get("offset", 0))
            return _decode_appended_block(
                appended[off:], dtype, header_dtype, compressed)
        return _decode_inline(da.text or "", dtype, header_dtype, compressed, fmt)

    pts = array("Points")
    if pts is None or pts.size == 0:
        return np.zeros((0, 3), np.float64), np.zeros((0, 2), np.int64)
    points = np.asarray(pts, dtype=np.float64).reshape(-1, 3)

    conn = array("Lines", name="connectivity")
    offs = array("Lines", name="offsets")
    if conn is None:  # older layout: arrays are unnamed, connectivity first
        conn, offs = array("Lines", index=0), array("Lines", index=1)
    if conn is None or offs is None or len(offs) == 0:
        return points, np.zeros((0, 2), np.int64)

    conn = np.asarray(conn, dtype=np.int64)
    offs = np.asarray(offs, dtype=np.int64)

    edges = set()
    prev = 0
    for end in offs:
        cell = conn[prev:int(end)]
        for a, b in zip(cell[:-1], cell[1:]):
            a, b = int(a), int(b)
            if a != b:
                edges.add((min(a, b), max(a, b)))
        prev = int(end)

    edges_arr = (np.array(sorted(edges), dtype=np.int64)
                 if edges else np.zeros((0, 2), dtype=np.int64))
    return points, edges_arr


def write_vtp_polylines(path, points, edges):
    """Write points+edges as an ASCII VTP with one line cell per edge.

    ASCII is intentional: it is what TopCoW MR already uses, it reads back
    through both this module and ``metrics_lib.load_vtp``, and it keeps the
    perturbed graphs inspectable in a text editor.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    n, e = len(points), len(edges)

    pts_txt = " ".join(f"{v:.6f}" for v in points.ravel())
    conn_txt = " ".join(str(int(v)) for v in edges.ravel())
    offs_txt = " ".join(str(2 * (i + 1)) for i in range(e))

    xml = f"""<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
  <PolyData>
    <Piece NumberOfPoints="{n}" NumberOfVerts="0" NumberOfLines="{e}" NumberOfStrips="0" NumberOfPolys="0">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">{pts_txt}</DataArray>
      </Points>
      <Lines>
        <DataArray type="Int64" Name="connectivity" format="ascii">{conn_txt}</DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii">{offs_txt}</DataArray>
      </Lines>
    </Piece>
  </PolyData>
</VTKFile>
"""
    Path(path).write_text(xml)


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        p, e = read_vtp_polylines(f)
        print(f"{Path(f).name}: {len(p)} points, {len(e)} edges")
