"""Write an exact-size vector PDF of the ArUco marker the pipeline detects (DICT_4X4_50, id 0, 80 mm).

Usage: uv run python data/aruco/make_marker.py  ->  data/aruco/aruco_4x4_id0_80mm.pdf
"""
from pathlib import Path

import cv2

MARKER_ID, SIDE_MM = 0, 80.0
PT_PER_MM = 72 / 25.4
PAGE_W, PAGE_H = 595.28, 841.89  # A4 in points
OUT = Path(__file__).with_name(f"aruco_4x4_id{MARKER_ID}_{SIDE_MM:.0f}mm.pdf")


def main() -> None:
    bits = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), MARKER_ID, 6)
    side, cell = SIDE_MM * PT_PER_MM, SIDE_MM * PT_PER_MM / 6
    x0, y0 = (PAGE_W - side) / 2, (PAGE_H - side) / 2
    ops = ["0 g"]
    for r in range(6):
        for c in range(6):
            if bits[r, c] == 0:  # black cell; PDF y grows upward
                ops.append(f"{x0 + c * cell:.3f} {y0 + (5 - r) * cell:.3f} {cell:.3f} {cell:.3f} re f")
    quiet = cell  # white margin the detector needs, outlined as a cut guide
    ops.append(f"0.3 w 0.6 G {x0 - quiet:.3f} {y0 - quiet:.3f} {side + 2 * quiet:.3f} {side + 2 * quiet:.3f} re S")
    label = f"ArUco DICT_4X4_50 id {MARKER_ID} - black square side {SIDE_MM:.0f} mm. Print at 100% / actual size, then measure."
    ops.append(f"BT /F1 10 Tf {x0 - quiet:.3f} {y0 - quiet - 24:.3f} Td ({label}) Tj ET")
    stream = "\n".join(ops).encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".encode(),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objs, 1):
        offsets.append(len(pdf))
        pdf += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(pdf)
    pdf += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    pdf += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    pdf += f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    OUT.write_bytes(bytes(pdf))
    print(OUT)


if __name__ == "__main__":
    main()
