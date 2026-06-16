#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path


STATUS_RE = re.compile(r"\b(OK|ERROR)\b")


def check_poc_md5_lines(dec_log_path: Path) -> bool:
    if not dec_log_path.exists():
        print(f"[FAIL] decoder log does not exist: {dec_log_path}")
        return False

    total_poc_lines = 0
    ok_lines = 0
    error_lines = 0
    unknown_status_lines = []

    with dec_log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.rstrip("\n")
            stripped = raw.lstrip()

            # VTM decoder checksum lines usually start with "POC"
            if not stripped.startswith("POC"):
                continue

            total_poc_lines += 1

            m = STATUS_RE.search(stripped)
            if not m:
                unknown_status_lines.append((line_no, raw))
                continue

            status = m.group(1)
            if status == "OK":
                ok_lines += 1
            elif status == "ERROR":
                error_lines += 1
                print(f"[FAIL] Decoder POC checksum ERROR found at line {line_no}:")
                print(raw)

    print(f"[INFO] decoder log: {dec_log_path}")
    print(f"[INFO] POC lines found: {total_poc_lines}")
    print(f"[INFO] POC OK lines: {ok_lines}")
    print(f"[INFO] POC ERROR lines: {error_lines}")
    print(f"[INFO] POC lines without OK/ERROR: {len(unknown_status_lines)}")

    if total_poc_lines == 0:
        print("[FAIL] no POC lines found in decoder log")
        return False

    if ok_lines == 0:
        print("[FAIL] no POC checksum OK line found")
        return False

    if error_lines > 0:
        print("[FAIL] decoder contains POC checksum ERROR")
        return False

    print("[PASS] decoder MD5/checksum POC lines are OK")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify VTM decoder POC checksum lines."
    )
    parser.add_argument(
        "--dec-log",
        required=True,
        help="Path to decoder log file",
    )

    # 나중에 encoder/decoder param checksum 비교까지 확장할 수 있게 받아만 둠.
    parser.add_argument(
        "--enc-log",
        required=False,
        default=None,
        help="Optional encoder log file. Currently unused.",
    )

    args = parser.parse_args()

    dec_log_path = Path(args.dec_log)

    ok = check_poc_md5_lines(dec_log_path)

    if ok:
        print("FINAL RESULT: PASS")
        return 0
    else:
        print("FINAL RESULT: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
