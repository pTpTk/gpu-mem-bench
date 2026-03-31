#!/usr/bin/env python3
import sys
import re

def parse_line(line):
    """
    Extract start offset (ns) and clock diff (us) from a line.
    Returns (offset_ns, diff_us) as integers/floats.
    """
    m = re.search(r"start offset:\s*(\d+),\s*clock diff\s*([\d.]+)us", line)
    if not m:
        return None
    return int(m.group(1)), float(m.group(2))

def main():
    entries = []

    i = 0
    # Read piped input
    for line in sys.stdin:
        parsed = parse_line(line)
        if parsed:
            entries.append(parsed)
        if i < 2:
            print(line)
            i += 1

    if not entries:
        print("No valid input detected.")
        return

    # Step 2: find lowest start offset
    min_offset = min(offset for offset, _ in entries)

    # Step 3: compute total time = adjusted_offset(ns) + clock_diff(us)
    # Convert clock diff from microseconds to nanoseconds: 1 us = 1000 ns
    totals = []
    for offset, diff_us in entries:
        adjusted = offset - min_offset
        total_ns = adjusted + diff_us * 1000.0
        totals.append(total_ns)

    # Step 4: report largest total time
    max_total = max(totals)

    print(f"Largest total time: {max_total:.2f} ns")

if __name__ == "__main__":
    main()