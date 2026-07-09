#!/usr/bin/env python3
"""Test the extended calc device against the ornith-notation patterns we saw
burn budget in MATH v11-v13 traces. Every failing pattern gets a passing test.

    PYTHONPATH=~/Code/LLMOS python3 test_calc_ornith.py
"""
import os, sys, math
sys.path.insert(0, os.path.expanduser("~/Code/LLMOS"))
from llmos.store import Store
from llmos.syscall import SyscallTable
from llmos.pcb import PCB, Status


class FakePCB:
    def __init__(self):
        self.pid = 1
        self.capabilities = {"dev.calc", "dev.clock", "mem.read", "mem.write"}
        self.tainted = False


def call_calc(sys_tbl, expr):
    r = sys_tbl._calc(FakePCB(), {"expr": expr})
    return r


def check(expr, expected, sys_tbl, tol=1e-9):
    r = call_calc(sys_tbl, expr)
    if "error" in r:
        print(f"  FAIL {expr!r}: {r['error']}")
        return False
    v = r["value"]
    ok = abs(v - expected) < tol if isinstance(expected, float) else v == expected
    tag = "OK  " if ok else "FAIL"
    resolved = r.get("resolved", "")
    print(f"  {tag} {expr!r:<40} -> {v} (expected {expected}) {('[resolved:' + resolved + ']') if resolved else ''}")
    return ok


def main():
    store = Store("/tmp/test_calc_ornith.db")
    sys_tbl = SyscallTable(store, fs_policy={})
    cases = [
        # --- factorials (blocked v11/v13 for MANY steps) ---
        ("5!", 120),
        ("1! + 2!", 3),
        ("8!", 40320),
        ("10!", 3628800),
        ("(3+2)!", 120),
        ("5! / (3! * 2!)", 10),

        # --- binomials (blocked v13 pid=10 on 'C(6,3) * C(5,2)') ---
        ("C(6, 3)", 20),
        ("C(6, 3) * C(5, 2)", 200),
        ("binomial(10, 4)", 210),
        ("comb(10, 4)", 210),

        # --- permutations ---
        ("P(5, 2)", 20),
        ("permutations(6, 3)", 120),

        # --- gcd / lcm (blocked v13 pid=31 on 'lcm(1..7)') ---
        ("gcd(12, 18)", 6),
        ("lcm(4, 6)", 12),
        ("lcm(1, 2, 3, 4, 5, 6, 7)", 420),

        # --- mod (blocked v13 pid=25 on '(2*4 + 2 - 18) mod 19') ---
        ("200 mod 7", 4),
        ("(2*4 + 2 - 18) mod 19", 11),

        # --- trig + pi (blocked v11 pid=15 on 'arcsin(0.31)*180/pi') ---
        ("sin(pi/2)", 1.0),
        ("cos(0)", 1.0),
        ("arcsin(0.31) * 180 / pi", math.asin(0.31) * 180 / math.pi),
        ("sqrt(3)/4 * (1 + 4 + 1)", math.sqrt(3) / 4 * 6),

        # --- exp / log ---
        ("log(e)", 1.0),
        ("ln(e**2)", 2.0),
        ("exp(0)", 1.0),

        # --- ^ as exponent (ornith uses this) ---
        ("2^10", 1024),
        ("(17/3)^2", (17/3)**2),

        # --- unicode ---
        ("π", math.pi),

        # --- pre-existing quantity-word support still works ---
        ("half a dozen * 6000", 36000),
        ("(half a dozen * 6000 - 1200) / (twenty dozen)", 145),
    ]
    passed = sum(check(e, exp, sys_tbl) for e, exp in cases)
    print(f"\n{passed}/{len(cases)} passed")
    sys.exit(0 if passed == len(cases) else 1)


if __name__ == "__main__":
    main()
