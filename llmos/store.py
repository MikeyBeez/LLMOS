"""Persistent store — delegated to SQLite (APFS-backed), per the implementation plan.

Holds three things: memory (the brain / disk), the trace ledger (single-writer,
append-only), and process snapshots (checkpoint/resume). We do not write a storage
engine; SQLite gives us persistence, transactions, and locking for free.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
  ns          TEXT NOT NULL,
  key         TEXT NOT NULL,
  value       TEXT NOT NULL,
  provenance  TEXT NOT NULL DEFAULT 'trusted',
  updated_at  REAL NOT NULL,
  PRIMARY KEY (ns, key)
);
CREATE TABLE IF NOT EXISTS trace (
  seq     INTEGER PRIMARY KEY AUTOINCREMENT,
  pid     INTEGER NOT NULL,
  pc      INTEGER NOT NULL,
  op      TEXT NOT NULL,
  args    TEXT NOT NULL,
  result  TEXT NOT NULL,
  ts      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS processes (
  pid         INTEGER PRIMARY KEY,
  snapshot    TEXT NOT NULL,
  updated_at  REAL NOT NULL
);
"""


class Store:
    def __init__(self, path: str):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self.db.commit()

    # --- memory: the brain / disk ---------------------------------------
    def mem_write(self, ns: str, key: str, value: Any, provenance: str = "trusted") -> None:
        self.db.execute(
            "INSERT INTO memory(ns,key,value,provenance,updated_at) VALUES(?,?,?,?,?) "
            "ON CONFLICT(ns,key) DO UPDATE SET "
            "value=excluded.value, provenance=excluded.provenance, updated_at=excluded.updated_at",
            (ns, key, json.dumps(value), provenance, time.time()),
        )
        self.db.commit()

    def mem_read(self, ns: str, key: str) -> Any:
        row = self.db.execute("SELECT value FROM memory WHERE ns=? AND key=?", (ns, key)).fetchone()
        return json.loads(row[0]) if row else None

    def mem_list(self, ns: str) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT key FROM memory WHERE ns=? ORDER BY key", (ns,))]

    # --- trace: the append-only ledger (single writer = the kernel) ------
    def trace_append(self, pid: int, pc: int, op: str, args: Any, result: Any) -> int:
        cur = self.db.execute(
            "INSERT INTO trace(pid,pc,op,args,result,ts) VALUES(?,?,?,?,?,?)",
            (pid, pc, op, json.dumps(args), json.dumps(result), time.time()),
        )
        self.db.commit()
        return cur.lastrowid

    def trace_read(self, pid: int) -> list[dict]:
        rows = self.db.execute(
            "SELECT seq,pid,pc,op,args,result,ts FROM trace WHERE pid=? ORDER BY seq", (pid,)
        ).fetchall()
        return [
            {"seq": r[0], "pid": r[1], "pc": r[2], "op": r[3],
             "args": json.loads(r[4]), "result": json.loads(r[5]), "ts": r[6]}
            for r in rows
        ]

    # --- process snapshots: checkpoint / resume --------------------------
    def save_process(self, pcb_dict: dict) -> None:
        self.db.execute(
            "INSERT INTO processes(pid,snapshot,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(pid) DO UPDATE SET snapshot=excluded.snapshot, updated_at=excluded.updated_at",
            (pcb_dict["pid"], json.dumps(pcb_dict), time.time()),
        )
        self.db.commit()

    def list_processes(self) -> list[dict]:
        return [json.loads(r[0]) for r in self.db.execute("SELECT snapshot FROM processes ORDER BY pid")]

    def close(self) -> None:
        self.db.close()
