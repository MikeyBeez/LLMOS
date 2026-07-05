"""Persistent store — delegated to SQLite (APFS-backed), per the implementation plan.

Holds four things: memory (the brain / disk), the trace ledger (single-writer,
append-only), process snapshots (checkpoint/resume), and a metrics table
(per-instruction timing/tokens, for measuring how slow the machine is and where
the time goes). We do not write a storage engine; SQLite gives us persistence,
transactions, and locking for free.
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
  topic       TEXT NOT NULL DEFAULT 'general',
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
CREATE TABLE IF NOT EXISTS metrics (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  pid            INTEGER NOT NULL,
  pc             INTEGER NOT NULL,
  op             TEXT NOT NULL,
  cpu_type       TEXT,
  model          TEXT,
  cpu_ms         REAL,      -- wall time of cpu.step() = one inference cycle (incl. retries)
  commit_ms      REAL,      -- wall time of the kernel commit (syscall + trace + snapshot)
  retries        INTEGER DEFAULT 0,
  prompt_tokens  INTEGER,   -- ollama prompt_eval_count (context tokens read)
  eval_tokens    INTEGER,   -- ollama eval_count (tokens generated)
  eval_ms        REAL,      -- ollama eval_duration (generation only)
  load_ms        REAL,      -- ollama load_duration (model cold-load; ~0 when warm)
  fault          INTEGER DEFAULT 0,
  ts             REAL NOT NULL
);
"""


class Store:
    def __init__(self, path: str):
        self.path = path
        self.db = sqlite3.connect(path)
        # Speed: WAL + NORMAL sync cut the per-instruction fsync cost of our
        # commit-after-every-write pattern without risking corruption (only a
        # crash mid-write loses the very last commit, which the trace tolerates).
        try:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError:
            pass
        self.db.executescript(SCHEMA)
        try:
            self.db.execute("ALTER TABLE memory ADD COLUMN topic TEXT NOT NULL DEFAULT 'general'")
        except sqlite3.OperationalError:
            pass   # column already exists
        self.db.commit()

    # --- memory: the brain / disk ---------------------------------------
    def mem_write(self, ns: str, key: str, value: Any, provenance: str = "trusted", topic: str = "general") -> None:
        self.db.execute(
            "INSERT INTO memory(ns,key,value,provenance,topic,updated_at) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(ns,key) DO UPDATE SET "
            "value=excluded.value, provenance=excluded.provenance, topic=excluded.topic, updated_at=excluded.updated_at",
            (ns, key, json.dumps(value), provenance, topic, time.time()),
        )
        self.db.commit()

    def mem_read(self, ns: str, key: str) -> Any:
        row = self.db.execute("SELECT value FROM memory WHERE ns=? AND key=?", (ns, key)).fetchone()
        return json.loads(row[0]) if row else None

    def mem_list(self, ns: str) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT key FROM memory WHERE ns=? ORDER BY key", (ns,))]

    def mem_by_topic(self, topic: str, ns: str = "mem") -> dict:
        rows = self.db.execute("SELECT key,value FROM memory WHERE ns=? AND topic=? ORDER BY key", (ns, topic)).fetchall()
        return {r[0]: json.loads(r[1]) for r in rows}

    def topics(self, ns: str = "mem") -> list[str]:
        return [r[0] for r in self.db.execute("SELECT DISTINCT topic FROM memory WHERE ns=? ORDER BY topic", (ns,))]

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

    # --- metrics: per-instruction timing / tokens ------------------------
    def metrics_append(self, pid: int, pc: int, op: str, cpu_type: str = None, model: str = None,
                        cpu_ms: float = None, commit_ms: float = None, retries: int = 0,
                        prompt_tokens: int = None, eval_tokens: int = None,
                        eval_ms: float = None, load_ms: float = None, fault: int = 0) -> None:
        self.db.execute(
            "INSERT INTO metrics(pid,pc,op,cpu_type,model,cpu_ms,commit_ms,retries,"
            "prompt_tokens,eval_tokens,eval_ms,load_ms,fault,ts) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, pc, op, cpu_type, model, cpu_ms, commit_ms, retries,
             prompt_tokens, eval_tokens, eval_ms, load_ms, fault, time.time()),
        )
        self.db.commit()

    def metrics_rows(self, cpu_type: str = None) -> list[dict]:
        q = ("SELECT pid,pc,op,cpu_type,model,cpu_ms,commit_ms,retries,"
             "prompt_tokens,eval_tokens,eval_ms,load_ms,fault,ts FROM metrics")
        params: tuple = ()
        if cpu_type:
            q += " WHERE cpu_type=?"
            params = (cpu_type,)
        q += " ORDER BY id"
        cols = ["pid", "pc", "op", "cpu_type", "model", "cpu_ms", "commit_ms", "retries",
                "prompt_tokens", "eval_tokens", "eval_ms", "load_ms", "fault", "ts"]
        return [dict(zip(cols, r)) for r in self.db.execute(q, params).fetchall()]

    def close(self) -> None:
        self.db.close()
