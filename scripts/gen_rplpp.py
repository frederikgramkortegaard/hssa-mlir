#!/usr/bin/env python3
"""Random ROOPL++ program generator for hssa-mlir differential fuzzing.

    python3 scripts/gen_rplpp.py [SEED]

Prints one random, well-formed, REVERSIBLE ROOPL++ program (`.rplpp`) to stdout.
SEED (optional) makes a program reproducible.

Design goal: emit a NARROW, ALWAYS-VALID stream. Every program this prints must
(a) compile with the reference compiler and (b) run reversibly, so that any
failure downstream is OUR bug, not a generator bug. We therefore track the
*exact* integer value of every int field/local as we build, which lets us:
  - never put the modified lvalue on its own RHS  (x += x is irreversible),
  - emit correct loop exit-witnesses and delocal assertions,
  - emit correct conditional exit-witnesses (e2) by constraining branches so the
    test expression's value is invariant across the branch, so e2 == e1.

Covered ROOPL++ subset (reversible by construction):
  - class Main with int fields (the program output = Main's fields)
  - optional helper classes with int fields + reversible methods, invoked via
    `call helper::method(args)` where args are LOCAL variables (never fields --
    passing a field to a local/foreign method violates reversibility)
  - statements: += -= ^= on fields/locals, swap <=>, nested arithmetic exprs,
    `local int x = e ... delocal int x = e2`, `from e1 do s loop s until e2`,
    `if e1 then s else s fi e2`
No heap/new/delete/arrays/uncall/inheritance yet (staged for later widening).
"""

import random
import sys

# 32-bit wrap to mirror the interpreter/JVM (and our i32 wrap semantics).
MASK = 0xFFFFFFFF


def wrap32(v):
    v &= MASK
    if v >= 0x80000000:
        v -= 0x100000000
    return v


class Var:
    """A tracked integer storage location with a known current value."""

    __slots__ = ("name", "value", "is_field")

    def __init__(self, name, value, is_field):
        self.name = name
        self.value = wrap32(value)
        self.is_field = is_field


class Gen:
    def __init__(self, rng):
        self.rng = rng
        self.lines = []
        self.indent = 0
        self.counter = 0  # fresh-name source

    # ---- emission helpers -------------------------------------------------
    def emit(self, s):
        self.lines.append("  " * self.indent + s)

    def fresh(self, prefix):
        self.counter += 1
        return f"{prefix}{self.counter}"

    # ---- expression building ----------------------------------------------
    def gen_expr(self, env, depth, exclude=None):
        """Build a (text, value) expression over the live int vars in `env`.

        `exclude` is a var NAME that must not appear (used so an assignment RHS
        never references its own lvalue -> keeps the statement reversible).
        `env` is a list of Var. Operators are restricted to the arithmetic set
        that is total and non-trapping: + - *  and the comparison/logic ops
        (only used at depth 0 callers that want a boolean-ish value). Here we
        emit only +,-,* and literals/vars to keep values well-defined; division
        and modulo are avoided to prevent divide-by-zero.
        """
        usable = [v for v in env if v.name != exclude]
        # Leaf: literal or variable.
        if depth <= 0 or not usable or self.rng.random() < 0.4:
            if usable and self.rng.random() < 0.55:
                v = self.rng.choice(usable)
                return v.name, v.value
            lit = self.rng.randint(-8, 12)
            return (f"({lit})" if lit < 0 else str(lit)), lit
        op = self.rng.choice(["+", "-", "*"])
        lt, lv = self.gen_expr(env, depth - 1, exclude)
        rt, rv = self.gen_expr(env, depth - 1, exclude)
        if op == "+":
            val = lv + rv
        elif op == "-":
            val = lv - rv
        else:
            val = lv * rv
        return f"({lt} {op} {rt})", wrap32(val)

    # ---- simple reversible statements -------------------------------------
    def gen_update(self, env):
        """A single reversible update on a live var: y += e / -= / ^= / swap."""
        targets = env
        if not targets:
            return
        tgt = self.rng.choice(targets)
        kind = self.rng.random()
        # swap only if we have two distinct vars
        others = [v for v in env if v.name != tgt.name]
        if kind < 0.18 and others:
            o = self.rng.choice(others)
            self.emit(f"{tgt.name} <=> {o.name}")
            tgt.value, o.value = o.value, tgt.value
            return
        op = self.rng.choice(["+=", "-=", "^="])
        # RHS must not mention tgt (reversibility). depth 0..2.
        etext, eval_ = self.gen_expr(env, self.rng.randint(0, 2), exclude=tgt.name)
        self.emit(f"{tgt.name} {op} {etext}")
        if op == "+=":
            tgt.value = wrap32(tgt.value + eval_)
        elif op == "-=":
            tgt.value = wrap32(tgt.value - eval_)
        else:
            tgt.value = wrap32(tgt.value ^ eval_)

    def gen_local_block(self, env, depth):
        """local int t = e  <body>  delocal int t = e2  (reversible assertion).

        We emit the init expr e over the CURRENT env, track t's value through
        the body (t participates in the body as a normal var), then emit e2 as
        the literal final value of t -- a correct assertion by construction.
        """
        name = self.fresh("t")
        einit, val = self.gen_expr(env, self.rng.randint(0, 1))
        self.emit(f"local int {name} = {einit}")
        self.indent += 1
        t = Var(name, val, is_field=False)
        env2 = env + [t]
        self.gen_seq(env2, depth - 1, nstmts=self.rng.randint(1, 3))
        self.indent -= 1
        # delocal assertion = t's exact final value.
        fv = t.value
        self.emit(f"delocal int {name} = {fv if fv >= 0 else f'({fv})'}")

    def gen_loop(self, env, depth):
        """from x = 0 do <s1> loop <s2> until x = N  with a counter local.

        A dedicated loop counter `k` starts at 0; the loop body does NOT touch
        k except the generator-controlled increment, so the exit witness is
        exactly k = N and the entry test is k = 0. Other env vars may be
        updated in the body (their running value is tracked by replaying the
        loop N times).
        """
        name = self.fresh("k")
        n = self.rng.randint(1, 4)
        self.emit(f"local int {name} = 0")
        self.indent += 1
        k = Var(name, 0, is_field=False)
        # `body_env` are the vars the body may touch; the loop counter `k` is
        # NEVER among them, so the entry test `k = 0` and exit witness `k = n`
        # stay exactly correct and the loop is reversible.
        body_env = list(env)
        self.emit(f"from {name} = 0 do")
        self.indent += 1
        self.emit("skip")
        self.indent -= 1
        self.emit("loop")
        self.indent += 1
        # Capture body statements by replaying N iterations to track values.
        # We generate the body text ONCE but must apply its effect N times. To
        # keep value tracking exact and simple, we generate a straight-line body
        # of pure updates (no nested blocks) and replay it.
        body_src_start = len(self.lines)
        # Build body on a scratch so we can both emit and replay deterministically.
        body_updates = self.rng.randint(1, 3)
        # First iteration: emit real lines + apply (counter excluded from body).
        for _ in range(body_updates):
            self.gen_update(body_env)
        self.emit(f"{name} += 1")
        k.value = wrap32(k.value + 1)
        body_lines = self.lines[body_src_start:]
        self.indent -= 1
        # Replay the remaining N-1 iterations against the value model only
        # (re-parse our own emitted update lines). Counter line is ignored by
        # _replay for value purposes except its own increment; include k so its
        # `+= 1` replays, but it was already excluded from body updates.
        env2 = body_env + [k]
        for _ in range(n - 1):
            self._replay(body_lines, env2)
        self.emit(f"until {name} = {n}")
        self.indent -= 1
        # After loop k == n; it is delocal'd implicitly? No -- local needs delocal.
        self.emit(f"delocal int {name} = {n}")

    def _replay(self, body_lines, env):
        """Replay previously-emitted straight-line update lines against the
        value model (no new emission). Supports `y += e`, `y -= e`, `y ^= e`,
        and `y <=> z`. Used to keep loop-carried values exact without
        re-implementing an expression evaluator: we reparse the RHS with a tiny
        safe evaluator over the current env."""
        by_name = {v.name: v for v in env}
        for raw in body_lines:
            s = raw.strip()
            if not s or s == "skip":
                continue
            if "<=>" in s:
                a, b = (x.strip() for x in s.split("<=>"))
                va, vb = by_name.get(a), by_name.get(b)
                if va and vb:
                    va.value, vb.value = vb.value, va.value
                continue
            for op, fn in (("+=", lambda a, b: a + b),
                           ("-=", lambda a, b: a - b),
                           ("^=", lambda a, b: a ^ b)):
                if f" {op} " in s:
                    lhs, rhs = s.split(f" {op} ", 1)
                    tgt = by_name.get(lhs.strip())
                    if tgt is None:
                        break
                    rv = self._eval(rhs.strip(), by_name)
                    tgt.value = wrap32(fn(tgt.value, rv))
                    break

    def _eval(self, text, by_name):
        """Safely evaluate a generated arithmetic expression string over the
        current var values. Only +,-,*,parens,ints,names (all generator-emitted)
        appear, so a restricted eval with names substituted is exact."""
        # Substitute variable names with their current values.
        # Tokenize on identifiers; replace each known name.
        import re
        def repl(m):
            nm = m.group(0)
            if nm in by_name:
                return f"({by_name[nm].value})"
            return nm
        expr = re.sub(r"[A-Za-z_]\w*", repl, text)
        try:
            return wrap32(eval(expr, {"__builtins__": {}}, {}))
        except Exception:
            return 0

    def gen_conditional(self, env, depth):
        """if e1 then s1 else s2 fi e2 with e2 == e1, CONSTRAINED so the test
        value is invariant across whichever branch runs.

        Strategy: pick a test on a dedicated boolean-source var that NEITHER
        branch modifies. We introduce the test over an existing field/local `w`
        and ensure branches only touch OTHER vars. Then e2 == e1 holds.
        """
        if len(env) < 2:
            self.gen_update(env)
            return
        w = self.rng.choice(env)
        rest = [v for v in env if v.name != w.name]
        if not rest:
            self.gen_update(env)
            return
        cmp = self.rng.choice(["<", ">", "=", "!=", "<=", ">="])
        lit = self.rng.randint(-5, 10)
        test = f"{w.name} {cmp} {lit}"
        # Evaluate which branch runs, for value tracking.
        cur = w.value
        truth = {"<": cur < lit, ">": cur > lit, "=": cur == lit,
                 "!=": cur != lit, "<=": cur <= lit, ">=": cur >= lit}[cmp]
        self.emit(f"if {test} then")
        self.indent += 1
        # then-branch: updates restricted to `rest` (never touch w).
        then_start = len(self.lines)
        nthen = self.rng.randint(1, 2)
        for _ in range(nthen):
            self.gen_update(rest)
        if len(self.lines) == then_start:
            self.emit("skip")
        then_lines = self.lines[then_start:]
        self.indent -= 1
        self.emit("else")
        self.indent += 1
        else_start = len(self.lines)
        nelse = self.rng.randint(1, 2)
        for _ in range(nelse):
            self.gen_update(rest)
        if len(self.lines) == else_start:
            self.emit("skip")
        else_lines = self.lines[else_start:]
        self.indent -= 1
        self.emit(f"fi {test}")
        # Apply the taken branch to the value model (the other was emitted but
        # its value effect must be rolled back -- gen_update already mutated the
        # live Var objects for BOTH branches, so recompute cleanly by replaying
        # only the taken branch from a snapshot).
        # Simpler: snapshot BEFORE, then we already mutated twice. To stay exact,
        # we reset rest to the pre-branch snapshot and replay only taken branch.
        # (Snapshot captured below via closure is cleaner -- restructure:)
        # NOTE: handled by snapshot/restore in caller wrapper gen_conditional_safe.

    # ---- sequence ---------------------------------------------------------
    def gen_seq(self, env, depth, nstmts):
        for _ in range(nstmts):
            r = self.rng.random()
            if depth > 0 and r < 0.14:
                self.gen_loop(env, depth)
            elif depth > 0 and r < 0.26:
                self.gen_local_block(env, depth)
            elif depth > 0 and r < 0.40:
                self._conditional_safe(env, depth)
            else:
                self.gen_update(env)

    def _conditional_safe(self, env, depth):
        """Wrapper that keeps the value model exact across the conditional by
        snapshotting the vars the branches may touch, letting gen_conditional
        emit+mutate, then restoring and replaying only the taken branch."""
        if len(env) < 2:
            self.gen_update(env)
            return
        # Snapshot all env values.
        snap = {v.name: v.value for v in env}
        before_lines = len(self.lines)
        self.gen_conditional(env, depth)
        # Figure out the test from the emitted `fi <test>` line.
        fi_line = None
        for raw in self.lines[before_lines:]:
            s = raw.strip()
            if s.startswith("fi "):
                fi_line = s[3:].strip()
        if fi_line is None:
            return  # gen_conditional degenerated to a plain update; values ok.
        # Restore pre-branch values, then replay the TAKEN branch only.
        by_name = {v.name: v for v in env}
        for v in env:
            v.value = snap[v.name]
        # Parse test: "<name> <cmp> <lit>"
        parts = fi_line.split()
        wname, cmp, lit = parts[0], parts[1], int(parts[2])
        cur = by_name[wname].value
        truth = {"<": cur < lit, ">": cur > lit, "=": cur == lit,
                 "!=": cur != lit, "<=": cur <= lit, ">=": cur >= lit}[cmp]
        # Extract then/else line ranges from the emitted block.
        block = self.lines[before_lines:]
        then_lines, else_lines, mode = [], [], None
        for raw in block:
            s = raw.strip()
            if s.startswith("if "):
                mode = "then"; continue
            if s == "else":
                mode = "else"; continue
            if s.startswith("fi "):
                mode = None; continue
            if mode == "then":
                then_lines.append(raw)
            elif mode == "else":
                else_lines.append(raw)
        self._replay(then_lines if truth else else_lines, env)

    # ---- helper classes & calls -------------------------------------------
    def gen_program(self):
        rng = self.rng
        use_helper = rng.random() < 0.45

        helper_name = "Helper"
        helper_fields = []
        helper_methods = []  # list of (mname, [param names], body_fn)
        if use_helper:
            nf = rng.randint(1, 3)
            helper_fields = [f"h{i}" for i in range(nf)]
            # One reversible method that updates its int params (params are
            # locals inside the method, so updating them is fine and reversible
            # as long as RHS excludes the lvalue). The CALLER passes LOCAL vars.
            nparams = rng.randint(1, 2)
            params = [f"p{i}" for i in range(nparams)]
            helper_methods.append(("bump", params))

        # ---- Main class ----
        nmain = rng.randint(1, 4)
        main_fields = [f"f{i}" for i in range(nmain)]

        self.emit("class Main")
        self.indent += 1
        for f in main_fields:
            self.emit(f"int {f}")
        if use_helper:
            self.emit(f"{helper_name} obj")
        self.emit("")
        self.emit("method main()")
        self.indent += 1

        env = [Var(f, 0, is_field=True) for f in main_fields]

        # Some straight-line + structured body.
        self.gen_seq(env, depth=2, nstmts=rng.randint(2, 5))

        # Optional helper usage: new obj, call obj::bump(locals), delete obj.
        if use_helper:
            # locals to pass (must be LOCAL vars, not fields).
            params = helper_methods[0][1]
            self.emit(f"new {helper_name} obj")
            locs = []
            for i, _p in enumerate(params):
                lname = self.fresh("a")
                einit, val = self.gen_expr(env, rng.randint(0, 1))
                self.emit(f"local int {lname} = {einit}")
                locs.append(Var(lname, val, is_field=False))
            self.indent += 0
            # call mutates params: model the method effect to roll locals forward.
            self.emit(f"call obj::bump({', '.join(v.name for v in locs)})")
            # method 'bump' adds a fixed constant to each param (see emission).
            bump_consts = [rng.randint(1, 7) for _ in params]
            # stash for method body emission
            self._bump_consts = bump_consts
            for lv, c in zip(locs, bump_consts):
                lv.value = wrap32(lv.value + c)
            # delocal in reverse order; assertion = exact final value.
            for lv in reversed(locs):
                fv = lv.value
                self.emit(f"delocal int {lv.name} = {fv if fv >= 0 else f'({fv})'}")
            # object state must be cleared before delete. 'bump' only touched
            # locals, so obj fields are all still 0 (freshly new'd). Safe.
            self.emit(f"delete {helper_name} obj")

        self.indent -= 1  # end method main
        self.indent -= 1  # end class Main

        # ---- Helper class definition ----
        if use_helper:
            self.emit("")
            self.emit(f"class {helper_name}")
            self.indent += 1
            for f in helper_fields:
                self.emit(f"int {f}")
            self.emit("")
            mname, params = helper_methods[0]
            self.emit(f"method {mname}({', '.join(f'int {p}' for p in params)})")
            self.indent += 1
            # Each param gets a fixed += constant (reversible; RHS has no lvalue).
            for p, c in zip(params, self._bump_consts):
                self.emit(f"{p} += {c}")
            self.indent -= 1
            self.indent -= 1

    def render(self):
        return "\n".join(self.lines) + "\n"


def main():
    seed = None
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    g = Gen(rng)
    g.gen_program()
    out = g.render()
    sys.stdout.write(out)


if __name__ == "__main__":
    main()
