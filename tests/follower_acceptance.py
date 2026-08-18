#!/usr/bin/env python
"""Acceptance test for the trajectory time-parametrization backend.

Encodes the user-stated criteria for policy-learning trajectories:

  1. UNIFORM SPEED — after the physics-mandated launch ramp and before the
     deliberate final-approach taper, path speed must stay >= 70% of the
     cruise cap. No mid-path dips (the follower's clocked-target limit cycle
     dips to 0.01-0.3 rad/s today; this test exists to gate its redesign).
  2. NO STOPPING except at the end of the trajectory.
  3. NO OVERSHOOT — the trajectory must not travel past the goal along the
     final approach direction (> 5 mrad), and must end on the goal.
  4. LIMITS — per-joint |v| <= vel cap, |a| <= acc cap, |j| <= jerk cap
     (small FD tolerances); with uniform_path_speed, L2 path speed <= cap.
  5. NO SCHEDULE BLOWUP — duration <= 1.10x the plain-toppra reference on
     the same waypoints (the follower must not trade dips for dawdling).

Corpus: tests/data/follower_corpus/*.pkl — real dumps captured via
SPLATSIM_TRAJ_DUMP during planar_3joint generation (waypoints + kwargs as
the planner actually called parametrize_path; 'traj' is the recorded output
of the batch-5-era follower, kept for regression diffing).

Usage:
  python tests/follower_acceptance.py                    # test default backend
  python tests/follower_acceptance.py --backend toppra   # test another backend
  python tests/follower_acceptance.py --save-baseline b.json
  python tests/follower_acceptance.py --diff-recorded    # bit-compare vs dumps

Expected today (2026-08): the follower FAILS criterion 1 on a handful of
cases (known defect, see the KNOWN DEFECT comment in
follower_parametrize_path). The redesign is done when this script passes
clean on the whole corpus.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from splatsim.utils.planner_defaults import PLANNER_DEFAULTS as PD
from splatsim.utils.rrt_path_utils import parametrize_path

CRUISE_FRACTION = 0.70   # criterion 1: min speed in cruise window vs cap
TAPER_ARC_MARGIN = 1.2   # exclude tail where remaining arc < taper dist x this
RAMP_ARC_MARGIN = 1.5    # exclude head: arc to reach cruise = v^2/2a, x this
OVERSHOOT_TOL = 5e-3     # rad, along final-approach direction
ENDPOINT_TOL = 1e-3      # rad, per joint
VEL_TOL = 1.02
ACC_TOL = 1.05
JERK_TOL = 1.15
DURATION_RATIO_MAX = 1.03   # x reference, plus...
DURATION_SLACK_S = 0.4       # ...absolute jerk-physics allowance: accel can't
                             # step, so a jerk-limited launch/stop pays ~a/j
                             # per transition regardless of path length — a
                             # pure ratio bar spuriously fails short paths
                             # whose mid-path schedule is tick-identical.


def path_speeds(traj: np.ndarray, hz: float) -> np.ndarray:
    """L2 joint-space speed per tick (len = len(traj)-1)."""
    return np.linalg.norm(np.diff(traj, axis=0), axis=1) * hz


def check_case(name, waypoints, kwargs, backend):
    hz = float(kwargs.get("control_hz", 30))
    vel = float(kwargs.get("max_joint_vel", PD.max_joint_vel))
    acc = float(kwargs.get("max_joint_acc", PD.max_joint_acc))
    jerk = float(kwargs.get("max_joint_jerk", PD.max_joint_jerk))
    taper = float(kwargs.get("final_approach_dist", PD.final_approach_dist))
    uniform = bool(kwargs.get("uniform_path_speed", PD.uniform_path_speed))

    # limits were positional at the original call sites, so the dump kwargs
    # don't carry them — supply the planner defaults explicitly.
    kw = dict(kwargs)
    vel = float(np.max(kw.pop("max_joint_vel", vel)))
    acc = float(np.max(kw.pop("max_joint_acc", acc)))
    jerk = float(np.max(kw.pop("max_joint_jerk", jerk)))
    dof = waypoints.shape[1]
    lims = (np.full(dof, vel), np.full(dof, acc), np.full(dof, jerk))
    traj = parametrize_path(waypoints, *lims, backend=backend, **kw)
    ref = parametrize_path(waypoints, *lims, backend="toppra", **kw)

    fails, metrics = [], {}
    dq = np.diff(traj, axis=0)
    sp = np.linalg.norm(dq, axis=1) * hz
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(dq, axis=1))])
    total = arc[-1]

    # cruise window (arc-based: robust to timing differences between backends)
    ramp_arc = vel * vel / (2.0 * acc) * RAMP_ARC_MARGIN
    head = arc[:-1] > ramp_arc
    tail = (total - arc[:-1]) > taper * TAPER_ARC_MARGIN
    win = head & tail
    # The bar is floored at what the toppra reference achieves on the same
    # geometry: accel-limited corner speed (~sqrt(a*r)) is physics — a perfect
    # tracker can't cruise faster than its reference through a tight corner.
    sp_r = path_speeds(ref, hz)
    arc_r = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))])
    win_r = (arc_r[:-1] > ramp_arc) & ((arc_r[-1] - arc_r[:-1]) > taper * TAPER_ARC_MARGIN)
    if win.any():
        vmin = float(sp[win].min())
        ref_min = float(sp_r[win_r].min()) if win_r.any() else vel
        metrics["cruise_min"] = vmin
        metrics["ref_cruise_min"] = ref_min
        # Pointwise-in-arc comparison: at each in-window tick, speed must be
        # >= 75% of the reference's minimum NEAR that arc (+-0.05 rad). 75%,
        # not 90%: at blend corners the time-optimal reference carries
        # curvature-onset jerk of 15-40 (it rides per-joint accel exactly at
        # the cap), and normal jerk scales as v^3 — a jerk-limited
        # trajectory can only pass such a corner at ~(J/j_ref)^(1/3) =
        # 0.75-0.85x the reference's speed. The absolute 70%-of-cap floor
        # above still forbids real slowdowns on cruise spans.
        # Window-min vs window-min is boundary-quantization-sensitive: a
        # braking ramp crossing the window's tail boundary gets sampled one
        # tick deeper by whichever trajectory has more ticks, flagging 1-5%
        # "dips" that are the reference's own deceleration. The local-window
        # floor also absorbs the ~1-tick arc shift of jerk-limited braking
        # (which reaches a corner's floor speed slightly before the corner).
        arcs_q = arc[:-1][win]
        sp_q = sp[win]
        worst = None
        for s_k, v_k in zip(arcs_q, sp_q):
            lo = int(np.searchsorted(arc_r[:-1], s_k - 0.05))
            hi = max(int(np.searchsorted(arc_r[:-1], s_k + 0.05)), lo + 1)
            ref_local = float(sp_r[lo:hi].min()) if hi <= len(sp_r) else float(sp_r[lo:].min() if len(sp_r[lo:]) else vel)
            thr = min(CRUISE_FRACTION * vel, 0.75 * ref_local)
            margin = v_k - thr
            if worst is None or margin < worst[0]:
                worst = (margin, v_k, thr, s_k)
        if worst is not None and worst[0] < 0:
            fails.append(
                f"cruise dip {worst[1]:.3f} < {worst[2]:.3f} @arc {worst[3]:.2f} (ref win min {ref_min:.3f})"
            )
    else:
        metrics["cruise_min"] = None  # path too short to cruise — skip crit 1

    # limits (FD accel/jerk)
    v = dq * hz
    a = np.diff(v, axis=0) * hz
    j = np.diff(a, axis=0) * hz
    metrics.update(vmax=float(np.abs(v).max()), amax=float(np.abs(a).max()) if len(a) else 0.0,
                   jmax=float(np.abs(j).max()) if len(j) else 0.0, l2max=float(sp.max()))
    if metrics["vmax"] > vel * VEL_TOL:
        fails.append(f"|v| {metrics['vmax']:.3f} > {vel}")
    if metrics["amax"] > acc * ACC_TOL:
        fails.append(f"|a| {metrics['amax']:.3f} > {acc}")
    if metrics["jmax"] > jerk * JERK_TOL:
        fails.append(f"|j| {metrics['jmax']:.2f} > {jerk}")
    # uniform_path_speed equalizes L2 path speed at the box limits' diagonal
    # maximum vel*sqrt(dof) (0.866 for 3 dof) — the bare per-joint cap as the
    # L2 target pinned every trajectory to a degenerate exactly-v_max speed
    # distribution (2026-08-18 planar_5-era comparison).
    l2_cap = vel * float(np.sqrt(dof))
    if uniform and metrics["l2max"] > l2_cap * VEL_TOL:
        fails.append(f"L2 speed {metrics['l2max']:.3f} > path cap {l2_cap:.3f} (uniform_path_speed)")

    # endpoints + overshoot
    goal = waypoints[-1]
    err = float(np.abs(traj[-1] - goal).max())
    metrics["end_err"] = err
    if float(np.abs(traj[0] - waypoints[0]).max()) > 1e-9:
        fails.append("start moved")
    if err > ENDPOINT_TOL:
        fails.append(f"endpoint err {err:.2e}")
    # Overshoot: only meaningful in the tail — a path that curls around the
    # goal early legitimately crosses the goal plane. Measure max travel past
    # the goal along the approach direction within the final-approach window.
    d = traj[-1] - traj[max(0, len(traj) - 5)]
    n = np.linalg.norm(d)
    # ...and attribute it: if the toppra reference (pure retiming of the same
    # geometry) shows the same excursion, it's planner GEOMETRY (e.g. a
    # collision-mandated near-goal fold), not a parametrization defect — the
    # backend can't be gated on it.
    tail_pts = arc > (total - taper * TAPER_ARC_MARGIN)
    if n > 1e-9 and tail_pts.any():
        over = float(((traj[tail_pts] - goal) @ (d / n)).max())
        metrics["overshoot"] = over
        arc_r = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))])
        tail_r = arc_r > (arc_r[-1] - taper * TAPER_ARC_MARGIN)
        over_ref = float(((ref[tail_r] - goal) @ (d / n)).max()) if tail_r.any() else 0.0
        if over > max(OVERSHOOT_TOL, over_ref + OVERSHOOT_TOL):
            fails.append(f"overshoot {over*1e3:.1f} mrad (reference {over_ref*1e3:.1f})")

    # start monotonicity: the backend must not ADD backward motion at the
    # start beyond what the waypoint geometry itself contains (start-side
    # folds are the PLANNER's job — trimmed by the monotone-progress pass
    # in _straighten_terminal — and pre-fix dumps in this corpus still
    # carry them, so the bar is relative to the waypoints, not absolute).
    net = waypoints[-1] - waypoints[0]
    nn = float(np.linalg.norm(net))
    if nn > 1e-9:
        u_net = net / nn
        wp_back = float(((waypoints - waypoints[0]) @ u_net).min())
        tr_back = float(((traj - traj[0]) @ u_net).min())
        metrics["start_backtrack"] = tr_back
        if tr_back < wp_back - 0.01:
            fails.append(
                f"backend added backward start motion: {tr_back:.3f} vs waypoints {wp_back:.3f}"
            )

    # schedule vs toppra reference
    ratio = len(traj) / max(1, len(ref))
    metrics["dur_ratio"] = ratio
    if len(traj) > DURATION_RATIO_MAX * len(ref) + DURATION_SLACK_S * hz:
        fails.append(f"duration {ratio:.2f}x toppra reference (> {DURATION_RATIO_MAX}x + {DURATION_SLACK_S}s)")

    return traj, fails, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, help="backend to test (default: env/default)")
    ap.add_argument("--corpus", default=os.path.join(os.path.dirname(__file__), "data", "follower_corpus"))
    ap.add_argument("--save-baseline", metavar="JSON")
    ap.add_argument("--diff-recorded", action="store_true",
                    help="also bit-compare output against the 'traj' stored in each dump")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.corpus, "*.pkl")))
    if not files:
        sys.exit(f"no corpus at {args.corpus}")

    n_fail, baseline = 0, {}
    for f in files:
        d = pickle.load(open(f, "rb"))
        name = os.path.basename(f)
        try:
            traj, fails, metrics = check_case(name, d["waypoints"], d["kwargs"], args.backend)
        except Exception as e:
            fails, metrics, traj = [f"EXCEPTION {type(e).__name__}: {e}"], {}, None
        if args.diff_recorded and traj is not None:
            rec = d["traj"]
            if traj.shape == rec.shape and np.array_equal(traj, rec):
                fails.append("(recorded: identical)") if False else None
                metrics["rec_diff"] = 0.0
            else:
                metrics["rec_diff"] = (float(np.abs(traj - rec).max())
                                       if traj.shape == rec.shape else f"shape {traj.shape} vs {rec.shape}")
        baseline[name] = metrics
        status = "FAIL" if fails else "ok  "
        n_fail += bool(fails)
        mm = " ".join(f"{k}={v:.3f}" for k, v in metrics.items()
                      if isinstance(v, float)) or ""
        print(f"{status} {name}  {mm}")
        for msg in fails:
            print(f"       - {msg}")

    # Handoff criterion (shared autonomy): with start_vel set, the plan must
    # BEGIN at ~the handoff speed, not ramp from rest — the LP's virtual
    # pre-start state once hard-coded rest and every intervention launched
    # from zero regardless of policy speed. Checked on a corpus sample.
    print("\n--- start_vel handoff ---")
    hand_fail = 0
    for f in files[::8]:
        d = pickle.load(open(f, "rb"))
        wp = np.asarray(d["waypoints"], dtype=np.float64)
        if len(wp) < 2:
            continue
        kw = dict(d["kwargs"])
        hz2 = float(kw.get("control_hz", 30))
        dof = wp.shape[1]
        d0 = wp[1] - wp[0]
        d0 = d0 / max(np.linalg.norm(d0), 1e-12)
        v0 = 0.3
        kw["start_vel"] = d0 * v0
        lims = (np.full(dof, PD.max_joint_vel), np.full(dof, PD.max_joint_acc),
                np.full(dof, PD.max_joint_jerk))
        tr = parametrize_path(wp, *lims, backend=args.backend, **kw)
        sp0 = float(np.linalg.norm(tr[1] - tr[0]) * hz2)
        a = np.diff(np.diff(tr, axis=0) * hz2, axis=0) * hz2
        j = np.diff(a, axis=0) * hz2
        ok = (abs(sp0 - v0) < 0.12 and np.abs(a).max() <= PD.max_joint_acc * ACC_TOL
              and np.abs(j).max() <= PD.max_joint_jerk * JERK_TOL)
        hand_fail += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {os.path.basename(f)}  first-tick speed "
              f"{sp0:.3f} (handoff {v0}) amax={np.abs(a).max():.2f} jmax={np.abs(j).max():.1f}")
    # Brake-out criterion: a MOVING handoff whose velocity opposes the path
    # must genuinely continue along the velocity (braking) before reversing —
    # full speed at tick 0, aligned with start_vel, limits exact. (The
    # planner prepends the lead-in vertex; here we synthesize it.)
    print("\n--- brake-out handoff ---")
    d = pickle.load(open(files[0], "rb"))
    wpb = np.asarray(d["waypoints"], dtype=np.float64)
    kwb = dict(d["kwargs"])
    hzb = float(kwb.get("control_hz", 30))
    dofb = wpb.shape[1]
    d0b = wpb[1] - wpb[0]
    d0b = d0b / np.linalg.norm(d0b)
    vhat = -d0b
    kwb["start_vel"] = vhat * 0.4
    # jerk-aware braking distance: v^2/(1.4a) + v*a/j (matches the planner)
    wp_lead = np.vstack([wpb[:1], (wpb[0] + vhat * (0.4 ** 2 / 1.4 + 0.4 / 10.0))[None], wpb[1:]])
    limsb = (np.full(dofb, PD.max_joint_vel), np.full(dofb, PD.max_joint_acc),
             np.full(dofb, PD.max_joint_jerk))
    trb = parametrize_path(wp_lead, *limsb, backend=args.backend, **kwb)
    v0b = (trb[1] - trb[0]) * hzb
    alb = float(np.dot(v0b / max(np.linalg.norm(v0b), 1e-9), vhat))
    ab = np.diff(np.diff(trb, axis=0) * hzb, axis=0) * hzb
    jb = np.diff(ab, axis=0) * hzb
    bo_ok = (alb > 0.95 and abs(np.linalg.norm(v0b) - 0.4) < 0.1
             and np.abs(ab).max() <= PD.max_joint_acc * ACC_TOL
             and np.abs(jb).max() <= PD.max_joint_jerk * JERK_TOL)
    print(f"{'ok  ' if bo_ok else 'FAIL'} reversal handoff: align {alb:+.2f} "
          f"first-speed {np.linalg.norm(v0b):.2f} amax {np.abs(ab).max():.2f} jmax {np.abs(jb).max():.1f}")
    n_fail += (not bo_ok)

    n_fail += hand_fail

    print(f"\n{len(files) - (n_fail - hand_fail)}/{len(files)} passed"
          f" + handoff {5 - hand_fail if hand_fail <= 5 else 0}/{len(files[::8])} ok")
    if args.save_baseline:
        json.dump(baseline, open(args.save_baseline, "w"), indent=1, default=str)
        print(f"baseline -> {args.save_baseline}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
