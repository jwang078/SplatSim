#!/usr/bin/env python
"""Acceptance suite for MOVING-HANDOFF behavior (shared-autonomy interventions).

Complements tests/follower_acceptance.py (which gates the retimed backend's
from-rest behavior on the recorded corpus). This file gates the handoff
stack that kept regressing during development:

  A. Parametrizer carry: for every handoff geometry class — aligned,
     mid-band misaligned (lead-in, no cusp), reversal (brake-out split),
     short-runway (escalated brake) — the trajectory's first tick must move
     at ~the handoff speed, in the velocity direction, within limits
     (escalation classes exempt from nominal caps by design).
     EXACT-CARRY (no exceptions): handoffs ABOVE the planner's own speed
     envelope (vmax, start license) must still launch at exactly the
     handed speed — the brake-in overlay sheds the surplus along the path,
     and the profile must be back inside vmax shortly after launch. A
     clamped launch recorded a 4x speed discontinuity at the policy->RRT
     seam (eval planar_3joint scenario 2, 2026-08-17).
  B. Planner start treatment: the braking lead-in must be placed for any
     misalignment beyond ~18 deg (dot < 0.95 — an aligned-looking chord can
     still curve immediately and choke toppra's controllable window), must
     fall back to CONTACT-level clearance when the planning-clearance spur
     is blocked, and must SURVIVE the trajopt-collides fallback
     (straighten_only re-application) — losing it there measured a 0.43
     rad/s handoff launching at 0.05.
  C. Brake feasibility: directional collision judgment used by the shield's
     micro-rewind, at the emergency (1 cm) clearance.

Run: python tests/handoff_acceptance.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from splatsim.utils.rrt_path_utils import parametrize_path  # noqa: E402

FPS = 30
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def _load_corpus_case():
    f = sorted(
        os.path.join(d, x)
        for d in [os.path.join(os.path.dirname(__file__), "data", "follower_corpus")]
        for x in os.listdir(d)
        if x.endswith(".pkl")
    )[3]
    d = pickle.load(open(f, "rb"))
    return np.asarray(d["waypoints"], dtype=np.float64), dict(d["kwargs"])


def section_a_parametrizer():
    print("--- A: parametrizer velocity carry ---")
    wp, kw = _load_corpus_case()
    dof = wp.shape[1]
    lims = (np.full(dof, 0.5), np.full(dof, 1.0), np.full(dof, 10.0))
    d0 = wp[1] - wp[0]
    d0u = d0 / np.linalg.norm(d0)
    perp = np.array([-d0u[1], d0u[0], 0.0])
    perp -= d0u * np.dot(perp, d0u)
    perp /= np.linalg.norm(perp)

    def run(vhat, v0, lead, label, exempt_caps=False, min_carry=0.9):
        wp2 = wp if lead is None else np.vstack([wp[:1], (wp[0] + vhat * lead)[None], wp[1:]])
        kw2 = dict(kw)
        kw2["start_vel"] = vhat * v0
        tr = parametrize_path(wp2, *lims, backend="retimed", **kw2)
        vel0 = (tr[1] - tr[0]) * FPS
        sp0 = float(np.linalg.norm(vel0))
        al = float(np.dot(vel0 / max(sp0, 1e-9), vhat))
        a = np.diff(np.diff(tr, axis=0) * FPS, axis=0) * FPS
        j = np.diff(a, axis=0) * FPS
        lim_ok = exempt_caps or (np.abs(a).max() <= 1.05 and np.abs(j).max() <= 11.5)
        check(
            label,
            sp0 >= min_carry * v0 and al > 0.95 and lim_ok,
            f"speed {sp0:.2f}/{v0} align {al:+.2f} amax {np.abs(a).max():.2f} jmax {np.abs(j).max():.1f}",
        )

    d_lead = lambda v: v * v / 1.4 + v / 10.0
    run(d0u, 0.30, None, "aligned handoff")
    for ang in (60, 90, 110):
        th = np.radians(ang)
        vh = np.cos(th) * d0u + np.sin(th) * perp
        run(vh, 0.43, d_lead(0.43), f"mid-band {ang}deg lead-in")
    for ang in (150, 180):
        th = np.radians(ang)
        vh = np.cos(th) * d0u + np.sin(th) * perp
        run(vh, 0.40, d_lead(0.40), f"reversal {ang}deg brake-out")
    # short runway: escalated brake (caps deliberately exceeded)
    run(-d0u, 0.40, 0.5 * d_lead(0.40), "short-runway 0.5x (escalated)", exempt_caps=True)
    run(-d0u, 0.40, 0.25 * d_lead(0.40), "short-runway 0.25x (escalated)", exempt_caps=True)

    # EXACT-CARRY overspeed: handoff above vmax (0.5) and therefore above
    # any start license. Launch must be AT the handed speed (no clamp), the
    # overlay must brake monotonically, and the profile must be back inside
    # vmax within a second (surplus shed early, remainder legal).
    def run_overspeed(vhat, v0, label, cusp_lead=None):
        wp2 = (
            wp if cusp_lead is None
            else np.vstack([wp[:1], (wp[0] + vhat * cusp_lead)[None], wp[1:]])
        )
        kw2 = dict(kw)
        kw2["start_vel"] = vhat * v0
        tr = parametrize_path(wp2, *lims, backend="retimed", **kw2)
        sp = np.linalg.norm(np.diff(tr, axis=0), axis=1) * FPS
        d1 = tr[1] - tr[0]
        al = float(np.dot(d1 / max(np.linalg.norm(d1), 1e-12), vhat))
        # Legal L2 ceiling under uniform_path_speed = per-joint cap * sqrt(dof)
        vmax = float(lims[0].max()) * float(np.sqrt(dof))
        over = sp > vmax * 1.001
        shed_tick = int(np.argmin(over)) if not over.all() else len(sp)
        mono = bool(np.all(np.diff(sp[: shed_tick + 1]) <= 1e-6)) if shed_tick > 0 else True
        legal_after = bool(np.all(sp[shed_tick:] <= vmax * 1.05))
        check(
            label,
            abs(sp[0] - v0) <= 0.02 * v0 and al > 0.95 and mono
            and legal_after and shed_tick <= FPS,
            f"launch {sp[0]:.3f}/{v0} align {al:+.2f} shed@{shed_tick} "
            f"ticks mono={mono} post-shed max {sp[shed_tick:].max():.2f}",
        )

    run_overspeed(d0u, 1.30, "overspeed aligned 1.5x L2 ceiling")
    run_overspeed(d0u, 1.00, "overspeed aligned 1.15x L2 ceiling")
    run_overspeed(-d0u, 1.00, "overspeed reversal brake-out 1.15x ceiling",
                  cusp_lead=d_lead(1.00))


def section_a2_stalled_onset():
    """Stalled handoff (start_vel ~ 0): the plan must RAMP, not launch at
    cruise. Interventions overwhelmingly trigger on the stuck heuristic, so
    the handed velocity is ~zero; before 2026-09-05 the strict onset row was
    gated to moving handoffs and a stalled takeover launched at ~1.1x cruise
    in one tick (start-speed audit of 04dagpl vs 03dag intervention data),
    producing onset labels the policy cannot follow. Demo generation (no
    start_vel) intentionally keeps the fast launch."""
    print("--- A2: stalled-handoff onset ramp ---")
    wp, kw = _load_corpus_case()
    dof = wp.shape[1]
    lims = (np.full(dof, 0.5), np.full(dof, 1.0), np.full(dof, 10.0))

    def launch_profile(with_sv, v0=0.0):
        kw2 = dict(kw)
        if with_sv:
            d0 = wp[1] - wp[0]
            kw2["start_vel"] = (d0 / np.linalg.norm(d0)) * v0
        else:
            kw2.pop("start_vel", None)
        tr = parametrize_path(wp, *lims, backend="retimed", **kw2)
        return np.linalg.norm(np.diff(tr, axis=0), axis=1) * FPS

    sp = launch_profile(True, 0.0)
    cruise = float(np.percentile(sp, 90))
    ramp_ok = sp[0] <= 0.25 * cruise
    mono_ok = bool(np.all(np.diff(sp[:5]) >= -1e-6))
    a = np.diff(sp) * FPS
    reach = int(np.argmax(sp >= 0.8 * cruise)) if np.any(sp >= 0.8 * cruise) else len(sp)
    check(
        "A2 stalled handoff ramps (start_vel=0)",
        ramp_ok and mono_ok and reach <= 2 * FPS and np.abs(a).max() <= 1.6,
        f"sp0 {sp[0]:.3f} cruise {cruise:.2f} reach80% @{reach} amax {np.abs(a).max():.2f}",
    )
    sp2 = launch_profile(True, 0.005)
    check(
        "A2 near-zero handoff ramps too (start_vel=0.005)",
        sp2[0] <= 0.25 * cruise,
        f"sp0 {sp2[0]:.3f}",
    )
    spd = launch_profile(False)
    check(
        "A2 demo-generation launch unchanged (no start_vel; fast start kept)",
        spd[0] >= 2.0 * sp[0] or spd[0] >= 0.5 * cruise,
        f"demo sp0 {spd[0]:.3f} vs handoff sp0 {sp[0]:.3f}",
    )


def section_b_planner():
    print("--- B: planner start treatment ---")
    import pybullet as pb
    import pybullet_utils.bullet_client as bc

    from splatsim.utils.rrt_to_goal import RRTToGoalPlanner

    client = bc.BulletClient(pb.DIRECT)
    urdf = os.path.join(
        os.path.dirname(__file__), "..", "splatsim", "robot_definitions", "urdf", "planar_3joint.urdf"
    )
    robot = client.loadURDF(urdf, useFixedBase=True)
    pl = RRTToGoalPlanner(
        pb_client=client._client, robot_id=robot, joint_indices=[1, 2, 3],
        ee_link_index=15, num_dofs=3, fps=30,
    )
    pl.load_obstacles({"objects": []})
    free = lambda q: False

    q0 = np.array([0.0, 0.5, -0.3])
    path = np.array([q0, q0 + [0.4, -0.1, 0.2], q0 + [0.9, -0.3, 0.5]])
    d0u = (path[1] - path[0]) / np.linalg.norm(path[1] - path[0])

    # B1: mid-band misalignment (dot ~0.87 = 30 deg) gets the CURVED
    # REDIRECT: start tangent along the handed velocity, NO cusp anywhere
    # (adjacent-segment dot > -0.5 throughout) — the no-stop replacement for
    # the straight lead-in + brake-out (which manufactured a full stop out
    # of what is task-space-continuous motion; planar scenario 6,
    # 2026-08-17).
    perp = np.array([-d0u[1], d0u[0], 0.0])
    perp -= d0u * np.dot(perp, d0u)
    perp /= np.linalg.norm(perp)
    vh = np.cos(np.radians(30)) * d0u + np.sin(np.radians(30)) * perp
    out = pl._straighten_terminal(path.copy(), free, start_vel=vh * 0.4)
    first = (out[1] - out[0]) / max(np.linalg.norm(out[1] - out[0]), 1e-12)

    def _no_cusp(p):
        d = np.diff(p, axis=0)
        n = np.linalg.norm(d, axis=1)
        keep = n > 1e-9
        d = d[keep] / n[keep][:, None]
        return bool(np.all(np.sum(d[:-1] * d[1:], axis=1) > -0.5)) if len(d) > 1 else True

    check("B1 mid-band curved redirect (no cusp)",
          len(out) > len(path) and float(np.dot(first, vh)) > 0.97 and _no_cusp(out),
          f"vertices {len(path)}->{len(out)} cusp-free={_no_cusp(out)}")

    # B2: straighten_only fallback preserves the redirect
    out2 = pl._postprocess_path(path.copy(), start_vel=vh * 0.4, straighten_only=True)
    first2 = (out2[1] - out2[0]) / max(np.linalg.norm(out2[1] - out2[0]), 1e-12)
    check("B2 straighten_only keeps redirect", len(out2) > len(path) and float(np.dot(first2, vh)) > 0.97)

    # B4: end-to-end no-stop — the redirect's parametrized profile must
    # CARRY speed through the turn (dip allowed at the curvature ceiling,
    # never a stop), at 30 and 90 deg of joint-space misalignment.
    # B5: contact-clearance second pass — when the planning-clearance sweep
    # is fully blocked (shield handoffs fire NEAR obstacles by construction)
    # but contact level is free, the arc must still be placed instead of
    # falling back to the straight lead-in's cusp (the post-fix 2026-08-17
    # run still logged 26 lead-in fallbacks vs 24 placed arcs).
    mock_full5 = lambda q: True      # planning clearance: everything collides
    mock_contact5 = lambda q: False  # contact clearance: free
    out5 = pl._straighten_terminal(
        path.copy(), mock_full5, start_vel=vh * 0.4, contact_fn=mock_contact5
    )
    first5 = (out5[1] - out5[0]) / max(np.linalg.norm(out5[1] - out5[0]), 1e-12)
    check("B5 arc placed via contact-clearance pass",
          len(out5) > len(path) and float(np.dot(first5, vh)) > 0.97 and _no_cusp(out5),
          f"vertices {len(path)}->{len(out5)} cusp-free={_no_cusp(out5)}")

    # B6: DENSE-path redirect — real planner output has chords of a few cm,
    # and an arc forced to complete its turn within the chord to the raw
    # next vertex is always too curved (2026-08-18 run: 26/26 arc failures
    # "too-curved", zero colliding; every one fell back to the cusp/stop).
    # The multi-target search must give the turn room by aiming at farther
    # path vertices: dense near-perpendicular handoffs place a cusp-free
    # redirect and the parametrized profile carries speed.
    dpath = np.array([q0 + d0u * (0.04 * k) for k in range(26)])
    for ang in (88, 110):
        vh_d = np.cos(np.radians(ang)) * d0u + np.sin(np.radians(ang)) * perp
        out_d = pl._straighten_terminal(dpath.copy(), free, start_vel=vh_d * 0.44)
        tr = parametrize_path(
            out_d, np.full(3, 0.5), np.full(3, 1.0), np.full(3, 10.0),
            control_hz=30, backend="retimed", start_vel=vh_d * 0.44,
            segment_at_sharp_corners=False, uniform_path_speed=True,
        )
        sp = np.linalg.norm(np.diff(np.asarray(tr), axis=0), axis=1) * 30.0
        core = sp[: int(0.6 * len(sp))]
        check(f"B6 dense-path {ang}deg redirect (no stop)",
              _no_cusp(out_d) and float(sp[0]) > 0.3 and float(core.min()) > 0.1,
              f"launch {sp[0]:.2f} min-through-turn {core.min():.2f}")

    # Speed floor scales with turn depth (mirrors _curved_redirect's
    # depth-scaled carry floor): mild turns barely dent cruise; a hairpin
    # legitimately slows hard through the apex — but NEVER stops.
    for ang, floor in ((30, 0.12), (90, 0.12), (135, 0.07)):
        vh_a = np.cos(np.radians(ang)) * d0u + np.sin(np.radians(ang)) * perp
        out_a = pl._straighten_terminal(path.copy(), free, start_vel=vh_a * 0.4)
        tr = parametrize_path(
            out_a, np.full(3, 0.5), np.full(3, 1.0), np.full(3, 10.0),
            control_hz=30, backend="retimed", start_vel=vh_a * 0.4,
            segment_at_sharp_corners=False, uniform_path_speed=True,
        )
        sp = np.linalg.norm(np.diff(np.asarray(tr), axis=0), axis=1) * 30.0
        mid = sp[: int(0.6 * len(sp))]  # transition region (excludes goal taper)
        launch, vmin = float(sp[0]), float(mid.min())
        check(f"B4 {ang}deg redirect carries speed (no stop)",
              _no_cusp(out_a) and launch > 0.3 and vmin > floor,
              f"launch {launch:.2f} min-through-turn {vmin:.2f} (floor {floor})")

    def _ee_at(q):
        for j, qi in zip([1, 2, 3], q):
            client.resetJointState(robot, j, float(qi))
        return np.array(client.getLinkState(robot, 15, computeForwardKinematics=True)[0])

    def _sweep_midpoint(vhat, v0):
        """EE position halfway along the braking spur — obstacles get placed
        relative to the COMPUTED sweep, not to guessed kinematics (two prior
        test iterations failed purely on wrong sweep-direction guesses)."""
        d = v0 * v0 / 1.4 + v0 / 10.0
        return _ee_at(q0 + vhat * (0.5 * d)), d

    # B3: emergency contact-clearance lead-in when plan clearance is
    # blocked. MOCKED collision functions (a physically-placed blocker is
    # brittle against the full arm-sweep geometry — three prior attempts
    # failed on incidental finger/base contacts): full-clearance blocks the
    # spur beyond 0.02 rad, contact level blocks beyond 0.08 — the
    # emergency search must place a shortened lead-in (~0.08 < full 0.12).
    vh2 = -d0u
    proj = lambda q: float(np.dot(np.asarray(q) - q0, vh2))
    mock_full = lambda q: proj(q) > 0.02
    mock_contact = lambda q: proj(q) > 0.08
    out3 = pl._straighten_terminal(
        path.copy(), mock_full, start_vel=vh2 * 0.35, contact_fn=mock_contact
    )
    placed = len(out3) > len(path)
    exc = float(np.linalg.norm(out3[1] - out3[0])) if placed else 0.0
    first3 = (out3[1] - out3[0]) / max(np.linalg.norm(out3[1] - out3[0]), 1e-12) if placed else vh2 * 0
    check("B3 emergency lead-in with shortened runway",
          placed and 0.03 <= exc <= 0.085 and float(np.dot(first3, vh2)) > 0.99,
          f"vertices {len(out3)} excursion {exc:.3f} (full braking needs 0.122)")

    # C: brake feasibility directionality (emergency clearance)
    print("--- C: brake feasibility ---")
    v_toward = np.array([0.4, 0.0, 0.0])
    mid_c, _ = _sweep_midpoint(v_toward / np.linalg.norm(v_toward), 0.4)
    obs2 = client.createMultiBody(
        0, client.createCollisionShape(pb.GEOM_SPHERE, radius=0.02),
        basePosition=mid_c.tolist(),  # ON the computed sweep
    )
    pl._loaded_obstacle_ids.append(obs2)
    pl._obstacle_names[obs2] = "c_sphere"
    check("C1 rest is feasible... near obstacle on future sweep",
          pl.is_brake_feasible(q0, np.zeros(3)))
    check("C2 toward obstacle infeasible", not pl.is_brake_feasible(q0, v_toward))


    # C3: `is_handoff_runway_free` is the planning-clearance sibling of
    # `is_brake_feasible` (same spur, 0.02 vs 0.01 clearance) — the shield
    # micro-rewind prefers runway-free states so handoffs launch with real
    # runway instead of braking in place. The distinction is real: scanning
    # a lateral obstacle toward the braking sweep must produce a band where
    # an emergency brake still fits but planning-clearance runway does not.
    pl._loaded_obstacle_ids.remove(obs2)
    del pl._obstacle_names[obs2]
    client.removeBody(obs2)
    v_back = -v_toward
    mid_b, _ = _sweep_midpoint(v_back / np.linalg.norm(v_back), 0.4)
    w = np.array([0.0, 1.0, 0.0])  # out-of-plane for the planar arm
    band = None
    states = []
    for off in np.arange(0.02, 0.12, 0.002):
        sp3 = client.createMultiBody(
            0, client.createCollisionShape(pb.GEOM_SPHERE, radius=0.02),
            basePosition=(mid_b + w * off).tolist(),
        )
        pl._loaded_obstacle_ids.append(sp3)
        pl._obstacle_names[sp3] = "c3_sphere"
        brake = pl.is_brake_feasible(q0, v_back)
        runway = pl.is_handoff_runway_free(q0, v_back)
        pl._loaded_obstacle_ids.remove(sp3)
        del pl._obstacle_names[sp3]
        client.removeBody(sp3)
        states.append((round(float(off), 3), brake, runway))
        if brake and not runway and band is None:
            band = float(off)
    far_ok = states[-1][1] and states[-1][2]  # far obstacle: both pass
    check("C3 runway check stricter than brake check (clearance band exists)",
          band is not None and far_ok,
          f"band at offset {band}, far state {states[-1]}")

    # B7: start-zone contact exemption in the plan-level re-gate. The
    # straightened handoff prefix (lead-in / redirect) may legitimately sit
    # at contact-level clearance; re-gating it at full planning clearance
    # silently vetoed it and the chunk executed WITHOUT the redirect (ep0
    # T=329, 2026-08-18). Scan a lateral obstacle toward a straight path's
    # PREFIX to find the band where planning clearance fails but contact
    # level passes; the gate must fail without the exemption and pass with
    # it. A violation placed BEYOND the zone must still fail either way.
    # Mirror production clearances (the bare test planner has none — the
    # exemption is only meaningful when planning clearance > contact).
    pl._collision_kwargs["obstacle_clearance"] = 0.02
    d7 = np.array([0.4, -0.1, 0.2]); d7 /= np.linalg.norm(d7)
    path7 = np.array([q0 + d7 * (0.12 * k) for k in range(15)])  # ~1.68 rad (> the 1.2 zone)
    def _ee_mid(frac):
        return _ee_at(q0 + d7 * (1.68 * frac))
    w7 = np.array([0.0, 1.0, 0.0])
    band_off = None
    for off in np.arange(0.02, 0.12, 0.002):
        sp7 = client.createMultiBody(
            0, client.createCollisionShape(pb.GEOM_SPHERE, radius=0.02),
            basePosition=(_ee_mid(0.1) + w7 * off).tolist(),
        )
        pl._loaded_obstacle_ids.append(sp7)
        pl._obstacle_names[sp7] = "b7_sphere"
        _, strict_coll = pl._densify_and_check_collision(path7)
        _, zone_coll = pl._densify_and_check_collision(path7, start_contact_arc=1.2)
        pl._loaded_obstacle_ids.remove(sp7)
        del pl._obstacle_names[sp7]
        client.removeBody(sp7)
        if strict_coll is not None and zone_coll is None:
            band_off = float(off)
            break
    check("B7 start-zone contact exemption admits prefix-margin geometry",
          band_off is not None, f"band at offset {band_off}")
    if band_off is not None:
        sp7b = client.createMultiBody(
            0, client.createCollisionShape(pb.GEOM_SPHERE, radius=0.02),
            basePosition=(_ee_mid(0.9) + w7 * band_off).tolist(),
        )
        pl._loaded_obstacle_ids.append(sp7b)
        pl._obstacle_names[sp7b] = "b7_sphere_far"
        _, far_coll = pl._densify_and_check_collision(path7, start_contact_arc=1.2)
        pl._loaded_obstacle_ids.remove(sp7b)
        del pl._obstacle_names[sp7b]
        client.removeBody(sp7b)
        check("B7b beyond-zone violations still gated",
              far_coll is not None, f"coll idx {far_coll}")
    pl._collision_kwargs.pop("obstacle_clearance", None)


def section_d_ctor_drift():
    """Guard: features added to RRTToGoalPlanner must be threaded into BOTH
    call sites — SplatSim's TrajectoryGenerator (demo generation) and
    lerobot's RRTGuidanceSource (interventions) — unless intentionally
    exempted below. Camera scoring lived generation-only for a while and
    intervention RRT silently lacked it; this section makes that class of
    drift a test failure instead of a video-review surprise."""
    print("--- D: generation<->intervention ctor drift ---")
    import ast as _ast

    def kwargs_of_call(path, funcname="RRTToGoalPlanner"):
        tree = _ast.parse(open(path).read())
        names = set()

        def collect(call):
            for kw in call.keywords:
                if kw.arg:
                    names.add(kw.arg)
                else:  # **{...} of (name, value) pairs / dict-comprehensions
                    for sub in _ast.walk(kw.value):
                        if isinstance(sub, _ast.Constant) and isinstance(sub.value, str):
                            names.add(sub.value)

        for node in _ast.walk(tree):
            if isinstance(node, _ast.Call):
                f = node.func
                if getattr(f, "id", getattr(f, "attr", None)) == funcname:
                    collect(node)
            # The SA source assembles its kwargs as `_explicit = dict(...)`
            # and constructs via RRTToGoalPlanner(**{**env_base, **_explicit})
            # (the env-config lowest-priority merge) — scan that dict too.
            if isinstance(node, _ast.Assign) and len(node.targets) == 1:
                t = node.targets[0]
                if (getattr(t, "id", None) == "_explicit"
                        and isinstance(node.value, _ast.Call)
                        and getattr(node.value.func, "id", None) == "dict"):
                    collect(node.value)
        return names

    gen = kwargs_of_call(os.path.join(os.path.dirname(__file__), "..", "splatsim", "utils", "trajectory_generation.py"))
    sa_path = os.path.expanduser("~/code/lerobot/src/lerobot/policies/guidance/rrt_source.py")
    if not os.path.isfile(sa_path):
        check("D1 ctor drift (lerobot missing, skipped)", True)
        return
    sa = kwargs_of_call(sa_path)
    # Intentional differences (justify each):
    exempt = {
        # generation-only by design:
        "plan_rng_seed",            # per-scene determinism; would deadlock SA retries
        "freeze_visualizer_during_plan",
        "soft_cost_mode", "soft_cost_weight", "soft_cost_points_per_link",
        "soft_cost_max_reduction", "soft_cost_surface_offsets",
        "soft_cost_aggregation", "soft_cost_surface_samples",
        # SA inherits the shared planner default (the None-inherit rule);
        # only thread if a generation CONFIG starts overriding it:
        "obstacle_clearance_factor",
        "camera_k_exp", "camera_k_sig", "camera_threshold",  # SA inherits planner defaults
        "parametrize_per_candidate",
        # SA-only by design:
        "in_progress_obstacle_clearance", "in_progress_self_collision_clearance",
        "escape_clearance_factor", "rewind_clearance_factor",
        "diagnostic_log_pairs", "ik_accept_arc_chord_ratio",
        "velocity_match_window", "ik_goal_selection",
        # construction plumbing (both pass, names may differ):
        "pb_client", "robot_id", "joint_indices", "ee_link_index", "num_dofs",
        "fps", "lower_limits", "upper_limits", "wrist_camera_link_index",
    }
    missing_in_sa = (gen - sa) - exempt
    missing_in_gen = (sa - gen) - exempt
    check("D1 generation-only planner kwargs threaded into SA", not missing_in_sa,
          f"missing in SA: {sorted(missing_in_sa)}" if missing_in_sa else "")
    check("D2 SA-only planner kwargs known to generation or exempt", not missing_in_gen,
          f"missing in generation: {sorted(missing_in_gen)}" if missing_in_gen else "")


def section_e_env_config_inherit():
    """The env traj-config JSON is the LOWEST-priority defaults layer for
    intervention planners: name-consistent keys pass through verbatim,
    plan_rng_seed is excluded (deterministic replans would deadlock SA
    retries), and explicit kwargs always win in the merge."""
    print("--- E: env traj-config inheritance ---")
    from splatsim.utils.rrt_to_goal import planner_kwargs_from_traj_config

    base = planner_kwargs_from_traj_config("planar_3joint")
    check("E1 camera scoring inherited from JSON",
          base.get("camera_score_weight") == 0.5 and "camera_k_exp" in base)
    check("E2 plan_rng_seed excluded", "plan_rng_seed" not in base)
    check("E3 unknown env -> empty", planner_kwargs_from_traj_config("no_such_env") == {})
    merged = {**base, **{"num_ik_candidates": 32, "camera_score_weight": 0.0}}
    check("E4 explicit values win in merge",
          merged["num_ik_candidates"] == 32 and merged["camera_score_weight"] == 0.0)
    # Name consistency: every JSON key that LOOKS like a planner knob must
    # BE a planner ctor param verbatim (the rename-table era is over).
    import inspect, json as _json
    from splatsim.utils.paths import traj_config_path
    from splatsim.utils.rrt_to_goal import RRTToGoalPlanner
    cfg = _json.load(open(traj_config_path("planar_3joint")))
    params = set(inspect.signature(RRTToGoalPlanner.__init__).parameters)
    legacy = {"k_exp", "k_sig", "threshold", "num_path_candidates"}
    check("E5 no legacy divergent names in config", not (set(cfg) & legacy),
          f"legacy keys present: {sorted(set(cfg) & legacy)}" if set(cfg) & legacy else "")

    # E6: taper drift guard. The traj-config JSONs restate the final-approach
    # taper verbatim (GUI exports write every field), and because the JSON is
    # an OVERRIDE layer, a retune of planner_defaults silently fails to
    # propagate to any env whose JSON still carries the old numbers — that is
    # exactly how the 2026-08-17 taper retune would have missed interventions.
    # Every JSON must match PLANNER_DEFAULTS on these keys unless the env is
    # declared here as an intentional override.
    import glob as _glob
    from splatsim.utils.paths import TRAJ_CONFIG_DIR
    from splatsim.utils.planner_defaults import PLANNER_DEFAULTS as _PD
    TAPER_KEYS = ("final_approach_dist", "final_approach_vel_scale", "final_approach_acc_scale")
    INTENTIONAL_TAPER_OVERRIDES: dict[str, dict] = {}  # env-file basename -> {key: value}
    drift = []
    for jf in sorted(_glob.glob(str(TRAJ_CONFIG_DIR / "*.json"))):
        c = _json.load(open(jf))
        allowed = INTENTIONAL_TAPER_OVERRIDES.get(os.path.basename(jf), {})
        for k in TAPER_KEYS:
            if k in c and c[k] != allowed.get(k, getattr(_PD, k)):
                drift.append(f"{os.path.basename(jf)}:{k}={c[k]} (default {getattr(_PD, k)})")
    check("E6 taper values match planner defaults in every env JSON", not drift,
          "; ".join(drift) if drift else "")


def main():
    section_a_parametrizer()
    section_a2_stalled_onset()
    section_b_planner()
    section_d_ctor_drift()
    section_e_env_config_inherit()
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURE(S): ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
