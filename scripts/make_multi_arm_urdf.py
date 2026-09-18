"""Compose one URDF from a base and N copies of an arm URDF — the shipped
`data/assets/dual_ur5/` was made with this:

    python scripts/make_multi_arm_urdf.py \\
        --arm data/assets/ur5/ur5.urdf --prefix left_ --mount "0.1 0.3" \\
        --arm data/assets/ur5/ur5.urdf --prefix right_ --mount "0.1 -0.3" \\
        --base-size 0.6 0.9 0.3 --wheels --out data/assets/dual_ur5/dual_ur5.urdf

Every link, joint, material and <mimic> reference of each arm gets its
prefix, a `world` root link (and the fixed joint under it) is dropped, and
mesh paths are rewritten relative to the output folder. `--wheels` adds a
differential drive: two driven wheels on the base's y axis (joints
`left_wheel_joint` / `right_wheel_joint`) plus two frictionless caster
spheres, sized from --wheel-radius / --track-width.
"""
import argparse
import copy
import os
import xml.etree.ElementTree as ET


def _fmt(v):
    return " ".join(f"{float(x):g}" for x in v)


def _prefix_arm(arm_path: str, prefix: str, out_dir: str) -> tuple:
    """(links, joints, materials) of the arm with names prefixed and mesh
    paths made relative to out_dir. Returns the root link's name too."""
    tree = ET.parse(arm_path)
    root = tree.getroot()
    arm_dir = os.path.dirname(os.path.abspath(arm_path))
    links, joints, materials = [], [], []
    for el in list(root):
        el = copy.deepcopy(el)
        if el.tag == "material":
            el.set("name", prefix + el.get("name"))
            materials.append(el)
        elif el.tag == "link":
            el.set("name", prefix + el.get("name"))
            links.append(el)
        elif el.tag == "joint":
            el.set("name", prefix + el.get("name"))
            el.find("parent").set("link", prefix + el.find("parent").get("link"))
            el.find("child").set("link", prefix + el.find("child").get("link"))
            m = el.find("mimic")
            if m is not None:
                m.set("joint", prefix + m.get("joint"))
            joints.append(el)
    for el in links + joints:
        for mat in el.iter("material"):
            if mat.get("name"):
                mat.set("name", prefix + mat.get("name"))
        for mesh in el.iter("mesh"):
            fn = mesh.get("filename", "")
            if fn.startswith("package://") or os.path.isabs(fn):
                continue
            mesh.set("filename", os.path.relpath(os.path.join(arm_dir, fn), out_dir).replace(os.sep, "/"))
    # Drop a `world` root and its fixed joint: the arm hangs off our base.
    child_of = {j.find("child").get("link"): j for j in joints}
    parents = {j.find("parent").get("link") for j in joints}
    roots = [l.get("name") for l in links if l.get("name") not in child_of]
    for r in list(roots):
        if r.endswith("world"):
            links = [l for l in links if l.get("name") != r]
            joints = [j for j in joints if j.find("parent").get("link") != r]
            roots.remove(r)
            child_of = {j.find("child").get("link"): j for j in joints}
            roots += [l.get("name") for l in links if l.get("name") not in child_of and l.get("name") not in roots]
    assert len(roots) == 1, f"{arm_path}: expected one root link after dropping world, got {roots}"
    return links, joints, materials, roots[0]


def _box_link(name, size, mass, rgba, z_offset):
    link = ET.Element("link", name=name)
    for tag in ("visual", "collision"):
        el = ET.SubElement(link, tag)
        ET.SubElement(ET.SubElement(el, "geometry"), "box", size=_fmt(size))
        ET.SubElement(el, "origin", xyz=_fmt((0, 0, z_offset)), rpy="0 0 0")
        if tag == "visual":
            ET.SubElement(ET.SubElement(el, "material", name=f"{name}_color"), "color", rgba=_fmt(rgba))
    # Inertial frame AT the link origin (not the box centre): pybullet's
    # base pose is the base link's inertial frame, so this keeps the yaml
    # base_position / placement sliders / getBasePositionAndOrientation all
    # meaning "the point between the wheels on the ground". The low COM also
    # makes the base hard to tip.
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:g}")
    x, y, z = size
    ET.SubElement(inertial, "inertia", ixx=f"{mass*(y*y+z*z)/12:g}", iyy=f"{mass*(x*x+z*z)/12:g}",
                  izz=f"{mass*(x*x+y*y)/12:g}", ixy="0", ixz="0", iyz="0")
    return link


def _wheel(name, radius, width, mass, rgba, xyz, driven: bool):
    """A cylinder wheel spinning about the base's y axis (driven), or a
    frictionless caster sphere (not driven)."""
    link = ET.Element("link", name=f"{name}_link")
    for tag in ("visual", "collision"):
        el = ET.SubElement(link, tag)
        g = ET.SubElement(el, "geometry")
        if driven:
            ET.SubElement(g, "cylinder", radius=f"{radius:g}", length=f"{width:g}")
            ET.SubElement(el, "origin", xyz="0 0 0", rpy=_fmt((1.5708, 0, 0)))
        else:
            ET.SubElement(g, "sphere", radius=f"{radius:g}")
        if tag == "visual":
            ET.SubElement(ET.SubElement(el, "material", name=f"{name}_color"), "color", rgba=_fmt(rgba))
        else:
            # pybullet reads <contact> for friction; casters slide freely.
            pass
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value=f"{mass:g}")
    i = 0.4 * mass * radius * radius
    ET.SubElement(inertial, "inertia", ixx=f"{i:g}", iyy=f"{i:g}", izz=f"{i:g}", ixy="0", ixz="0", iyz="0")
    if not driven:
        c = ET.SubElement(link, "contact")
        ET.SubElement(c, "lateral_friction", value="0.0")
        ET.SubElement(c, "rolling_friction", value="0.0")
        ET.SubElement(c, "spinning_friction", value="0.0")
    joint = ET.Element("joint", name=f"{name}_joint", type="continuous" if driven else "fixed")
    ET.SubElement(joint, "parent", link="base")
    ET.SubElement(joint, "child", link=f"{name}_link")
    ET.SubElement(joint, "origin", xyz=_fmt(xyz), rpy="0 0 0")
    if driven:
        ET.SubElement(joint, "axis", xyz="0 1 0")
        ET.SubElement(joint, "limit", effort="50", velocity="20")
    return link, joint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="arm URDF (repeat per arm)")
    ap.add_argument("--prefix", action="append", required=True, help="name prefix for that arm (repeat)")
    ap.add_argument("--mount", action="append", required=True, help='"x y" (on the box top) or "x y z" of the arm root (repeat)')
    ap.add_argument("--mount-rpy", action="append", default=None, help='"r p y" of the arm root (repeat; default 0 0 0)')
    ap.add_argument("--name", default=None, help="robot name (default: output stem)")
    ap.add_argument("--base-size", nargs=3, type=float, default=(0.6, 0.9, 0.3), help="box base x y z (m); its bottom is z=0")
    ap.add_argument("--base-mass", type=float, default=60.0)
    ap.add_argument("--wheels", action="store_true", help="add a differential drive under the box")
    ap.add_argument("--wheel-radius", type=float, default=0.1)
    ap.add_argument("--wheel-width", type=float, default=0.05)
    ap.add_argument("--track-width", type=float, default=None, help="driven-wheel separation (default: wheels just outside the box)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = len(args.arm)
    if not (len(args.prefix) == len(args.mount) == n):
        ap.error("--arm, --prefix and --mount must be given the same number of times")
    rpys = args.mount_rpy or ["0 0 0"] * n
    if len(rpys) != n:
        ap.error("--mount-rpy must be given once per arm (or not at all)")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    robot = ET.Element("robot", name=args.name or os.path.splitext(os.path.basename(args.out))[0])
    bx, by, bz = args.base_size
    # With wheels the box floats one wheel radius above the ground (wheel
    # axles sit at the box's bottom face).
    lift = args.wheel_radius if args.wheels else 0.0
    robot.append(_box_link("base", (bx, by, bz), args.base_mass, (0.35, 0.35, 0.38, 1), lift + bz / 2))
    if args.wheels:
        r, w = args.wheel_radius, args.wheel_width
        # Wheels clear of the box sides (the sim loads with self-collision on;
        # only the direct parent is excluded), casters below its bottom face.
        track = args.track_width or (by + w + 0.02)
        for name, y in (("left_wheel", track / 2), ("right_wheel", -track / 2)):
            link, joint = _wheel(name, r, w, 2.0, (0.1, 0.1, 0.1, 1), (0.0, y, lift), driven=True)
            robot.append(link); robot.append(joint)
        # Frictionless caster spheres under the box, 5 mm shy of the ground:
        # level casters would carry most of the weight (they sit at the box's
        # ends, the wheels at its middle) and the driven wheels would slip.
        # The base rocks ~1 degree onto one caster instead.
        rc = r / 2   # top at 2*rc = r = the box's bottom face
        for name, x in (("front_caster", bx / 2 - rc), ("back_caster", -bx / 2 + rc)):
            link, joint = _wheel(name, rc, w, 0.5, (0.2, 0.2, 0.2, 1), (x, 0.0, rc + 0.005), driven=False)
            robot.append(link); robot.append(joint)

    for arm, prefix, mount, rpy in zip(args.arm, args.prefix, args.mount, rpys):
        links, joints, materials, root_link = _prefix_arm(arm, prefix, out_dir)
        for m in materials:
            robot.append(m)
        j = ET.Element("joint", name=f"{prefix}mount_joint", type="fixed")
        ET.SubElement(j, "parent", link="base")
        ET.SubElement(j, "child", link=root_link)
        xyz = [float(v) for v in mount.split()]
        if len(xyz) == 2:
            xyz.append(lift + bz)   # on the box top
        ET.SubElement(j, "origin", xyz=_fmt(xyz), rpy=_fmt([float(v) for v in rpy.split()]))
        robot.append(j)
        for el in links + joints:
            robot.append(el)

    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(args.out, xml_declaration=True, encoding="utf-8")
    print(f"wrote {args.out}: {n} arm(s), base {bx}x{by}x{bz} m" + (", differential drive" if args.wheels else ""))


if __name__ == "__main__":
    main()
