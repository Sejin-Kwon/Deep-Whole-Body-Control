# from urdfpy import URDF
# import os

# URDF_PATH = "/home/sejink//Deep-Whole-Body-Control/legged_gym/resources/robots/go2viper/go2viper/go2_viperx.urdf"
# ROBOT_DIR = os.path.dirname(os.path.abspath(URDF_PATH))
# os.chdir(ROBOT_DIR)

# robot = URDF.load(os.path.basename(URDF_PATH))
# print("Links:", [l.name for l in robot.links])
# print("Joints:", [j.name for j in robot.joints])

# # 시각 메시 기반 간단 뷰어
# scene = robot.scene()  # trimesh Scene
# scene.show()


import os, numpy as np, trimesh
from urdfpy import URDF

URDF_PATH = "/home/sejink//Deep-Whole-Body-Control/legged_gym/resources/robots/go2viper/go2viper/go2_viperx_patched2.urdf"
# URDF_PATH = "/home/sejink//Deep-Whole-Body-Control/go2.urdf"
ROBOT_DIR = os.path.dirname(os.path.abspath(URDF_PATH))
os.chdir(ROBOT_DIR)

robot = URDF.load(os.path.basename(URDF_PATH))
print("Links:", [l.name for l in robot.links])
print("Joints:", [j.name for j in robot.joints])

# 모든 조인트 0으로 (필요시 dict로 특정 관절만 값 설정 가능)
cfg = {j.name: 0.0 for j in robot.joints}

# 링크 포즈(FK)
link_T = robot.link_fk(cfg=cfg)  # dict: {Link: 4x4}

scene = trimesh.Scene()

def as_np(T):
    if T is None:
        return np.eye(4)
    # urdfpy pose는 4x4 numpy일 것
    return np.array(T)

for link in robot.links:
    T_link = as_np(link_T.get(link, np.eye(4)))
    for vis in link.visuals or []:
        geom = vis.geometry
        T_vis = T_link @ as_np(getattr(vis, "origin", None))
        try:
            if geom.mesh is not None:
                # 메시 로드 + 스케일
                m = trimesh.load_mesh(geom.mesh.filename, force='mesh')
                if geom.mesh.scale is not None:
                    m.apply_scale(geom.mesh.scale)
                scene.add_geometry(m, transform=T_vis)
            elif geom.box is not None:
                m = trimesh.creation.box(extents=geom.box.size)
                scene.add_geometry(m, transform=T_vis)
            elif geom.cylinder is not None:
                r = geom.cylinder.radius; h = geom.cylinder.length
                m = trimesh.creation.cylinder(radius=r, height=h, sections=32)
                scene.add_geometry(m, transform=T_vis)
            elif geom.sphere is not None:
                m = trimesh.creation.icosphere(radius=geom.sphere.radius, subdivisions=3)
                scene.add_geometry(m, transform=T_vis)
        except Exception as e:
            print(f"[warn] skip visual on link {link.name}: {e}")

scene.show()  # pyglet 뷰어
