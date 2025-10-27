# load_asset_smoke.py
from isaacgym import gymapi, gymutil

URDF_PATH = "/home/sejink/Deep-Whole-Body-Control/legged_gym/resources/robots/go2viper/go2viper/go2_viperx.urdf"  # 현재 사용하는 파일로 교체
# URDF_PATH = "/home/sejink/Downloads/go2_viperx_isaac_sanitized.urdf"
            
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis               = gymapi.UP_AXIS_Z
sim_params.dt                    = 1.0/200.0
sim_params.substeps              = 1
sim_params.use_gpu_pipeline      = True        # ✅ CPU 파이프라인
sim_params.physx.use_gpu         = True        # ✅ CPU PhysX
sim_params.physx.num_position_iterations = 8
sim_params.physx.num_threads     = 8
sim_params.physx.contact_offset  = 0.02
sim_params.physx.rest_offset     = 0.0
sim_params.physx.solver_type = 1 

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise RuntimeError("Failed to create sim")

# Ground
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0,0,1)
gym.add_ground(sim, plane_params)

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# --- AssetOptions (이 버전에 존재하는 필드만 사용) ---
opts = gymapi.AssetOptions()
opts.fix_base_link = False
opts.disable_gravity = False
opts.collapse_fixed_joints = True             # 전체 병합 ON (mount joint에 dont_collapse 넣었으면 유지됨)
opts.replace_cylinder_with_capsule = True
opts.flip_visual_attachments = False
opts.armature = 0.01
opts.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE) 

asset = gym.load_asset(sim, "", URDF_PATH, opts)  # ← 여기서 크래시 나면 URDF/mesh 문제
env = gym.create_env(sim, gymapi.Vec3(-1,0,0), gymapi.Vec3(1,0,0), 1)
pose = gymapi.Transform(); pose.p.z = 0.35
actor = gym.create_actor(env, asset, pose, "robot", 0, 1)

print("Loaded OK. Stepping...")
for i in range(240):
    gym.simulate(sim); gym.fetch_results(sim, True)

print("Done.")
