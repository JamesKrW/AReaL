from view_suite.envs.view_spatial_bench.gym_view_spatial_no_tool import ViewSpatialNoToolGym
from view_suite.envs.view_spatial_bench.gym_view_spatial_tool import ViewSpatialToolGym
from view_suite.envs.sokoban.gym_sokoban import GymSokoban
from view_suite.envs.scannet_proxy_task.gym_proxy_tool import GymProxyTool
from view_suite.envs.scannet_proxy_task.gym_proxy_no_tool import GymProxyNoTool

REGISTERED_ENVS = {
    "ViewSpatialNoToolGym": ViewSpatialNoToolGym,
    "ViewSpatialToolGym": ViewSpatialToolGym,
    "GymSokoban": GymSokoban,
    "GymProxyTool": GymProxyTool,
    "GymProxyNoTool": GymProxyNoTool,
}