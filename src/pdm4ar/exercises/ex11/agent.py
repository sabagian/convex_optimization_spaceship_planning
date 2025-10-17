from dataclasses import dataclass

from typing import Sequence

from dg_commons import DgSampledSequence, PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry, SpaceshipParameters

from pdm4ar.exercises.ex10 import test_myscenario
from pdm4ar.exercises.ex11.planner import SpaceshipPlanner
from pdm4ar.exercises_def.ex11.goal import SpaceshipTarget, DockingTarget
from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import cvxpy as cvx
import time

matplotlib.use("Agg")


@dataclass(frozen=True)
class MyAgentParams:
    """
    You can for example define some agent parameters.
    """

    my_tol: float = 0.1


class SpaceshipAgent(Agent):
    """
    This is the PDM4AR agent.
    Do *NOT* modify this class name
    Do *NOT* modify the naming of the existing methods and input/output types.
    """

    init_state: SpaceshipState
    satellites: dict[PlayerName, SatelliteParams]
    planets: dict[PlayerName, PlanetParams]
    goal_state: DynObstacleState

    cmds_plan: DgSampledSequence[SpaceshipCommands]
    state_traj: DgSampledSequence[SpaceshipState]
    myname: PlayerName
    planner: SpaceshipPlanner
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: SpaceshipGeometry
    sp: SpaceshipParameters
    X_error: np.ndarray
    test_case: int
    A: np.ndarray
    B: np.ndarray
    F: np.ndarray
    r: np.ndarray
    plotted = False
    X_interp: np.ndarray
    prob: cvx.Problem
    Ak: cvx.Parameter
    Bk: cvx.Parameter
    Fk: cvx.Parameter
    rk: cvx.Parameter
    x_current: cvx.Parameter
    x_ref_window: cvx.Parameter
    x: cvx.Variable
    u: cvx.Variable
    N: int

    def __init__(
        self,
        init_state: SpaceshipState,
        satellites: dict[PlayerName, SatelliteParams],
        planets: dict[PlayerName, PlanetParams],
    ):
        """
        Initializes the agent.
        This method is called by the simulator only before the beginning of each simulation.
        Provides the SpaceshipAgent with information about its environment, i.e. planet and satellite parameters and its initial position.
        """
        self.init_state = init_state
        self.satellites = satellites
        self.planets = planets
        self.plotted = False
        self.goal_state = None  # This will be set in on_episode_init

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        """
        This method is called by the simulator only at the beginning of each simulation.
        We suggest to compute here an initial trajectory/node graph/path, used by your planner to navigate the environment.

        Do **not** modify the signature of this method.
        """
        self.myname = init_sim_obs.my_name
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params
        assert isinstance(init_sim_obs.goal, SpaceshipTarget | DockingTarget)
        # if isinstance(init_sim_obs.goal, SpaceshipTarget):
        #     self.goal_state = init_sim_obs.goal.target
        #     if isinstance(init_sim_obs.goal, DockingTarget):
        #         self.goal_state.x = self.goal_state.x - init_sim_obs.goal.offset * np.cos(self.goal_state.psi)
        #         self.goal_state.y = self.goal_state.y - init_sim_obs.goal.offset * np.sin(self.goal_state.psi)
        self.planner = SpaceshipPlanner(
            planets=self.planets,
            satellites=self.satellites,
            sg=self.sg,
            sp=self.sp,
            goal_state=init_sim_obs.goal,
            init_state=self.init_state,
            scenario=init_sim_obs.dg_scenario,
        )
        self.goal = init_sim_obs.goal.target
        self.plotted = False
        self.X_error = np.zeros((8, 1))
        if init_sim_obs.goal.target.x == 8.5:
            self.test_case = 1
        if init_sim_obs.goal.target.x == 23.0:
            self.test_case = 2
        if init_sim_obs.goal.target.x == 8.0:
            self.test_case = 3
        self.plotted = False

        #
        # TODO: Implement Compute Initial Trajectory
        #

        self.cmds_plan, self.state_traj, self.A, self.B, self.F, self.r = self.planner.compute_trajectory(
            self.init_state, init_sim_obs.goal
        )
        # print(f"p: {self.cmds_plan.get_end()}")
        X_bar = np.zeros((8, 50))
        for i, v in enumerate(self.state_traj._values):
            for j in range(8):
                X_bar[j, i] = v.as_ndarray()[j].value

        num_steps = int(float(self.cmds_plan.get_end()) / 0.1) + 1  # number of 0.1s timesteps
        new_times = np.linspace(0, float(self.cmds_plan.get_end()), num_steps)
        original_times = np.linspace(0, float(self.cmds_plan.get_end()), 50)  # for your K=50 trajectory

        self.X_interp = np.zeros((8, num_steps))
        for i in range(8):
            self.X_interp[i, :] = np.interp(new_times, original_times, X_bar[i, :])

        self.N = 30  # Prediction horizon
        Q = np.eye(8)  # State tracking cost
        # R = 0.1 * np.eye(2)  #
        self.x_ref_window = cvx.Parameter((8, self.N + 1))
        self.x_current = cvx.Parameter(8)
        self.Ak = cvx.Parameter((self.N + 1, 64))
        self.Bk = cvx.Parameter((self.N + 1, 16))
        self.Fk = cvx.Parameter((self.N + 1, 8))
        self.rk = cvx.Parameter((self.N + 1, 8))
        self.x_ref_window.value = self.X_interp[:, 0 : self.N + 1]
        self.Ak.value = self.A[0 : self.N + 1, :]
        self.Bk.value = self.B[0 : self.N + 1, :]
        self.Fk.value = self.F[0 : self.N + 1, :]
        self.rk.value = self.r[0 : self.N + 1, :]

        self.x = cvx.Variable((8, self.N + 1))
        self.u = cvx.Variable((2, self.N))
        cost = 0
        constraints = []
        for i in range(self.N):

            cost += cvx.quad_form(self.x[:, i] - self.x_ref_window[:, i], Q)
            constraints += [
                self.x[:, i + 1]
                == self.Ak[i].reshape((8, 8), order="F") @ self.x[:, i]
                + self.Bk[i].reshape((8, 2), order="F") @ self.u[:, i]
                + self.rk[i]
                + self.Fk[i] * float(self.cmds_plan.get_end())
            ]
            constraints += [
                self.u[0, :] <= 2,
                self.u[0, :] >= -2,
                self.u[1, :] <= np.deg2rad(45),
                self.u[1, :] >= -np.deg2rad(45),
            ]
        constraints += [self.x[:, 0] == self.x_current]  # Initial condition
        cost += cvx.quad_form(self.x[:, self.N] - self.x_ref_window[:, self.N], Q)  # Terminal cost

        # Solve MPC
        self.prob = cvx.Problem(cvx.Minimize(cost), constraints)

    def get_commands(self, sim_obs: SimObservations) -> SpaceshipCommands:
        """
        This method is called by the simulator at every simulation time step. (0.1 sec)
        We suggest to perform two tasks here:
         - Track the computed trajectory (open or closed loop)
         - Plan a new trajectory if necessary
         (e.g., our tracking is deviating from the desired trajectory, the obstacles are moving, etc.)


        Do **not** modify the signature of this method.
        """
        current_state = sim_obs.players[self.myname].state
        next_desired_state = self.state_traj.at_interp(float(sim_obs.time) + 0.1)
        expected_state = self.state_traj.at_interp(sim_obs.time)

        next_desired_state_vec = [next_desired_state.as_ndarray()[i].value for i in range(8)]
        expected_state_vec = [expected_state.as_ndarray()[i].value for i in range(8)]

        self.X_error = np.hstack((self.X_error, (current_state.as_ndarray() - expected_state_vec).reshape(-1, 1)))
        # print(f"Error at time {sim_obs.time}: {self.X_error[0:2, -1]}")
        # if self.cmds_plan.get_end() - float(sim_obs.time) < 3 and not self.plotted:
        #     self.plotted = True
        #     print(f"Plotting case {self.test_case}")
        #     fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        #     axs[0].plot(self.X_error[0, :].T)
        #     axs[0].set_title("X error")
        #     axs[0].set_xlabel("timesteps")

        #     axs[1].plot(self.X_error[1, :].T)
        #     axs[1].set_title("Y error")
        #     axs[1].set_xlabel("timesteps")

        #     axs[2].plot(self.X_error[2, :].T)
        #     axs[2].set_title("Psi error")
        #     axs[2].set_xlabel("timesteps")

        #     plt.tight_layout()
        #     plt.savefig(f"src/pdm4ar/exercises/ex11/plots/case{self.test_case}-sim2real.png")
        #     plt.close()
        # if np.any(current_state.as_ndarray() - expected_state_vec > 0.01):
        #    print(f"Current - expected at time {sim_obs.time}: {current_state.as_ndarray() - expected_state_vec}")

        #
        # TODO: Implement scheme to replan
        #
        # if np.any(current_state.as_ndarray() - expected_state_vec > 0.1):
        #    self.cmds_plan, self.state_traj = self.planner.compute_trajectory(current_state, self.goal_state)
        # k = int(10 * (float(sim_obs.time) - self.cmds_plan.get_start()))
        # if k >= self.cmds_plan.get_end() / 0.1 - 1:
        #     return SpaceshipCommands.from_array(np.zeros((2)))
        # else:
        #     Ak = self.A[k].reshape((8, 8), order="F")
        #     Bk = self.B[k].reshape((8, 2), order="F")
        #     p = self.cmds_plan.get_end()
        #     cmds = np.linalg.pinv(Bk) @ (
        #         next_desired_state_vec - Ak @ current_state.as_ndarray() - self.r[k] - self.F[k] * p
        #     )

        #     cmds[0] = np.clip(cmds[0], -2, 2)
        #     cmds[1] = np.clip(cmds[1], -np.deg2rad(45), np.deg2rad(45))
        #     print(
        #         f"Command difference at {k}: thrust{cmds[0] - self.cmds_plan.at_interp(sim_obs.time).thrust} ddelta{cmds[1] - self.cmds_plan.at_interp(sim_obs.time).ddelta}"
        #     )

        #     # cmds = 0.5 * cmds + 0.5 * self.cmds_plan.at_interp(sim_obs.time).as_ndarray()
        #     # cmds = self.cmds_plan.at_interp(sim_obs.time)

        #     return SpaceshipCommands.from_array(cmds)
        use_mpc = True
        if use_mpc == False:
            return self.cmds_plan.at_interp(sim_obs.time)
        else:
            start_time = time.time()

            k = int(1 / 0.1 * (float(sim_obs.time) - float(self.cmds_plan.get_start())))

            # update parameters
            if k < self.A.shape[0] - self.N - 1:

                self.Ak.value = self.A[k : k + self.N + 1, :]
                self.Bk.value = self.B[k : k + self.N + 1, :]
                self.Fk.value = self.F[k : k + self.N + 1, :]
                self.rk.value = self.r[k : k + self.N + 1, :]
                self.x_ref_window.value = self.X_interp[:, k : k + self.N + 1]
                self.x_current.value = current_state.as_ndarray()  # Initial state

                self.prob.solve()
                solve_time = time.time() - start_time
                # print(
                #     f"Solve time: {solve_time}, cmds diff: {self.u.value[:, 0] - self.cmds_plan.at_interp(sim_obs.time).as_ndarray()}"
                # )
                return SpaceshipCommands.from_array(self.u.value[:, 0])
            else:
                return self.cmds_plan.at_interp(sim_obs.time)
