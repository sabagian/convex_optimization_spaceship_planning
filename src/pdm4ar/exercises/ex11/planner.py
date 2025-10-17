from calendar import c
from dataclasses import dataclass, field
from hmac import new
from os import error
from typing import Union
from sklearn.covariance import MinCovDet
import sympy as spy
import numpy as np

import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import (
    SpaceshipGeometry,
    SpaceshipParameters,
)
from dg_commons.sim.scenarios import DgScenario
from shapely.geometry import LineString

from pdm4ar.exercises.ex11.discretization import *

from pdm4ar.exercises_def.ex09 import goal
from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams
from pdm4ar.exercises_def.ex11.goal import SpaceshipTarget, DockingTarget

from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Set the backend to 'Agg' for non-GUI use


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "CLARABEL"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 100  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable weight
    weight_p: NDArray = field(default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time

    tr_radius: float = 5  # initial trust region radius
    min_tr_radius: float = 1e-4  # min trust region radius
    max_tr_radius: float = 100  # max trust region radius
    rho_0: float = 0.0  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.0  # div factor trust region update
    beta: float = 3.2  # mult factor trust region update

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    stop_crit: float = 1e-5  # Stopping criteria constant


class SpaceshipPlanner:
    """
    Feel free to change anything in this class.
    """

    planets: dict[PlayerName, PlanetParams]
    satellites: dict[PlayerName, SatelliteParams]
    spaceship: SpaceshipDyn
    sg: SpaceshipGeometry
    sp: SpaceshipParameters
    params: SolverParameters

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    old_objective: float
    new_objective: float

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        satellites: dict[PlayerName, SatelliteParams],
        sg: SpaceshipGeometry,
        sp: SpaceshipParameters,
        init_state: SpaceshipState,
        goal_state: SpaceshipTarget,
        scenario: DgScenario,
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.satellites = satellites
        print(self.satellites)
        self.sg = sg
        self.sp = sp
        self.scenario = scenario

        # Solver Parameters
        self.params = SolverParameters()

        # Spaceship Dynamics
        self.spaceship = SpaceshipDyn(self.sg, self.sp)
        self.n_x = self.spaceship.n_x
        self.n_u = self.spaceship.n_u
        self.n_p = self.spaceship.n_p
        self.n_x_obstacles: int = 3  # number of states used for obstacles avoidance

        # Discretization Method
        # self.integrator = ZeroOrderHold(self.Spaceship, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(self.spaceship, self.params.K, self.params.N_sub)

        # TODO DEBUG (Temporary goal and init state, used in convex)
        # self.goal_state = DynObstacleState(0, 0, 0, 0, 0, 0)
        # self.init_state = SpaceshipState(0, 0, 0, 0, 0, 0, 0, 2)

        self.init_state = init_state
        self.is_docking = False
        if isinstance(goal_state, DockingTarget):
            self.is_docking = True
            self.arms_length = goal_state.arms_length
            self.A, self.B, self.C, self.A1, self.A2, p = goal_state.get_landing_constraint_points()
            print(f"A {self.A1} a2 {self.A2}")
        self.goal_state = goal_state.target

        # print(f"goal state: {goal_state.x} {goal_state.y}")
        # print(f"n_x and n_x_obstacles: {self.n_x} {self.n_x_obstacles}")
        self.dist_init_goal = cvx.norm(
            self.init_state.x + self.init_state.y - self.goal_state.x - self.goal_state.y, "fro"
        )
        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()
        # assign trust region radius
        self.problem_parameters["radius_trust_region"].value = self.params.tr_radius

        self.initial_guess()  # update directly the problem parameters

        # Variables
        self.variables = self._get_variables()

        # Constraints
        constraints = self._get_constraints()

        # Objective
        objective = self._get_objective()

        # Cvx Optimisation Problem
        self.problem = cvx.Problem(objective, constraints)

    def compute_trajectory(
        self, init_state: SpaceshipState, goal_state: SpaceshipTarget
    ) -> tuple[
        DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState], NDArray, NDArray, NDArray, NDArray
    ]:
        """
        Compute a trajectory from init_state to goal_state.
        """
        # overwrite previous (temporary) init and goal state
        self.init_state = init_state
        self.is_docking = False
        if isinstance(goal_state, DockingTarget):
            self.is_docking = True
        self.goal_state = goal_state.target

        # Initialize state and control trajectories
        self.best_initial_guess()
        # used for plotting only
        # self.v_dyn = np.zeros((self.spaceship.n_x, self.params.K - 1))

        max_iterations = self.params.max_iterations

        for iteration in range(max_iterations):
            print(f"Iteration {iteration + 1}")
            self._convexification()
            # plot before solving
            # self.plot_predicted_and_real_results(iteration)

            # print the C matrix
            # print(self.problem_parameters["C_planets"].value)

            if iteration != 0:
                self.old_objective = self.problem.objective.value
            try:
                error = self.problem.solve(
                    verbose=self.params.verbose_solver, solver=self.params.solver, canon_backend=cvx.SCIPY_CANON_BACKEND
                )
                # print(f"Iteration {iteration + 1} error: {error}")
                print(f"Iteration {iteration + 1} objective value: {self.problem.objective.value}")
                self.new_objective = self.problem.objective.value
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")
            # Check for constraint violations
            # for constraint in self.problem.constraints:
            #    if constraint.dual_value is not None and np.any(constraint.dual_value < 0):
            #        print(f"Violated Constraint: {constraint}, Value: {constraint.dual_value}")
            # Check the solution status
            if self.problem.status == cvx.OPTIMAL:
                print("Optimal solution found.")
            elif self.problem.status == cvx.INFEASIBLE:
                print("Problem is infeasible. Check constraints or feasibility.")
            elif self.problem.status == cvx.UNBOUNDED:
                print("Problem is unbounded. Check if constraints are missing.")
            elif self.problem.status == cvx.OPTIMAL_INACCURATE:
                print(
                    "Solution is near-optimal but not accurate. Reason: Numerical instability, insufficient solver iterations, or precision issues.."
                )
            elif self.problem.status == cvx.INFEASIBLE_INACCURATE:
                print(
                    "Problem may be infeasible, but the solver could not confirm. Poor scaling, tight constraints, or solver limitations."
                )
            elif self.problem.status == cvx.UNBOUNDED_INACCURATE:
                print("Problem may be unbounded, but the solver could not confirm. Add bounds or regularization.")
            elif self.problem.status == cvx.USER_LIMIT:
                print("Solver stopped due to user-defined iteration or time limits. Increase the limits.")
            else:
                print("Solver did not converge. Check formulation and solver settings.")
            temp_X = self.variables["X"].value.copy()
            temp_p = self.variables["p"].value.copy()
            # if iteration != 0 and self._check_convergence2(temp_X, temp_p):
            #     print(f"p{temp_p}")
            #     print("Convergence reached.")
            #     break
            # compute flow_map
            flow_map_pre_solver = self.integrator.integrate_nonlinear_piecewise(
                self.problem_parameters["X_bar"].value,
                self.problem_parameters["U_bar"].value,
                self.problem_parameters["p_bar"].value,
            )
            delta_pre_solver = flow_map_pre_solver - self.problem_parameters["X_bar"].value
            objective_pre_solver_non_discretized = self._get_non_discretize_objective(delta_pre_solver)

            flow_map_post_solver = self.integrator.integrate_nonlinear_piecewise(
                self.variables["X"].value, self.variables["U"].value, self.variables["p"].value
            )
            delta_post_solver = flow_map_post_solver - self.variables["X"].value
            objective_post_solver_non_discretized = self._get_non_discretize_objective(delta_post_solver)
            if iteration != 0 and self._check_convergence(objective_pre_solver_non_discretized):
                print(f"p{temp_p}")
                print("Convergence reached.")
                break
            rho = (objective_pre_solver_non_discretized - objective_post_solver_non_discretized) / (
                objective_pre_solver_non_discretized - self.new_objective
            )
            print("RHO ", rho)
            if rho >= self.params.rho_0:
                self.problem_parameters["X_bar"].value = self.variables["X"].value
                self.problem_parameters["U_bar"].value = self.variables["U"].value
                self.problem_parameters["p_bar"].value = self.variables["p"].value

            # update trust region
            self._update_trust_region(rho)

            # used for debug only
            self.v_dyn = self.variables["v_s"].value
            # print(f"V {self.v_dyn}")

            # if iteration != 0 and self._check_convergence(objective_pre_solver_non_discretized):
            #    break

        # last plot
        # self.plot_predicted_and_real_results("final")

        # Example data: sequence from array
        mycmds, mystates = self._extract_seq_from_array(
            # tuple(range(self.params.K)),
            tuple(np.arange(self.params.K) * self.variables["p"].value / (self.params.K - 1)),
            self.problem_parameters["U_bar"].value[0],
            self.problem_parameters["U_bar"].value[1],
            self.problem_parameters["X_bar"].T,
        )
        p = self.problem_parameters["p_bar"].value[0]
        X_bar = self.problem_parameters["X_bar"].value
        U_bar = self.problem_parameters["U_bar"].value
        num_steps = int(p / 0.1) + 1  # number of 0.1s timesteps
        new_times = np.linspace(0, p, num_steps)

        # Create interpolated trajectories at 0.1s intervals
        # Assuming you have a time vector for your original trajectories
        original_times = np.linspace(0, p, 50)  # for your K=50 trajectory

        X_interp = np.zeros((X_bar.shape[0], num_steps))
        U_interp = np.zeros((U_bar.shape[0], num_steps))
        for i in range(X_bar.shape[0]):
            X_interp[i, :] = np.interp(new_times, original_times, X_bar[i, :])
        for i in range(U_bar.shape[0]):
            U_interp[i, :] = np.interp(new_times, original_times, U_bar[i, :])
        zoh_integrator = ZeroOrderHold(self.spaceship, num_steps, self.params.N_sub)
        Ak, Bk, Fk, rk = zoh_integrator.calculate_discretization(
            X_interp, U_interp, self.problem_parameters["p_bar"].value
        )
        return mycmds, mystates, Ak.T, Bk.T, Fk.T, rk.T

    def best_initial_guess(self):
        self.initial_guess()
        self._convexification()
        try:
            error = self.problem.solve(
                verbose=self.params.verbose_solver, solver=self.params.solver, canon_backend=cvx.SCIPY_CANON_BACKEND
            )
            print(f"Initial guess1 objective: {error}")
            obj_value = self.problem.objective.value

        except cvx.SolverError:
            print(f"SolverError: {self.params.solver} failed to solve the problem.")
            obj_value = 1000000000000000
        if self.problem.status == cvx.INFEASIBLE:
            obj_value = 1000000000000000
        min = obj_value
        best_X = self.variables["X"].value
        best_U = self.variables["U"].value
        best_p = self.variables["p"].value

        self.initial_guess2()
        self._convexification()
        try:
            error = self.problem.solve(
                verbose=self.params.verbose_solver, solver=self.params.solver, canon_backend=cvx.SCIPY_CANON_BACKEND
            )
            print(f"Initial guess2 objective: {error}")
            obj_value = self.problem.objective.value

        except cvx.SolverError:
            print(f"SolverError: {self.params.solver} failed to solve the problem.")
            obj_value = 1000000000000000
        if self.problem.status == cvx.INFEASIBLE:
            obj_value = 1000000000000000
        if obj_value < min:
            min = self.problem.objective.value
            best_X = self.variables["X"].value
            best_U = self.variables["U"].value
            best_p = self.variables["p"].value

        self.initial_guess3()
        self._convexification()
        try:
            error = self.problem.solve(
                verbose=self.params.verbose_solver, solver=self.params.solver, canon_backend=cvx.SCIPY_CANON_BACKEND
            )
            print(f"Initial guess3 error: {error}")
            obj_value = self.problem.objective.value

        except cvx.SolverError:
            print(f"SolverError: {self.params.solver} failed to solve the problem.")
            obj_value = 1000000000000000
        if self.problem.status == cvx.INFEASIBLE:
            obj_value = 1000000000000000
        if obj_value < min:
            min = self.problem.objective.value
            best_X = self.variables["X"].value
            best_U = self.variables["U"].value
            best_p = self.variables["p"].value

        self.problem_parameters["X_bar"].value = best_X
        self.problem_parameters["U_bar"].value = best_U
        self.problem_parameters["p_bar"].value = best_p

    def plot_predicted_and_real_results(self, iteration):
        x0 = self.problem_parameters["X_bar"].value[:, 0]
        integrated_X = self.integrator.integrate_nonlinear_full(
            x0, self.problem_parameters["U_bar"].value, self.problem_parameters["p_bar"].value
        )
        fig, axs = plt.subplots(6, 2, figsize=(20, 15))

        # Plot predicted position trajectory
        axs[0][0].plot(self.problem_parameters["X_bar"].value[0, :].T, self.problem_parameters["X_bar"].value[1, :].T)
        axs[0][0].set_title("Predicted Position")
        axs[0][0].set_xlabel("X")
        axs[0][0].set_ylabel("y")

        # Plot real position trajectory
        axs[0][1].plot(integrated_X[0, :].T, integrated_X[1, :].T)
        axs[0][1].set_title("Real Position")
        axs[0][1].set_xlabel("Step")
        axs[0][1].set_ylabel("Position")

        # Plot phi trajectory
        axs[1][0].plot(self.problem_parameters["X_bar"].value[2, :].T)
        axs[1][0].set_title("Predicted phi")
        axs[1][0].set_xlabel("Step")
        axs[1][0].set_ylabel("phi ")

        # Plot real phi trajectory
        axs[1][1].plot(integrated_X[2, :].T)
        axs[1][1].set_title("Real phi")
        axs[1][1].set_xlabel("Step")
        axs[1][1].set_ylabel("phi ")

        # Plot predicted velocities
        axs[2][0].plot(self.problem_parameters["X_bar"].value[3:5, :].T)
        axs[2][0].set_title("Predicted Velocities")
        axs[2][0].set_xlabel("Step")
        axs[2][0].set_ylabel("Predicted Velocities")
        axs[2][0].legend(["VX", "VY"])

        # Plot real velocities
        axs[2][1].plot(integrated_X[3:5, :].T)
        axs[2][1].set_title("Real Velocities")
        axs[2][1].set_xlabel("Step")
        axs[2][1].set_ylabel("Real Velocities")
        axs[2][1].legend(["VX", "VY"])

        # Plot U
        axs[3][0].plot(self.problem_parameters["U_bar"].value.T)
        axs[3][0].set_title(f"U, p: {self.problem_parameters['p_bar'].value}")
        axs[3][0].set_xlabel("Step")
        axs[3][0].set_ylabel("U")

        # Plot vdyn
        axs[3][1].plot(self.v_dyn.T)
        axs[3][1].set_title("VDYN")
        axs[3][1].set_xlabel("Step")
        axs[3][1].set_ylabel("VDYN")
        axs[3][1].legend(["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"])

        # Plot predicted mass
        axs[4][0].plot(self.problem_parameters["X_bar"].value[7, :].T)
        axs[4][0].set_title("Predicted Mass")
        axs[4][0].set_xlabel("Step")
        axs[4][0].set_ylabel("Mass")

        # Plot real mass
        axs[4][1].plot(integrated_X[7, :].T)
        axs[4][1].set_title("Real Mass")
        axs[4][1].set_xlabel("Step")
        axs[4][1].set_ylabel("Mass")

        # plot v_final_pose
        axs[5][0].plot(self.variables["v_final_pose"].value.T)
        axs[5][0].set_title("v_final_pose")
        axs[5][0].set_xlabel("Step")
        axs[5][0].set_ylabel("v_final_pose")
        axs[5][0].legend(["V1", "V2", "V3", "V4"])

        if self.init_state.as_ndarray()[0] == -9.0:
            test_case = "1"
        elif self.goal_state.as_ndarray()[0] == 8.0:
            test_case = "2"
        elif self.goal_state.as_ndarray()[0] == 23.0:
            test_case = "3"
        else:
            test_case = "boh"

        plt.tight_layout()
        plt.savefig(f"src/pdm4ar/exercises/ex11/plots/case{test_case}-it{iteration}.png")
        plt.close()

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        returns X_bar, U_bar, p_bar initial guess for SCvx.
        """
        # TODO
        K = self.params.K

        X = np.zeros((self.spaceship.n_x, K))
        X[0, :] = np.arange(self.params.K) * (self.goal_state.x - self.init_state.x) / (self.params.K - 1)
        X[1, :] = np.arange(self.params.K) * (self.goal_state.y - self.init_state.y) / (self.params.K - 1)
        X[0, :] += self.init_state.x
        X[1, :] += self.init_state.y
        # X[0, :] = self.problem_parameters["init_state"][0]
        # X[1, :] = self.problem_parameters["init_state"][1]
        # X[7, :] = self.problem_parameters["init_state"][7]
        X[7, :] = self.sg.m

        U = np.zeros((self.spaceship.n_u, K))
        p = np.ones((self.spaceship.n_p))

        # update problem parameters
        self.problem_parameters["X_bar"].value = X
        self.problem_parameters["U_bar"].value = U
        self.problem_parameters["p_bar"].value = p

    def initial_guess2(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        returns X_bar, U_bar, p_bar initial guess for SCvx.
        """
        # TODO
        K = self.params.K

        X = np.zeros((self.spaceship.n_x, K))
        X[0, 0:24] = np.arange(24) * (self.goal_state.x - self.init_state.x) / (24 - 1)
        X[0, 24:50] = self.goal_state.x - self.init_state.x
        print(np.arange(25) * (self.goal_state.y - self.init_state.y) / (25 - 1))
        X[1, 25:50] = np.arange(25) * (self.goal_state.y - self.init_state.y) / (25 - 1)
        X[0, :] += self.init_state.x
        X[1, :] += self.init_state.y
        # X[0, :] = self.problem_parameters["init_state"][0]
        # X[1, :] = self.problem_parameters["init_state"][1]
        # X[7, :] = self.problem_parameters["init_state"][7]
        X[7, :] = self.sg.m

        U = np.zeros((self.spaceship.n_u, K))
        p = np.ones((self.spaceship.n_p))

        # update problem parameters
        self.problem_parameters["X_bar"].value = X
        self.problem_parameters["U_bar"].value = U
        self.problem_parameters["p_bar"].value = p

    def initial_guess3(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        returns X_bar, U_bar, p_bar initial guess for SCvx.
        """
        # TODO
        K = self.params.K

        X = np.zeros((self.spaceship.n_x, K))
        X[1, 0:24] = np.arange(24) * (self.goal_state.y - self.init_state.y) / (24 - 1)
        X[1, 24:50] = self.goal_state.y - self.init_state.y
        X[0, 25:50] = np.arange(25) * (self.goal_state.x - self.init_state.x) / (25 - 1)
        X[0, :] += self.init_state.x
        X[1, :] += self.init_state.y
        # X[0, :] = self.problem_parameters["init_state"][0]
        # X[1, :] = self.problem_parameters["init_state"][1]
        # X[7, :] = self.problem_parameters["init_state"][7]
        X[7, :] = self.sg.m

        U = np.zeros((self.spaceship.n_u, K))
        p = np.ones((self.spaceship.n_p))

        # update problem parameters
        self.problem_parameters["X_bar"].value = X
        self.problem_parameters["U_bar"].value = U
        self.problem_parameters["p_bar"].value = p

    def _set_goal(self):
        """
        Sets goal for SCvx.
        """
        # TODO what is (6,1)?
        # self.goal = cvx.Parameter((6, 1))
        pass

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        variables = {
            "X": cvx.Variable((self.spaceship.n_x, self.params.K), name="X"),
            "U": cvx.Variable((self.spaceship.n_u, self.params.K), name="U"),  # 0: thrust, 1: ddelta
            "p": cvx.Variable(self.spaceship.n_p, name="p"),  # final time
            "v_dyn": cvx.Variable((self.spaceship.n_x, self.params.K - 1), name="v_dyn"),
            "v_init_state": cvx.Variable(self.spaceship.n_x, name="v_init_state"),
            "v_goal_config": cvx.Variable(self.spaceship.n_x - 2, name="v_goal_config"),
            "v_s": cvx.Variable(3 * (len(self.planets) + len(self.satellites)) + 1, name="v_s"),
            "v_final_pose": cvx.Variable(4, name="v_final_pose"),
            "v_cone": cvx.Variable(1, name="v_cone"),
            "v_boundaries": cvx.Variable(1, name="v_boundaries"),
            # "v_goal_coords": cvx.Variable(name="v_goal_coords"),
            # "v_goal_pose": cvx.Variable(name="v_goal_pose"),
            # "v_goal_vel": cvx.Variable(name="v_goal_vel"),
        }
        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        problem_parameters = {
            "init_state": cvx.Parameter(self.spaceship.n_x, name="init_state"),
            "goal_config": cvx.Parameter(6, name="goal_config"),
            "A_bar": cvx.Parameter((self.n_x * self.n_x, self.params.K - 1), name="A_bar"),  # shape 64 x 49
            "B_plus_bar": cvx.Parameter((self.n_x * self.n_u, self.params.K - 1), name="B_plus_bar"),  # shape 16 x 49
            "B_minus_bar": cvx.Parameter((self.n_x * self.n_u, self.params.K - 1), name="B_minus_bar"),  # shape 16 x 49
            "F_bar": cvx.Parameter((self.n_x * self.n_p, self.params.K - 1), name="F_bar"),  # shape 8 x 49
            "r_bar": cvx.Parameter((self.n_x, self.params.K - 1), name="r_bar"),  # shape 8 x 49
            "radius_trust_region": cvx.Parameter(name="radius_trust_region"),
            "X_bar": cvx.Parameter((self.n_x, self.params.K), name="X_bar"),
            "U_bar": cvx.Parameter((self.n_u, self.params.K), name="U_bar"),
            "p_bar": cvx.Parameter(self.n_p, name="p_bar"),
            "deltax": cvx.Parameter((self.n_x, self.params.K), name="deltax"),
            "deltau": cvx.Parameter((self.n_u, self.params.K), name="deltau"),
            "deltap": cvx.Parameter(self.n_p, name="deltap"),
            "C_planets": cvx.Parameter(
                ((3 * (len(self.planets) + len(self.satellites)) + 1) * self.n_x_obstacles, self.params.K),
                name="C_planets",
            ),
            "r_planets": cvx.Parameter(
                (3 * (len(self.planets) + len(self.satellites)) + 1, self.params.K), name="r_planets"
            ),
            "G_planets": cvx.Parameter(
                (3 * (len(self.planets) + len(self.satellites)) + 1, self.params.K), name="G_planets"
            ),
            # "C_times_X": cvx.Parameter((len(self.planets) * 3, self.params.K), name="C_times_X"),
            # "S_matrix": cvx.Parameter((len(self.planets) * 3, self.params.K), name="S_matrix"),
        }

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """

        constraints = []

        self._convexification()
        A = self.problem_parameters["A_bar"].T
        B_plus = self.problem_parameters["B_plus_bar"].T
        B_minus = self.problem_parameters["B_minus_bar"].T
        F = self.problem_parameters["F_bar"].T
        r = self.problem_parameters["r_bar"].T
        # A[0].reshape((self.n_x, self.n_x), order="F")

        constraints = [
            # Initial state costraint (WAS ALREADY IN THE EXAMPLE)
            self.variables["X"][:, 0] - self.problem_parameters["init_state"] - self.variables["v_init_state"] == 0,
            # Final state constraint
            self.variables["X"][0:6, self.params.K - 1]
            - self.problem_parameters["goal_config"]
            - self.variables["v_goal_config"]
            == 0,
            # Input constraints
            self.variables["U"][:, 0] - np.zeros(self.spaceship.n_u) == 0,
            self.variables["U"][:, self.params.K - 1] - np.zeros(self.spaceship.n_u) == 0,
            # Needs to be close to the goal
            # cvx.norm(goal_coords - fin_coords, "fro") - self.variables["v_goal_coords"] <= 0,
            # # Orientation constraint
            # pose_fin - pose_goal - self.variables["v_goal_pose"] == 0,
            # # Specified velocity constraint
            # cvx.norm(v_fin - v_goal, "fro") - self.variables["v_goal_vel"] <= 0,
            # Mass constraint
            self.variables["X"][7, :] - self.sg.m >= 0,
            # Thrust constraint
            self.variables["U"][0, :] - self.sp.thrust_limits[0] >= 0,
            self.variables["U"][0, :] - self.sp.thrust_limits[1] <= 0,
            # Thruster angle costraint
            self.variables["X"][6, :] - self.sp.delta_limits[0] >= 0,
            self.variables["X"][6, :] - self.sp.delta_limits[1] <= 0,
            # Maximum time
            # TODO
            # Rate of change
            self.variables["U"][1, :] - self.sp.ddelta_limits[0] >= 0,
            self.variables["U"][1, :] - self.sp.ddelta_limits[1] <= 0,
            self.variables["p"] >= 0,
            # add constraints for trust region
            cvx.norm(self.problem_parameters["deltax"], "fro")
            + cvx.norm(self.problem_parameters["deltau"], "fro")
            + cvx.norm(self.problem_parameters["deltap"], "fro")
            - self.problem_parameters["radius_trust_region"]
            <= 0,
        ]

        # if self.is_docking:
        #     intermediate_pose = self.problem_parameters["goal_config"].value
        #     intermediate_pose[0] = intermediate_pose[0] + self.arms_length / 2 * np.cos(intermediate_pose[2])
        #     intermediate_pose[1] = intermediate_pose[1] + self.arms_length / 2 * np.sin(intermediate_pose[2])
        #     constraints += [
        #         self.variables["X"][0:6, self.params.K - 2] - intermediate_pose == 0,
        #     ]
        # Dynamic constraints
        # TODO vettorizzare usando cvx.matmul se si riesce, ma si lamenta perchè non sa gestire 3 dimensioni
        constraints += [
            self.variables["X"][:, i]
            == cvx.matmul(A[i - 1].reshape((self.n_x, self.n_x), order="F"), self.variables["X"][:, i - 1].T)
            + cvx.matmul(B_plus[i - 1].reshape((self.n_x, self.n_u), order="F"), self.variables["U"][:, i].T)
            + cvx.matmul(B_minus[i - 1].reshape((self.n_x, self.n_u), order="F"), self.variables["U"][:, i - 1].T)
            + cvx.matmul(F[i - 1].reshape((self.n_x, self.n_p), order="F"), self.variables["p"].T)
            + r[i - 1]
            + self.variables["v_dyn"][:, i - 1]
            for i in range(1, self.params.K)
        ]

        # obstacle avoidance of planets
        for k in range(self.params.K):
            m = self.problem_parameters["C_planets"][:, k].reshape(
                (3 * (len(self.planets) + len(self.satellites)) + 1, self.n_x_obstacles), order="C"
            )
            c = (
                cvx.matmul(m, self.variables["X"][: self.n_x_obstacles, k])
                + self.problem_parameters["r_planets"][:, k]
                + self.problem_parameters["G_planets"][:, k] * self.variables["p"]
                - self.variables["v_s"]
                <= 0
            )
            constraints += [c]
        boundaries = self.scenario.static_obstacles
        for b in boundaries:
            if isinstance(b.shape, LineString):
                self.minx, self.miny, self.maxx, self.maxy = b.shape.bounds

        constraints += [
            self.variables["X"][0, :] - self.minx - (self.sg.l_f + self.sg.l_c + 0.2 + self.variables["v_boundaries"])
            >= 0,
            self.variables["X"][0, :] - self.maxx + (self.sg.l_f + self.sg.l_c + 0.2 + self.variables["v_boundaries"])
            <= 0,
            self.variables["X"][1, :] - self.miny - (self.sg.l_f + self.sg.l_c + 0.2 + self.variables["v_boundaries"])
            >= 0,
            self.variables["X"][1, :] - self.maxy + (self.sg.l_f + self.sg.l_c + 0.2 + self.variables["v_boundaries"])
            <= 0,
        ]

        if self.is_docking:
            By = self.B[1]
            Bx = self.B[0]
            Cy = self.C[1]
            Cx = self.C[0]
            xA = self.A[0]
            yA = self.A[1]
            mB = (By - yA) / (Bx - xA)
            bB = yA - mB * xA
            mC = (Cy - yA) / (Cx - xA)
            bC = yA - mC * xA
            if By - mC * Bx - bC > 0:
                constraints += [
                    self.variables["X"][1, self.params.K - 10 :]
                    - mC * self.variables["X"][0, self.params.K - 10 :]
                    - bC
                    >= self.variables["v_cone"]
                ]
                print("y - ", mC, "x - ", bC, " > =0")
            else:
                constraints += [
                    self.variables["X"][1, self.params.K - 10 :]
                    - mC * self.variables["X"][0, self.params.K - 10 :]
                    - bC
                    <= -self.variables["v_cone"]
                ]
                print("y - ", mC, "x - ", bC, " < =0")
            if Cy - mB * Cx - bB > 0:
                constraints += [
                    self.variables["X"][1, self.params.K - 10 :]
                    - mB * self.variables["X"][0, self.params.K - 10 :]
                    - bB
                    >= self.variables["v_cone"]
                ]
                print("y - ", mB, "x - ", bB, " > =0")
            else:
                constraints += [
                    self.variables["X"][1, self.params.K - 10 :]
                    - mB * self.variables["X"][0, self.params.K - 10 :]
                    - bB
                    <= -self.variables["v_cone"]
                ]
                print("y - ", mB, "x - ", bB, " < =0")
        return constraints

    def _jacobians_obstacles(self):
        """
        Convexify obstacles for SCvx.
        """
        x = spy.Matrix(spy.symbols("x y psi", real=True))
        p = spy.symbols("p", real=True)
        k = spy.symbols("k", integer=True)
        # array of C, D, G matrixes and r vectors

        # TODO: change the following dimensions after adding satellites
        s_function = spy.zeros(3 * (len(self.planets) + len(self.satellites)) + 1, 1)

        # rectangular part
        d = (self.sg.l_r + self.sg.l_f) / 2
        r = spy.sqrt(d**2 + self.sg.w_half**2)
        cx = x[0] + (d - self.sg.l_r) * spy.cos(x[2])
        cy = x[1] + (d - self.sg.l_r) * spy.sin(x[2])
        r = r * 1.7

        # triangular part
        a = spy.sqrt(self.sg.w_half**2 + self.sg.l_c**2)
        r2 = (a**2 * self.sg.w_half * 2) / (4 * self.sg.l_c * self.sg.w_half)
        cx2 = x[0] + (self.sg.l_f + self.sg.l_c - r2) * spy.cos(x[2])
        cy2 = x[1] + (self.sg.l_f + self.sg.l_c - r2) * spy.sin(x[2])
        r2 = r2 * 1.7

        # thrust part
        cx3 = x[0] - (self.sg.l_r) * spy.cos(x[2])
        cy3 = x[1] - (self.sg.l_r) * spy.sin(x[2])
        r3 = self.sg.l_t_half
        r3 = r3 * 1.7

        i = 0
        for planet in self.planets.values():
            # to check collision with each obtacle we create 3 circles around the different parts
            # of the spaceship

            s_function[i] = (planet.radius + r) ** 2 - ((cx - planet.center[0]) ** 2 + (cy - planet.center[1]) ** 2)
            i += 1

            s_function[i] = (planet.radius + r2) ** 2 - ((cx2 - planet.center[0]) ** 2 + (cy2 - planet.center[1]) ** 2)
            i += 1

            s_function[i] = (planet.radius + r3) ** 2 - ((cx3 - planet.center[0]) ** 2 + (cy3 - planet.center[1]) ** 2)
            i += 1

            # s_function[i] = (
            #     (planet.radius + 2 * self.sg.l_r) ** 2 - (x[0] - planet.center[0]) ** 2 - (x[1] - planet.center[1]) ** 2
            # )
            # i += 1

        for planet_key, satellite in self.satellites.items():
            planet = self.planets[planet_key.split("/")[0]]
            x_satellite = planet.center[0] + satellite.orbit_r * spy.cos(
                satellite.omega * k * p / (self.params.K - 1) + satellite.tau
            )
            y_satellite = planet.center[1] + satellite.orbit_r * spy.sin(
                satellite.omega * k * p / (self.params.K - 1) + satellite.tau
            )
            # s_function[i] = (
            #     (satellite.radius + 2 * self.sg.l_r) ** 2 - (x[0] - x_satellite) ** 2 - (x[1] - y_satellite) ** 2
            # )
            # obstacle avoidance of the rectangular part
            s_function[i] = (satellite.radius + r) ** 2 - ((cx - x_satellite) ** 2 + (cy - y_satellite) ** 2)
            i += 1
            # obstacle avoidance of the triangular part
            s_function[i] = (satellite.radius + r2) ** 2 - ((cx2 - x_satellite) ** 2 + (cy2 - y_satellite) ** 2)
            i += 1
            # obstacle avoidance of the thrust part
            s_function[i] = (satellite.radius + r3) ** 2 - ((cx3 - x_satellite) ** 2 + (cy3 - y_satellite) ** 2)
            i += 1
        if self.is_docking:
            # circle
            # dist_base = np.linalg.norm(self.A1 - self.A2)
            # center_x = (self.A1[0] + self.A2[0]) / 2 - np.cos(
            #     self.problem_parameters["goal_config"][2].value
            # ) * 2 * dist_base
            # center_y = (self.A1[1] + self.A2[1]) / 2 - np.sin(
            #     self.problem_parameters["goal_config"][2].value
            # ) * 2 * dist_base
            # # radius = np.sqrt((self.A1[0] - center_x) ** 2 + (self.A1[1] - center_y) ** 2)
            # radius = 2 * dist_base
            # s_function[i] = (radius) ** 2 - (x[0] - center_x) ** 2 - (x[1] - center_y) ** 2

            # ellipse
            dist_base = np.linalg.norm(self.A1 - self.A2)
            theta = self.problem_parameters["goal_config"][2].value + np.pi / 2
            center_x = (self.A1[0] + self.A2[0]) / 2
            center_y = (self.A1[1] + self.A2[1]) / 2
            a = dist_base / 2 + self.sg.l_f + self.sg.l_c  # semiasse maggiore
            b = 0.3  # semiasse minore
            s_function[i] = (
                -(((x[0] - center_x) * np.cos(theta) + (x[1] - center_y) * np.sin(theta)) ** 2) / (a**2)
                - (((x[0] - center_x) * np.sin(theta) - (x[1] - center_y) * np.cos(theta)) ** 2) / (b**2)
                + 1
            )
        else:
            s_function[i] = 0
        # jacobian of the obstacle function
        C = s_function.jacobian(x)
        G = s_function.diff(p)
        # print shape of C
        print(f"Shape of C: {C.shape}")
        # print number of planets
        print(f"Number of planets: {len(self.planets)}")
        print(f"Number of satellites: {len(self.satellites)}")
        s_func = spy.lambdify((x, k, p), s_function, "numpy")
        C_func = spy.lambdify((x, k, p), C, "numpy")
        G_fun = spy.lambdify((x, k, p), G, "numpy")

        return s_func, C_func, G_fun

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # TODO
        objective = 0
        initial_pose = self.problem_parameters["goal_config"][2]
        # for i in range(self.params.K - 1):
        #     objective += cvx.norm(
        #         0.5
        #         / self.params.K
        #         * (
        #             self.variables["X"][0:3, i]
        #             + self.variables["X"][0:3, i + 1]
        #             - 2 * self.problem_parameters["goal_config"][0:3]
        #         ),
        #         "fro",
        #     )
        #     objective += cvx.norm(
        #         10 * 0.5 / self.params.K * (self.variables["X"][3:5, i] + self.variables["X"][3:5, i + 1]),
        #         "fro",
        #     )

        #     # objective += cvx.sum(
        #     #     10 * 0.5 / self.params.K * (self.variables["X"][2, i] - self.variables["X"][2, i + 1]) ** 2
        #     # )
        #     # objective += cvx.norm(
        #     #     0.5
        #     #     / self.params.K
        #     #     * (
        #     #         self.variables["X"][2, i]
        #     #         + self.variables["X"][2, i + 1]
        #     #         - 2 * self.problem_parameters["goal_config"][2]
        #     #     ),
        #     #     "fro",
        #     # )
        #     objective += cvx.sum(
        #         100 * 0.5 / self.params.K * (self.variables["U"][:, i] - self.variables["U"][:, i + 1]) ** 2
        #     )

        # Slices
        X = self.variables["X"]
        U = self.variables["U"]
        goal_config = self.problem_parameters["goal_config"]

        # Constants
        K = self.params.K
        alpha = 0.5 / K

        # Compute the terms for the cost function
        X_mid = X[:, :-1] + X[:, 1:]  # Midpoints of X
        goal_term = X_mid[0:3, :] - 2 * cvx.reshape(goal_config[0:3], (3, 1))
        control_diff = U[:, :-1] - U[:, 1:]  # Difference in control inputs
        position_differences = X[0:2, 1:] - X[0:2, :-1]
        distance_traveled = cvx.norm(position_differences, axis=0, p=2)
        actuation_effort = cvx.norm(U, axis=0, p=1)

        # Vectorized norms and summations
        objective = cvx.sum(alpha * cvx.norm(goal_term, axis=0, p="fro"))
        # objective += cvx.sum(10 * alpha * cvx.norm(X_mid[3:5, :], axis=0, p="fro"))
        objective += cvx.sum(10 * alpha * cvx.square(control_diff))
        objective += cvx.sum(distance_traveled)
        # objective += cvx.sum(actuation_effort)
        objective += 0.4 * self.variables["p"]

        objective += (
            1000 * cvx.sum(cvx.abs(1 / self.params.K * self.variables["v_dyn"]))
            + 1000 * cvx.sum(cvx.abs(self.variables["v_init_state"]))
            + 1000 * cvx.sum(cvx.abs(self.variables["v_goal_config"]))
            + 1000 * cvx.sum(cvx.abs(self.variables["v_s"]))
            + 1000 * cvx.sum(cvx.abs(self.variables["v_boundaries"]))
            + 100 * cvx.sum(cvx.abs(self.variables["v_cone"]))
            # + 100 * cvx.sum(cvx.abs(self.variables["v_final_pose"]))
        )
        # add the final time
        return cvx.Minimize(objective)

    def _get_non_discretize_objective(self, delta):
        objective = 0
        initial_pose = self.problem_parameters["goal_config"][2]
        # for i in range(self.params.K - 1):
        #     objective += cvx.norm(
        #         0.5
        #         / self.params.K
        #         * (
        #             self.variables["X"][0:3, i]
        #             + self.variables["X"][0:3, i + 1]
        #             - 2 * self.problem_parameters["goal_config"][0:3]
        #         ),
        #         "fro",
        #     )
        #     objective += cvx.norm(
        #         10 * 0.5 / self.params.K * (self.variables["X"][3:5, i] + self.variables["X"][3:5, i + 1]),
        #         "fro",
        #     )

        #     # objective += cvx.sum(
        #     #     10 * 0.5 / self.params.K * (self.variables["X"][2, i] - self.variables["X"][2, i + 1]) ** 2
        #     # )
        #     # objective += cvx.norm(
        #     #     0.5
        #     #     / self.params.K
        #     #     * (
        #     #         self.variables["X"][2, i]
        #     #         + self.variables["X"][2, i + 1]
        #     #         - 2 * self.problem_parameters["goal_config"][2]
        #     #     ),
        #     #     "fro",
        #     # )
        #     objective += cvx.sum(
        #         100 * 0.5 / self.params.K * (self.variables["U"][:, i] - self.variables["U"][:, i + 1]) ** 2
        #     )
        X = self.variables["X"]
        U = self.variables["U"]
        goal_config = self.problem_parameters["goal_config"]

        # Constants
        K = self.params.K
        alpha = 0.5 / K

        # Compute the terms for the cost function
        X_mid = X[:, :-1] + X[:, 1:]  # Midpoints of X
        goal_term = X_mid[0:3, :] - 2 * cvx.reshape(goal_config[0:3], (3, 1))
        control_diff = U[:, :-1] - U[:, 1:]  # Difference in control inputs
        position_differences = X[0:2, 1:] - X[0:2, :-1]

        distance_traveled = cvx.norm(position_differences, axis=0, p=2)
        actuation_effort = cvx.norm(U, axis=0, p=1)

        # Vectorized norms and summations
        objective = cvx.sum(alpha * cvx.norm(goal_term, axis=0, p="fro"))
        # objective += cvx.sum(10 * alpha * cvx.norm(X_mid[3:5, :], axis=0, p="fro"))
        objective += cvx.sum(10 * alpha * cvx.square(control_diff))
        # objective += cvx.sum(actuation_effort)
        objective += cvx.sum(distance_traveled)
        objective += 0.4 * self.variables["p"]
        # print(f"integrated: {objective.value}")
        objective += 1000 * cvx.sum(cvx.abs(1 / self.params.K * delta))
        # print(f"plus delta: {objective.value}")
        objective += 1000 * cvx.sum(cvx.abs(self.variables["X"][:, 0] - self.problem_parameters["init_state"]))
        # print(f"plus init: {objective.value}")
        objective += 1000 * cvx.sum(
            cvx.abs(self.variables["X"][0:6, self.params.K - 1] - self.problem_parameters["goal_config"])
        )
        # print(f"plus goal: {objective.value}, v_config: {self.variables['v_goal_config'].value}")
        # objective += 1000 * cvx.sum(cvx.abs(self.variables["X"][0:6, self.params.K - 2] - self.intermediate_pose))
        # final pose
        # objective += 10 * cvx.sum(
        #     cvx.abs(
        #         self.variables["X"][2][self.params.K - 5 : self.params.K - 1]
        #         - self.problem_parameters["goal_config"][2] * np.ones(4)
        #     )
        # )

        # for k in range(self.params.K):
        #     m = self.problem_parameters["C_planets"][:, k].reshape(
        #         (len(self.planets) * 3, self.n_x_obstacles), order="C"
        #     )
        #     c = cvx.matmul(m, self.variables["X"][: self.n_x_obstacles, k]) + self.problem_parameters["r_planets"][:, k]
        s_matrix = np.zeros((3 * (len(self.planets) + len(self.satellites)) + 1, self.params.K))

        for k in range(self.params.K):
            x = self.variables["X"][: self.n_x_obstacles, k].value
            x = np.array(x)
            s_matrix[:, k] = self.s_function(x, k, self.variables["p"][0].value).flatten()
        max_s = np.max(s_matrix, 1)
        objective += 1000 * np.sum(max_s[max_s > 0])
        # print(f"plus obstacle: {objective.value}")
        bound = np.array(
            [
                self.variables["X"][0, :].value - self.minx - (self.sg.l_f + self.sg.l_c),
                -self.variables["X"][0, :].value + self.maxx - (self.sg.l_f + self.sg.l_c),
                self.variables["X"][1, :].value - self.miny - (self.sg.l_f + self.sg.l_c),
                -self.variables["X"][1, :].value + self.maxy - (self.sg.l_f + self.sg.l_c),
            ]
        )
        bound = np.concatenate(bound)
        bound_min = np.min(bound)
        if bound_min < 0:
            objective += 1000 * np.abs(bound_min)

        if self.is_docking:
            cone = []
            By = self.B[1]
            Bx = self.B[0]
            Cy = self.C[1]
            Cx = self.C[0]
            xA = self.A[0]
            yA = self.A[1]
            mB = (By - yA) / (Bx - xA)
            bB = yA - mB * xA
            mC = (Cy - yA) / (Cx - xA)
            bC = yA - mC * xA
            if By - mC * Bx - bC > 0:
                cone += [
                    self.variables["X"][1, self.params.K - 10 :].value
                    - mC * self.variables["X"][0, self.params.K - 10 :].value
                    - bC
                    >= 0
                ]
            else:
                cone += [
                    -self.variables["X"][1, self.params.K - 10 :].value
                    + mC * self.variables["X"][0, self.params.K - 10 :].value
                    + bC
                    >= 0
                ]
            if Cy - mB * Cx - bB > 0:
                cone += [
                    self.variables["X"][1, self.params.K - 10 :].value
                    - mB * self.variables["X"][0, self.params.K - 10 :].value
                    - bB
                    >= 0
                ]
            else:
                cone += [
                    -self.variables["X"][1, self.params.K - 10 :].value
                    + mB * self.variables["X"][0, self.params.K - 10 :].value
                    + bB
                    >= 0
                ]
            cone = np.array(cone)
            cone = np.concatenate(cone)
            cone_min = np.min(cone)
            if cone_min < 0:
                objective += 100 * np.abs(cone_min)

        return objective.value

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        Returns A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar from discretization around X_bar, U_bar, p.
        """
        # ZOH
        # A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
        # FOH
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(
            self.problem_parameters["X_bar"].value,
            self.problem_parameters["U_bar"].value,
            self.problem_parameters["p_bar"].value,
        )

        # tempA = A_bar[:, 0].reshape((8, 8), order="F")

        self.problem_parameters["init_state"].value = self.init_state.as_ndarray()
        self.problem_parameters["goal_config"].value = self.goal_state.as_ndarray()
        # TODO populate other problem parameters + function is not modifying anything
        self.problem_parameters["A_bar"].value = A_bar  # .T.reshape((-1, self.n_x, self.n_x), order="F")
        self.problem_parameters["B_plus_bar"].value = B_plus_bar  # .T.reshape((-1, self.n_x, self.n_u), order="F")
        self.problem_parameters["B_minus_bar"].value = B_minus_bar  # .T.reshape((-1, self.n_x, self.n_u), order="F")
        self.problem_parameters["F_bar"].value = F_bar  # .T.reshape((-1, self.n_x, self.n_p), order="F")
        self.problem_parameters["r_bar"].value = r_bar  # .T.reshape((-1, self.n_x), order="F")

        # get matrix C for obstacle avoidance
        self.s_function, C, G = self._jacobians_obstacles()
        # construct the C matrix and s_matrix calling C and s_function at each time step between 0 and K
        C_matrix = np.zeros(((3 * (len(self.planets) + len(self.satellites)) + 1) * self.n_x_obstacles, self.params.K))
        s_matrix = np.zeros((3 * (len(self.planets) + len(self.satellites)) + 1, self.params.K))
        G_matrix = np.zeros((3 * (len(self.planets) + len(self.satellites)) + 1, self.params.K))
        for k in range(self.params.K):
            x = self.problem_parameters["X_bar"].value[: self.n_x_obstacles, k]
            # reshape x to be a numpy array
            x = np.array(x)
            # print(self.s_function(x, k, self.problem_parameters["p_bar"][0].value))
            # print(self.s_function(x, k, self.problem_parameters["p_bar"][0].value).shape)
            s_matrix[:, k] = self.s_function(x, k, self.problem_parameters["p_bar"][0].value).reshape(-1).flatten()
            C_matrix[:, k] = C(x, k, self.problem_parameters["p_bar"][0].value).flatten()
            G_matrix[:, k] = G(x, k, self.problem_parameters["p_bar"][0].value).flatten()
        # self.problem_parameters["S_matrix"].value = s_matrix
        self.problem_parameters["C_planets"].value = C_matrix
        self.problem_parameters["G_planets"].value = G_matrix
        # compute C_matrix * X_bar
        C_times_X = np.zeros((3 * (len(self.planets) + len(self.satellites)) + 1, self.params.K))
        for k in range(self.params.K):
            C_times_X[:, k] = np.matmul(
                C_matrix[:, k].reshape((3 * (len(self.planets) + len(self.satellites)) + 1, self.n_x_obstacles)),
                self.problem_parameters["X_bar"].value[: self.n_x_obstacles, k],
            )
        # self.problem_parameters["C_times_X"].value = C_times_X
        self.problem_parameters["r_planets"].value = (
            s_matrix - C_times_X - G_matrix * self.problem_parameters["p_bar"].value
        )

        self.problem_parameters["deltax"] = self.variables["X"] - self.problem_parameters["X_bar"]
        self.problem_parameters["deltau"] = self.variables["U"] - self.problem_parameters["U_bar"]
        self.problem_parameters["deltap"] = self.variables["p"] - self.problem_parameters["p_bar"]

    def _check_convergence(self, objective_pre_solver_non_discretized: float) -> bool:
        """
        Check convergence of SCvx.
        """
        return abs(self.new_objective - objective_pre_solver_non_discretized) < self.params.stop_crit * abs(
            objective_pre_solver_non_discretized
        )

    def _check_convergence2(self, X_solution, P_solution) -> bool:
        """
        Check convergence of SCvx.
        """
        print(" X", cvx.norm(X_solution - self.problem_parameters["X_bar"].value, 1).value)
        print(" P", cvx.norm(P_solution - self.problem_parameters["p_bar"].value, 1).value)
        return bool(
            (
                cvx.norm(X_solution - self.problem_parameters["X_bar"].value, 1)
                + cvx.norm(P_solution - self.problem_parameters["p_bar"].value, 1)
            ).value
            <= self.params.stop_crit
        )

    def _update_trust_region(self, rho: float):
        """
        Update trust region radius.
        """
        if rho < self.params.rho_1:
            self.problem_parameters["radius_trust_region"].value = max(
                self.params.min_tr_radius, self.problem_parameters["radius_trust_region"].value / self.params.alpha
            )
        elif rho >= self.params.rho_1 and rho < self.params.rho_2:
            pass
        else:
            self.problem_parameters["radius_trust_region"].value = min(
                self.params.max_tr_radius, self.params.beta * self.problem_parameters["radius_trust_region"].value
            )

    @staticmethod
    def _extract_seq_from_array(
        ts: tuple, F: np.ndarray, ddelta: np.ndarray, npstates: np.ndarray
    ) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Create DgSampledSequence from provided numpy arrays and timestamps.

        Parameters:
            ts (tuple[int, ...]): Timestamps corresponding to the samples.
            F (np.ndarray): Force or action inputs (1D array).
            ddelta (np.ndarray): Secondary inputs, e.g., deltas or accelerations (1D array).
            npstates (np.ndarray): State trajectory (2D array with one row per timestamp).

        Returns:
            tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
            Command and state sequences.
        """
        # Validate inputs
        # assert (
        #     len(ts) == F.shape[0] == ddelta.shape[0] == npstates.shape[0]
        # ), "All inputs must have the same length as the number of timestamps."
        assert len(ts) == F.shape[0], "Length of ts must match the number of rows in F."
        assert len(ts) == ddelta.shape[0], "Length of ts must match the number of rows in ddelta."
        assert len(ts) == npstates.shape[0], "Length of ts must match the number of rows in npstates."

        # Create the command sequence
        cmds_list = [SpaceshipCommands(f, dd) for f, dd in zip(F, ddelta)]
        mycmds = DgSampledSequence[SpaceshipCommands](timestamps=ts, values=cmds_list)

        # Create the state sequence
        states = [SpaceshipState(*v) for v in npstates]
        mystates = DgSampledSequence[SpaceshipState](timestamps=ts, values=states)

        return mycmds, mystates


class Tester:
    """
    Class for testing the planner.
    """

    def __init__(self, planner):
        self.planner = planner

    def check_dynamics(self):
        """
        Check the dynamics between integration of A_bar, B_minus, B_plus, F_bar, r_bar and the result of
        """
        # X_bar, U_bar, p_bar = planner.initial_guess()
        U_bar = np.random.randint(-2, 2, (2, self.planner.params.K))
        p_bar = np.random.randint(0, 10, (1))
        x0 = np.random.rand(8)
        X_integrated_ground_truth = self.planner.integrator.integrate_nonlinear_full(x0, U_bar, p_bar)

        self.planner.X_bar = X_integrated_ground_truth
        self.planner.U_bar = U_bar
        self.planner.p_bar = p_bar

        print(f"X_integrated_gt shape: {X_integrated_ground_truth.shape}")
        print(f"U_bar shape: {U_bar.shape}")
        print(f"p_bar shape: {p_bar.shape}")

        A_bar, B_minus, B_plus, F_bar, r_bar = planner._convexification()
        X_res = np.zeros_like(X_integrated_ground_truth)
        X_res[:, 0] = X_integrated_ground_truth[:, 0]

        for k in range(X_integrated_ground_truth.shape[1] - 1):
            X_res[:, k + 1] = (
                A_bar[k] @ X_res[:, k]
                + B_minus[k] @ U_bar[:, k]
                + B_plus[k] @ U_bar[:, k + 1]
                + (F_bar[k] * p_bar)
                + r_bar[k]
            )
            # print(X_res[:, k + 1])
            # if X_res[7, k + 1] != 2:
            #     print(f"A_bar term: {A_bar[k] @ X_res[:, k]}")
            #     print(f"B_minus term: {B_minus[k] @ U_bar[:, k]}")
            #     print(f"B_plus term: {B_plus[k] @ U_bar[:, k + 1]}")
            #     print(f"F_bar term: {(F_bar[k] * p_bar).reshape(-1)}")
            #     print(f"r_bar term: {r_bar[k]}")

        if not np.allclose(X_res, X_integrated_ground_truth):
            print("Dynamics are not correct.")
            tol = 0.001
            indices = np.where(np.any(X_res - X_integrated_ground_truth >= tol, axis=1))[0]
            print(f"indices: {indices}")
            for i in indices:
                # print(f"X_res at {i}: {X_res[:, i]}")
                # print(f"X_integrated_GT at {i}: {X_integrated_ground_truth[:,i]}")
                print(f"Error at {i}: {X_res[:, i] - X_integrated_ground_truth[:, i]}")
        else:
            print("Dynamics are correct.")


# incompleto, non dovrebbe servire.
if __name__ == "__main__":
    sg = SpaceshipGeometry(
        color="royalblue",
        m=2.0,  # MASS TO BE INTENDED AS MASS OF THE ROCKET WITHOUT FUEL
        Iz=1e-00,
        w_half=0.4,
        l_c=0.8,
        l_f=0.5,
        l_r=1,
        l_t_half=0.3,
        e=0.1,  # Example value for 'e'
        w_t_half=0.3,
        F_max=3.0,
    )  # Example value for 'F_max'

    sp = SpaceshipParameters(
        m_v=2.0,
        C_T=0.01,
        vx_limits=(-10 / 3.6, 10 / 3.6),
        acc_limits=(-1.0, 1.0),
        thrust_limits=(-2.0, 2.0),
        delta_limits=(-np.deg2rad(60), np.deg2rad(60)),
        ddelta_limits=(-np.deg2rad(45), np.deg2rad(45)),
    )
    planets = {}
    satellites = {}
    planner = SpaceshipPlanner(planets, satellites, sg, sp)
    init_state = SpaceshipState(0, 0, 0, 0, 0, 0, 0, 2)
    goal_state = DynObstacleState(10, 10, 0.1, 0.1, 0.1, 0)

    tester = Tester(planner)
    tester.check_dynamics()
    # mycmds, mystates = planner.compute_trajectory(init_state, goal_state)
