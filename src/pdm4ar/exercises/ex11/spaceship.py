import sympy as spy
import numpy as np

from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry, SpaceshipParameters


class SpaceshipDyn:
    sg: SpaceshipGeometry
    sp: SpaceshipParameters

    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    f: spy.Function
    A: spy.Function
    B: spy.Function
    F: spy.Function

    def __init__(self, sg: SpaceshipGeometry, sp: SpaceshipParameters):
        self.sg = sg
        self.sp = sp

        self.x = spy.Matrix(spy.symbols("x y psi vx vy dpsi delta m", real=True))  # states
        self.u = spy.Matrix(spy.symbols("thrust ddelta", real=True))  # inputs
        self.p = spy.Matrix([spy.symbols("t_f", positive=True)])  # final time

        self.n_x = self.x.shape[0]  # number of states
        self.n_u = self.u.shape[0]  # number of inputs
        self.n_p = self.p.shape[0]

    def get_dynamics(self) -> tuple[spy.Function, spy.Function, spy.Function, spy.Function]:
        """
        Define dynamics and extract jacobians.
        Get dynamics for SCvx.
        0x 1y 2psi 3vx 4vy 5dpsi 6delta 7m
        """
        # Dynamics
        f = spy.zeros(self.n_x, 1)
        # print(self.x[3], type(self.x[3]))
        f[0] = self.x[3] * spy.cos(self.x[2]) - self.x[4] * spy.sin(self.x[2])
        f[1] = self.x[3] * spy.sin(self.x[2]) + self.x[4] * spy.cos(self.x[2])
        f[2] = self.x[5]
        f[3] = 1 / self.x[7] * spy.cos(self.x[6]) * self.u[0] + self.x[5] * self.x[4]
        f[4] = 1 / self.x[7] * spy.sin(self.x[6]) * self.u[0] - self.x[5] * self.x[3]
        f[5] = -(self.sg.l_r / self.sg.Iz) * spy.sin(self.x[6]) * self.u[0]
        f[6] = self.u[1]
        f[7] = -1 * self.u[0] * self.sp.C_T
        f = f * self.p[0]
        A = f.jacobian(self.x)
        B = f.jacobian(self.u)
        F = f.jacobian(self.p)

        f_func = spy.lambdify((self.x, self.u, self.p), f, "numpy")
        A_func = spy.lambdify((self.x, self.u, self.p), A, "numpy")
        B_func = spy.lambdify((self.x, self.u, self.p), B, "numpy")
        F_func = spy.lambdify((self.x, self.u, self.p), F, "numpy")

        return f_func, A_func, B_func, F_func


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

    sd = SpaceshipDyn(sg, sp)
    f_func, A_func, B_func, F_func = sd.get_dynamics()

    # Define symbolic variables
    x0, x1, x2, x3, x4, x5, x6, x7, u1, u2, tf = spy.symbols("x0 x1 x2 x3 x4 x5 x6 x7 u1 u2 tf")

    # Create the matrix with the symbolic variable
    x = spy.Matrix([x0, x1, x2, x3, x4, x5, x6, x7])

    u = spy.Matrix([u1, u2])
    p = spy.Matrix([tf])
    # Add these debug prints before the print(f_func(x, u, p)) line
    print(f"x: {x}, type: {type(x)}")
    print(f"u: {u}, type: {type(u)}")
    print(f"p: {p}, type: {type(p)}")

    print(f_func([0, 0, np.pi, 2, 0, 0, 0, 1], [0, 0], [10]))
