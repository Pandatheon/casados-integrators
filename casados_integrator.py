# Copyright Jonathan Frey, Jochem De Schutter, Moritz Diehl

# The 2-Clause BSD License

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from casadi import Callback, Sparsity, Function, CasadiMeta, DM
import casadi
from acados_template import AcadosSimSolver, AcadosSim, casadi_length
import numpy as np



def check_casadi_version():
    casadi_version = CasadiMeta.version()
    major, minor, patch = casadi_version.split(".")
    if int(major) < 3:
        raise Exception(
            f"casadi version {casadi_version} is too old. Need at least 3.0.0"
        )
    elif int(major) == 3:
        if int(minor) < 6:
            raise Exception(
                f"This version of CasadosIntegrator supports CasADi version >= 3.6.0. CasADi version {casadi_version} was found. Please look for an older version of CasadosIntegrator at https://github.com/FreyJo/casados-integrators or upgrade your CasADi version."
            )
    else:
        print(f"Warning: This version of CasadosIntegrator supports CasADi version >= 3.6.0. CasADi version {casadi_version} was found and is not tested.")


class CasadosIntegrator(Callback):
    """
    This class is a wrapper of the acados integrator (AcadosSimSolver) into a CasADi Callback.
    It offers:
        - first order forward sensitivities (via get_jacobian())
        - first order adjoint sensitivities (via get_reverse())
        - second order sensitivities (hessians) with adjoint seed (via get_reverse() + get_jacobian()) (for acados integrators that offer second order senitivities)
    This makes it fully functional within CasADi NLPs
    """

    def __init__(self, acados_sim: AcadosSim, use_cython=True, code_reuse=False):

        check_casadi_version()

        if use_cython:
            json_file = f"acados_sim_{acados_sim.model.name}.json"
            if not code_reuse:
                AcadosSimSolver.generate(acados_sim, json_file=json_file)
                AcadosSimSolver.build(
                    acados_sim.code_export_directory, with_cython=True
                )
            self.acados_integrator = AcadosSimSolver.create_cython_solver(json_file)
        else:
            self.acados_integrator = AcadosSimSolver(acados_sim)

        self.nx = casadi_length(acados_sim.model.x)
        self.nu = casadi_length(acados_sim.model.u)
        self.np_ = casadi_length(acados_sim.model.p)
        self.npg = casadi_length(acados_sim.model.p_global)
        self.nparam = self.nu + self.np_ + self.npg
        self.n_extra = self.np_ + self.npg

        self.model_name = acados_sim.model.name
        self.print_level = 0
        # self.print_level = 1

        self.x0 = None
        self.u0 = None

        # needed to keep the callback alive
        self.jac_callback = None
        self.adj_callback = None
        self.hess_callback = None

        self.reset_timings()

        Callback.__init__(self)
        self.construct("CasadosIntegrator")


    def set_z_guess(self, z0):
        '''
        Set initial guess of the algebraic variables

        Parameters
        ----------
        z0 : np.ndarray
            value of the algebraic variables guess
        '''
        # set the value in the integrator to be used in the first call
        self.acados_integrator.set("z", z0)

    def set_xdot_guess(self, xdot0):
        '''
        Set initial guess of xdot

        Parameters
        ----------
        xdot0 : np.ndarray
            value of xdot guess
        '''
        # set the value in the integrator to be used in the first call
        self.acados_integrator.set("xdot", xdot0)

    def get_sparsity_in(self, i):
        if i == 0:
            out = Sparsity.dense(self.nx)
        elif i == 1:
            out = Sparsity.dense(self.nparam)   # [u; p; p_global]
        elif i == 2:
            out = Sparsity.dense(1, 1)
        return out

    def get_sparsity_out(self, i):
        out = Sparsity.dense(self.nx)
        return out

    def get_name_in(self, i):
        if i == 0:
            out = "x0"
        elif i == 1:
            out = "p"
        elif i == 2:
            out = "dt"
        return out

    def get_n_in(self):
        return 3

    def get_n_out(self):
        return 1

    def get_name_out(self, i):
        return "xf"

    def eval(self, arg):
        # extract inputs
        x0 = np.array(arg[0])
        param = np.array(arg[1])
        dt = float(arg[2])
        if self.print_level:
            print(f"CasadosIntegrator: x0 {x0} param {param} dt {dt}")

        self.acados_integrator.options_set("sens_forw", False)
        self.acados_integrator.options_set("sens_adj", False)
        self.acados_integrator.options_set("sens_hess", False)
        # set
        self._set_solver_inputs(x0, param, dt)

        # solve
        status = self.acados_integrator.solve()

        # output
        x_next = self.acados_integrator.get("x")

        self.time_sim += self.acados_integrator.get("time_tot")
        return [x_next]

    def has_jacobian(self, *args) -> bool:
        return True

    def get_jacobian(self, *args):

        if self.jac_callback is None:
            self.jac_callback = CasadosIntegratorSensForw(self)

        return self.jac_callback

    def has_reverse(self, nadj) -> bool:
        return nadj == 1

    def get_reverse(self, *args) -> "casadi.Function":

        if self.adj_callback is None:
            self.adj_callback = CasadosIntegratorSensAdj(self)

        return self.adj_callback

    def reset_timings(self):
        self.time_sim = 0.0
        self.time_forw = 0.0
        self.time_adj = 0.0
        self.time_hess = 0.0

    def _set_solver_inputs(self, x0, param, dt):
        """Split the combined param vector and push everything into the solver."""
        param = np.asarray(param).flatten()
        u0 = param[: self.nu]
        self.acados_integrator.set("x", np.asarray(x0).flatten())
        self.acados_integrator.set("u", u0)
        if self.np_ > 0:
            self.acados_integrator.set("p", param[self.nu : self.nu + self.np_])
        if self.npg > 0:
            self.acados_integrator.set_p_global_and_precompute_dependencies(
                param[self.nu + self.np_ :]
            )
        self.acados_integrator.set("T", float(dt))

# NOTE: doesnt even get called -> dead end -> see https://github.com/casadi/casadi/issues/2019
# def uses_output(self, *args) -> bool:
#     r"""
#     uses_output(Function self) -> bool
#     Do the derivative functions need nondifferentiated outputs?
#     """
#     print("in uses_output()\n\n")
#     return False


# JACOBIAN
class CasadosIntegratorSensForw(Callback):
    def __init__(self, casados_integrator):
        self.acados_integrator = casados_integrator.acados_integrator
        self.casados_integrator = casados_integrator

        self.nx = casados_integrator.nx
        self.nu = casados_integrator.nu
        self.nparam = casados_integrator.nparam
        self.n_extra = casados_integrator.n_extra
        self.print_level = casados_integrator.print_level

        Callback.__init__(self)
        self.construct("CasadosIntegratorSensForw")
        # casados_integrator.jac_callback = self

    def get_sparsity_in(self, i):
        if i == 0:
            out = Sparsity.dense(self.nx)
        elif i == 1:
            out = Sparsity.dense(self.nparam)
        elif i == 2:
            out = Sparsity.dense(1, 1)
        elif i == 3:
            out = Sparsity.dense(self.nx)
        return out

    def get_sparsity_out(self, i):
        if i == 0:
            out = Sparsity.dense(self.nx, self.nx)
        elif i == 1:
            out = Sparsity.dense(self.nx, self.nparam)
        elif i == 2:
            out = Sparsity(self.nx, 1)
        return out

    def get_name_in(self, i):
        return ["x0", "p", "dt", "xf"][i]

    def get_n_in(self):
        return 4

    def get_n_out(self):
        return 3

    def get_name_out(self, i):
        return ["jac_xf_x0", "jac_xf_p", "jac_xf_dt"][i]

    def eval(self, arg):
        # extract inputs
        x0 = np.array(arg[0])
        param = np.array(arg[1])
        dt = float(arg[2])
        if self.print_level:
            print(f"CasadosIntegratorSensForw: x0 {x0} param {param} dt {dt}")

        self.casados_integrator._set_solver_inputs(x0, param, dt)
        self.acados_integrator.options_set("sens_forw", True)
        self.acados_integrator.options_set("sens_adj", False)
        self.acados_integrator.options_set("sens_hess", False)
        # solve
        status = self.acados_integrator.solve()

        # output
        S_forw = self.acados_integrator.get("S_forw")
        # S_forw = np.ascontiguousarray(S_forw.reshape(S_forw.shape, order="F"))
        self.casados_integrator.time_forw += self.acados_integrator.get("time_tot")

        jac_x0 = S_forw[:, : self.nx]
        jac_p = np.hstack([S_forw[:, self.nx :], np.zeros((self.nx, self.n_extra))])
        return [jac_x0, jac_p, DM(self.nx, 1)]

    def has_jacobian(self, *args) -> bool:
        return False

    def has_reverse(self, nadj) -> bool:
        # print(f"CasadosIntegratorSensForw: has_reverse, nadj: {nadj}\n")
        return False


# Citing casadi docstrings:
# Get a function that calculates nadj adjoint derivatives.

# Returns a function with n_in + n_out + n_out inputs and n_in outputs.
# The first n_in inputs correspond to nondifferentiated inputs.
# The next n_out inputs correspond to nondifferentiated outputs.
# The last n_out inputs correspond to adjoint seeds, stacked horizontally

# The n_in outputs correspond to adjoint sensitivities, stacked horizontally.
# (n_in = n_in(),
# n_out = n_out())

# (n_in = n_in(), n_out = n_out())

# ADJOINT
class CasadosIntegratorSensAdj(Callback):
    def __init__(self, casados_integrator):
        self.acados_integrator = casados_integrator.acados_integrator
        self.casados_integrator = casados_integrator

        self.nx = casados_integrator.nx
        self.nu = casados_integrator.nu
        self.nparam = casados_integrator.nparam
        self.n_extra = casados_integrator.n_extra
        self.print_level = casados_integrator.print_level

        Callback.__init__(self)
        self.construct("CasadosIntegratorSensAdj")

    def get_sparsity_in(self, i):
        if i == 0:
            out = Sparsity.dense(self.nx, 1)
        elif i == 1:
            out = Sparsity.dense(self.nparam, 1)
        elif i == 2:
            out = Sparsity.dense(1, 1)
        elif i == 3:
            out = Sparsity(self.nx, 1)
        elif i == 4:
            out = Sparsity.dense(self.nx, 1)
        return out

    def get_sparsity_out(self, i):
        if i == 0:
            out = Sparsity.dense(self.nx)
        elif i == 1:
            out = Sparsity.dense(self.nparam)
        elif i == 2:
            out = Sparsity(1, 1)
        return out

    def get_name_in(self, i):
        return ["x0", "p", "dt", "nominal_out", "adj_seed"][i]

    def get_n_in(self):
        return 5

    def get_n_out(self):
        return 3

    def get_name_out(self, i):
        return ["S_adj_x0", "S_adj_p", "S_adj_dt"][i]

    def eval(self, arg):
        # extract inputs
        x0 = np.array(arg[0])
        param = np.array(arg[1])
        dt = float(arg[2])
        seed = np.array(arg[4])
        if self.print_level:
            print(f"CasadosIntegratorSensAdj: x0 {x0} param {param} dt {dt} seed {seed}")

        # set adj seed:
        self.acados_integrator.set("seed_adj", seed)
        self.casados_integrator._set_solver_inputs(x0, param, dt)
        self.acados_integrator.options_set("sens_adj", True)
        self.acados_integrator.options_set("sens_forw", False)
        self.acados_integrator.options_set("sens_hess", False)
        status = self.acados_integrator.solve()

        S_adj = self.acados_integrator.get("S_adj")   # (nx+nu,)

        if self.print_level > 1:
            print(f"\nevaluated acados integrator in callback, got S_adj: {S_adj}\n")

        S_adj_x = S_adj[: self.nx]
        S_adj_u = S_adj[self.nx :]
        S_adj_p = np.concatenate([S_adj_u, np.zeros(self.n_extra)])

        self.casados_integrator.time_adj += self.acados_integrator.get("time_tot")

        return [S_adj_x, S_adj_p, DM(1, 1)]

    def has_jacobian(self, *args) -> bool:
        return True

    def get_jacobian(self, *args):

        if self.casados_integrator.hess_callback is None:
            self.casados_integrator.hess_callback = CasadosIntegratorSensHess(
                self.casados_integrator
            )

        return self.casados_integrator.hess_callback


# HESSIAN
class CasadosIntegratorSensHess(Callback):
    def __init__(self, casados_integrator):
        self.acados_integrator = casados_integrator.acados_integrator
        self.casados_integrator = casados_integrator

        self.nx = casados_integrator.nx
        self.nu = casados_integrator.nu
        self.nparam = casados_integrator.nparam
        self.n_extra = casados_integrator.n_extra
        self.print_level = casados_integrator.print_level

        Callback.__init__(self)
        self.construct("CasadosIntegratorSensHess")

    def get_sparsity_in(self, i):
        if i == 0:
            out = Sparsity.dense(self.nx, 1)
        elif i == 1:
            out = Sparsity.dense(self.nparam, 1)
        elif i == 2:
            out = Sparsity.dense(1, 1)
        elif i == 3:
            out = Sparsity.dense(self.nx, 1)
        elif i == 4:
            out = Sparsity.dense(self.nx, 1)
        elif i == 5:
            out = Sparsity.dense(self.nx, 1)
        elif i == 6:
            out = Sparsity.dense(self.nparam, 1)
        elif i == 7:
            out = Sparsity.dense(1, 1)
        return out

    def get_sparsity_out(self, i):
        nx, nu, npar = self.nx, self.nu, self.nparam
        if i == 0:
            out = Sparsity.dense(nx, nx)
        elif i == 1:
            out = Sparsity.dense(nx, npar)
        elif i == 2:
            out = Sparsity(nx, 1)
        elif i == 3:
            out = Sparsity.dense(nx, nx)
        elif i == 4:
            out = Sparsity.dense(nx, nx)
        elif i == 5:
            out = Sparsity.dense(npar, nx)
        elif i == 6:
            out = Sparsity.dense(npar, npar)
        elif i == 7:
            out = Sparsity(npar, 1)
        elif i == 8:
            out = Sparsity.dense(npar, nx)
        elif i == 9:
            out = Sparsity.dense(npar, nx)
        elif i == 10:
            out = Sparsity(1, nx)
        elif i == 11:
            out = Sparsity(1, npar)
        elif i == 12:
            out = Sparsity(1, 1)
        elif i == 13:
            out = Sparsity(1, nx)
        elif i == 14:
            out = Sparsity(1, nx)
        return out

    def get_name_in(self, i):
        if i == 0:
            out = "x0"
        elif i == 1:
            out = "u0"
        elif i == 2:
            out = "nominal_out"
        elif i == 3:
            out = "adj_seed"
        elif i == 4:
            out = "S_adj_out_x"
        elif i == 5:
            out = "S_adj_out_u"
        elif i == 6:
            out = "S_adj_out_p"
        elif i == 7:
            out = "S_adj_out_dt"
        return out

    def get_n_in(self):
        return 8

    def get_n_out(self):
        return 15

    def eval(self, arg):
        # extract inputs
        x0 = np.array(arg[0])
        param = np.array(arg[1])
        dt = float(arg[2])
        seed = np.array(arg[4])

        if self.print_level:
            print(f"CasadosIntegratorSensHess: x0 {x0} param {param} dt {dt} seed {seed}")

        # set adj seed:
        self.acados_integrator.set("seed_adj", seed)
        self.casados_integrator._set_solver_inputs(x0, param, dt)
        self.acados_integrator.options_set("sens_hess", True)
        self.acados_integrator.options_set("sens_forw", True)
        self.acados_integrator.options_set("sens_adj", True)
        status = self.acados_integrator.solve()

        S_hess = self.acados_integrator.get("S_hess")
        S_forw = self.acados_integrator.get("S_forw")

        self.casados_integrator.time_hess += self.acados_integrator.get("time_tot")

        nx, nu, ne = self.nx, self.nu, self.n_extra

        Hxx = S_hess[:nx, :nx]
        Hxu = S_hess[:nx, nx:]
        Hux = S_hess[nx:, :nx]
        Huu = S_hess[nx:, nx:]
        FxT = S_forw[:, :nx].T
        FuT = (S_forw[:, nx:]).T

        Hxp = np.hstack([Hxu, np.zeros((nx, ne))])
        Hpx = np.vstack([Hux, np.zeros((ne, nx))])
        Hpp = np.hstack([
            np.vstack([Huu, np.zeros((ne, nu))]),
            np.zeros((nu + ne, ne)),
        ])
        FpT = np.vstack([FuT, np.zeros((ne, nx))])

        return [
            Hxx,
            Hxp,
            DM(nx, 1),
            np.zeros((nx, nx)),
            FxT,
            Hpx,
            Hpp,
            DM(self.nparam, 1),
            np.zeros((self.nparam, nx)),
            FpT,
            DM(1, nx),
            DM(1, self.nparam),
            DM(1, 1),
            DM(1, nx),
            DM(1, nx),
        ]

    def has_jacobian(self, *args) -> bool:
        return False
