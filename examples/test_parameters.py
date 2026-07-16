from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, AcadosCasadiOcpSolver
import numpy as np
import sys
sys.path.insert(0, '../getting_started')

import casadi as ca

def export_pendulum_ode_model() -> AcadosModel:
    # constants
    m_cart = 1. # mass of the cart [kg]

    # parameters
    g = ca.MX.sym("g")
    p = g
    m = ca.MX.sym("m")
    l = ca.MX.sym("l")
    p_global = ca.vertcat(m,l)

    # set up states & controls
    x1      = ca.MX.sym('x1')
    theta   = ca.MX.sym('theta')
    v1      = ca.MX.sym('v1')
    dtheta  = ca.MX.sym('dtheta')

    x = ca.vertcat(x1, theta, v1, dtheta)

    F = ca.MX.sym('F')
    u = ca.vertcat(F)

    # xdot
    nx = x.shape[0]
    xdot = ca.MX.sym('xdot', nx)

    # dynamics
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    denominator = m_cart + m - m*cos_theta**2
    f_expl = ca.vertcat(v1,
                     dtheta,
                     (-m*l*sin_theta*dtheta**2 + m*g*cos_theta*sin_theta+F)/denominator,
                     (-m*l*cos_theta*sin_theta*dtheta**2 + F*cos_theta+(m_cart+m)*g*sin_theta)/(l*denominator)
                     )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p
    model.p_global = p_global
    model.name = 'pendulum_ode'

    # store meta information
    model.x_labels = ['$x$ [m]', r'$\theta$ [rad]', '$v$ [m]', r'$\dot{\theta}$ [rad/s]']
    model.u_labels = ['$F$']
    model.t_label = '$t$ [s]'

    return model


def ocp_formulation() -> AcadosOcp:

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_pendulum_ode_model()
    ocp.model = model

    # dimensions
    nx = model.x.rows()
    nu = model.u.rows()

    #parameters
    yref_param = ca.MX.sym('yref', nx+nu)
    constraint_quotient = ca.MX.sym('C')
    p = ca.vertcat(yref_param, constraint_quotient, model.p)
    ocp.model.p = p

    # set cost
    Q = 2*np.diag([1e3, 1e3, 1e-2, 1e-2])
    R = 2*np.diag([1e-2])

    # path cost
    ocp.cost.cost_type = 'EXTERNAL'
    residual = ca.vertcat(model.x, model.u) - yref_param
    W = ca.diagcat(Q, R).full()
    ocp.model.cost_expr_ext_cost = residual.T @ W @ residual

    # terminal cost
    ocp.cost.cost_type_e = 'EXTERNAL'
    res_e = model.x - yref_param[0:nx]
    ocp.model.cost_expr_ext_cost_e = res_e.T @ Q @ res_e

    # set constraints
    Fmax = 80
    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])

    constraint_quotient = p[nx+nu]
    ocp.model.con_h_expr = model.x[2]/ constraint_quotient
    ocp.constraints.lh = np.array([-1])
    ocp.constraints.uh = np.array([1])

    ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])

    p_0 = np.hstack((np.zeros((nx+nu,)), 0.1, 9.81))
    ocp.parameter_values = p_0
    ocp.p_global_values = np.array([0.1, 0.8])

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'EXACT' # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization

    Tf = 1.0
    N_horizon = 20

    # set prediction horizon
    ocp.solver_options.tf = Tf
    ocp.solver_options.N_horizon = N_horizon

    return ocp


def main(stage_varying=True, global_p=True, uniform_time=True):

    print(f"\n\nRunning example with stage_varying_p={stage_varying} and global_p={global_p}\n\n")
    # create ocp
    ocp = ocp_formulation()
    if not uniform_time:
        tau = np.linspace(0, 1, ocp.solver_options.N_horizon+1)
        nodes = ocp.solver_options.tf * tau**3
        ocp.solver_options.time_steps = np.diff(nodes)

    # create acados solver
    print(f"Creating ocp solver with p_global = {ocp.model.p_global}, p = {ocp.model.p}")
    ocp_solver = AcadosOcpSolver(ocp, generate=True, build=True, verbose=False, save_p_global=True)
    nx = ocp.dims.nx
    nu = ocp.dims.nu
    N_horizon = ocp.solver_options.N_horizon
    if stage_varying:
        for i in range(0, N_horizon+1):
            ocp_solver.set(stage_= i, field_= 'p', value_=np.hstack((np.zeros((nx+nu,)), 0.1, 9.81+i*0.3)))
    if global_p:
        ocp_solver.set_p_global_and_precompute_dependencies(np.array([0.2, 0.6]))
    status = ocp_solver.solve()
    ocp_solver.print_statistics()

    if status != 0:
        raise Exception(f'acados returned status {status}.')
    result = ocp_solver.get_iterate()

    acasadi_solver = AcadosCasadiOcpSolver(ocp, 
                                              verbose=False,
                                              with_casados=True,)
    if stage_varying:
        for i in range(0, N_horizon+1):
            acasadi_solver.set(stage= i, field= 'p', value_=np.hstack((np.zeros((nx+nu,)), 0.1, 9.81+i*0.3)))
    if global_p:
        acasadi_solver.set_p_global_and_precompute_dependencies(np.array([0.2, 0.6]))
    acasadi_solver.set_iterate(result)
    acasadi_solver.solve()

    x_acados = ocp_solver.get_flat('x')
    u_acados = ocp_solver.get_flat('u')
    lambda_acados = ocp_solver.get_flat('lam')
    pi_acados = ocp_solver.get_flat('pi')

    x_casadi = acasadi_solver.get_flat('x')
    u_casadi = acasadi_solver.get_flat('u')
    lambda_casadi = acasadi_solver.get_flat('lam')
    pi_casadi = acasadi_solver.get_flat('pi')

    diff_x = np.linalg.norm(x_acados - x_casadi)
    print(f"||x_acados - x_casadi|| = {diff_x:.6e}")
    diff_u = np.linalg.norm(u_acados - u_casadi)
    print(f"||u_acados - u_casadi|| = {diff_u:.6e}")
    diff_lambda = np.linalg.norm(lambda_acados - lambda_casadi)
    print(f"||lambda_acados - lambda_casadi|| = {diff_lambda:.6e}")
    diff_pi = np.linalg.norm(pi_acados - pi_casadi)
    print(f"||pi_acados - pi_casadi|| = {diff_pi:.6e}")
    max_diff = max(diff_x, diff_u)
    if max_diff > 5e-4:
        raise Exception(f"Max difference between acados and casadi solver is {max_diff:.6e} >= 5e-4, FAILED")

if __name__ == "__main__":
    main(stage_varying=False, global_p=True, uniform_time=True)
    main(stage_varying=False, global_p=True, uniform_time=False)