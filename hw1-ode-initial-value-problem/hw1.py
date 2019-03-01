import os
import numpy
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.constants


def planetary_motion(y0, t0, t1, dt, G=1, M=1, a=1, omega=1):
    def system(t, ys):
        x = ys[0]
        y = ys[1]
        u = ys[2]
        v = ys[3]
        return numpy.vstack((u, v, -G * M * x / (x*x + y*y)**1.5, -G * M * y / (x*x + y*y)**1.5))

    solver = scipy.integrate.ode(system).set_integrator("dopri5")
    solver.set_initial_value(y0, t0)

    iter_outs = []

    while solver.t < t1:
        solver.integrate(solver.t + dt)
        iter_outs.append(numpy.array([solver.t, solver.y]))
    
    return numpy.array(iter_outs)


def plot_trajectory(solved, v0, title):
    fig = plt.figure()
    plt.scatter([0], [0], c="red")

    if not isinstance(v0, list):
        x = [s[0] for s in solved[:,1]]
        y = [s[1] for s in solved[:,1]]
        plt.plot(x, y)
    else:
        plots = []
        diff_v = list(set(numpy.abs(numpy.array(v0))))
        colors = ["b", "g", "y", "m", "c"]
        v_to_color = {}
        for i in range(len(diff_v)):
            v_to_color[diff_v[i]] = colors[i]
       
        v_seen = []
        for i, sol in enumerate(solved):
            x = [s[0] for s in sol[:,1]]
            y = [s[1] for s in sol[:,1]]
            p = plt.plot(x, y, v_to_color[abs(v0[i])], linewidth=1)
            if abs(v0[i]) not in v_seen:
                plots += p
                v_seen.append(abs(v0[i]))

        labels = [v if v != numpy.sqrt(2) else "$\\sqrt{2}$" for v in sorted(diff_v)]
        plt.legend(plots, labels, loc="center left", title="$v_0$")

    plt.axis("equal")
    plt.ylim(-1, 1)
    plt.grid(alpha=0.3)
    plt.title("Trajectories of planetary motion ($G=M=a=\\omega=1$)")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    fig.savefig("images/{}.pdf".format(title), bbox_inches="tight")
    plt.show()


def trajectory_err_elliptic(solved, dts):
    fig, axes = plt.subplots(2, 2, figsize=(20,8))
    fig.suptitle("Elliptic trajectories for $v_0 = 0.5$ with different $\\mathrm{d}t$'s")

    for i, sol in enumerate(solved):
        print("Plotting {}".format(i))
        x = [s[0] for s in sol[:,1]]
        y = [s[1] for s in sol[:,1]]
        
        axes[i//2, i%2].scatter([0], [0], c="red")
        axes[i//2, i%2].set_title("$\\mathrm{{d}}t = {0}$".format(dts[i]))
        axes[i//2, i%2].set_ylim(bottom=-0.4, top=0.4)
        axes[i//2, i%2].plot(x, y, linewidth=0.5)
    
    fig.savefig("images/err_elliptic_dt__v0=0,5.pdf", bbox_inches="tight")
    plt.show()


def trajectory_err_diff_v(solved, dts, vs):

    fig, axes = plt.subplots(2, 2, figsize=(10,10))
    fig.suptitle("Elliptic trajectories for $v_0 \\in \\{0.5, 0.75, 1\\}$ with different $\\mathrm{d}t$'s")

    for v in vs:
        for i, sol in enumerate(solved[v]):
            print("Plotting {}".format(i))
            x = [s[0] for s in sol[:,1]]
            y = [s[1] for s in sol[:,1]]
            
            axes[i//2, i%2].scatter([0], [0], c="red")
            axes[i//2, i%2].set_title("$\\mathrm{{d}}t = {0}$".format(dts[i]))
            axes[i//2, i%2].set_xlim((-1.1,1.1))
            axes[i//2, i%2].set_ylim(bottom=-1.1, top=1.1)
            axes[i//2, i%2].plot(x, y, linewidth=0.5)
    
    fig.savefig("images/err_elliptic_dt_v0.pdf", bbox_inches="tight")
    plt.show()


def task1():
    t0, t1 = 0, 15
    dt = 0.01
    v0 = [0.5, 1, 1.2, numpy.sqrt(2), -numpy.sqrt(2), 2, -2]
    solved = []
    for v in v0:
        y0 = [1, 0, 0, v]
        solved.append(planetary_motion(y0, t0, t1, dt))

    plot_trajectory(solved, v0, "trajectory")


def task1_err_ellipse():
    t0, t1 = 0, 20
    dts = [0.5, 0.1, 0.01, 0.001]
    v0 = 0.5
    y0 = [1, 0, 0, v0]
    solved = []
    for dt in dts:
        print("dt = {}".format(dt))
        solved.append(planetary_motion(y0, t0, t1, dt))

    trajectory_err_elliptic(solved, dts)


def task1_err_ellipses():
    t0, t1 = 0, 20
    dts = [0.5, 0.1, 0.01, 0.001]
    v0 = [0.5, 0.75, 1]

    solved = {}
    for v in v0:
        y0 = [1, 0, 0, v]
        solved[v] = []
        for dt in dts:
            print("dt = {}".format(dt))
            solved[v].append(planetary_motion(y0, t0, t1, dt))

    trajectory_err_diff_v(solved, dts, v0)


def task1_err_to_dt():
    t0, t1 = 0, 100
    dt1 = 7
    N = 10000 * dt1
    dts = numpy.flip(numpy.linspace(0.0001, dt1, N))
    v0 = 1
    y0 = [1, 0, 0, v0]
    solved = []
    errors = []
    errorsm = []
    for dt in dts:
        print("dt = {}".format(dt))

        sol = planetary_motion(y0, t0, t1, dt)
        solved.append(sol)

        x = numpy.array([s[0] for s in sol[:,1]])
        y = numpy.array([s[1] for s in sol[:,1]])
        steps = len(x)
        errors.append(numpy.sqrt(numpy.sum((numpy.ones(steps) - numpy.sqrt(x*x + y*y))**2)) / steps)

        # midpoints
        xm = numpy.zeros(steps)
        ym = numpy.zeros(steps)
        for i in range(steps-1):
            xm[i+1] = (x[i] + x[i+1]) / 2
            ym[i+1] = (y[i] + y[i+1]) / 2
        errorsm.append(numpy.sqrt(numpy.sum((numpy.ones(steps) - numpy.sqrt(xm*xm + ym*ym))**2)) / steps)

    fig = plt.figure()
    plt.title("Error of planetary motion with $v_0=1$ (circular) -- calculated points")
    plt.xlabel("$\\mathrm{d}t$")
    plt.ylabel("$\\mathrm{error} = ||\\, \\mathbf{1} - \\sqrt{\\mathbf{x}_c^2 + \\mathbf{y}_c^2}\\,||$")
    plt.plot(dts, errors, linewidth=0.5)
    fig.savefig("images/error_circular_wr_dt_calculated_{}.pdf".format(dt1), bbox_inches="tight")
    plt.show()

    fig = plt.figure()
    plt.title("Error of planetary motion with $v_0=1$ (circular) -- midpoints")
    plt.xlabel("$\\mathrm{d}t$")
    plt.ylabel("$\\mathrm{error} = ||\\, \\mathbf{1} - \\sqrt{\\mathbf{x}_m^2 + \\mathbf{y}_m^2}\\,||$")
    plt.plot(dts, errorsm, linewidth=0.5)
    fig.savefig("images/error_circular_wr_dt_midpoints_{}.pdf".format(dt1), bbox_inches="tight")
    plt.show()

    fig = plt.figure()
    plt.title("Error of planetary motion with $v_0=1$ (circular)")
    plt.xlabel("$\\mathrm{d}t$")
    plt.ylabel("$\\mathrm{error} = ||\\, \\mathbf{1} - \\sqrt{\\mathbf{x}^2 + \\mathbf{y}^2}\\,||$")
    p1, = plt.plot(dts, errors, linewidth=0.5)
    p2, = plt.plot(dts, errorsm, linewidth=0.5)
    plt.legend([p1, p2], ["calculated", "midpoints"], loc="upper left")
    fig.savefig("images/error_circular_wr_dt_{}.pdf".format(dt1), bbox_inches="tight")
    plt.show()


def task1_circular_dt(dt):
    t0, t1 = 0, 1000
    v0 = 1
    y0 = [1, 0, 0, v0]
    solved = planetary_motion(y0, t0, t1, dt)

    plot_trajectory(solved, v0, "trajectory_dt={}".format(dt))


def task1_stability():
    t0, t1 = 0, 1000
    v0 = 1
    y0 = [1, 0, 0, v0]
    dts = [1, 0.1, 0.01, 0.001, 0.0001]

    errors_radius = {}
    errors_energy = {}
    errors_angmoment = {}

    E0 = -0.5
    L0 = y0[0]*y0[3] - y0[1]*y0[2]

    for dt in dts:
        print(dt)

        sol = planetary_motion(y0, t0, t1*dt, dt)

        x = numpy.array([s[0] for s in sol[:,1]])
        y = numpy.array([s[1] for s in sol[:,1]])
        u = numpy.array([s[2] for s in sol[:,1]])
        v = numpy.array([s[3] for s in sol[:,1]])
        steps = len(x)
        errors_radius[dt] = numpy.abs(numpy.ones(steps) - numpy.sqrt(x*x + y*y))
        errors_energy[dt] = numpy.abs((x*x + y*y) / 2 - 1 - E0*numpy.ones(steps))
        errors_angmoment[dt] = numpy.abs(x*v - y*u - L0)

    # stability of distance
    fig = plt.figure()
    plt.title("Radius error")
    plots = []
    for dt in dts:
        plots += plt.semilogy(range(t1), errors_radius[dt][:t1], linewidth=0.5)
    plt.legend(plots, dts, loc="upper right")
    plt.xlabel("iter")
    plt.ylabel("$r_{\\mathrm{err}} = \\left|\\,1 - \\sqrt{x^2 + y^2}\\,\\right|$")
    fig.savefig("images/task2_stability_r.pdf", bbox_inches="tight")
    plt.show()

    # stability of energy
    fig = plt.figure()
    plt.title("Energy error")
    plots = []
    for dt in dts:
        plots += plt.semilogy(range(t1), errors_energy[dt][:t1], linewidth=0.5)
    plt.legend(plots, dts, loc="upper right")
    plt.xlabel("iter")
    plt.ylabel("$E_{\\mathrm{err}} = \\left|\\,\\left(\\frac{1}{2}(x^2 + y^2) - 1\\right) - \\frac{1}{2} \\,\\right|$")
    fig.savefig("images/task2_stability_E.pdf", bbox_inches="tight")
    plt.show()

    # stability of angular momentum
    fig = plt.figure()
    plt.title("Angular momentum error")
    plots = []
    for dt in dts:
        plots += plt.semilogy(range(t1), errors_angmoment[dt][:t1], linewidth=0.5)
    plt.legend(plots, dts, loc="upper right")
    plt.xlabel("iter")
    plt.ylabel("$L_{\\mathrm{err}} = \\left|\\, (xv - yu) - 1 \\,\\right|$")
    fig.savefig("images/task2_stability_L.pdf", bbox_inches="tight")
    plt.show()


def task1_small_ellipse():
    t0, t1 = 0, 100
    v0 = 0.1
    y0 = [1, 0, 0, v0]
    dt = 0.1
    solved = planetary_motion(y0, t0, t1, dt)

    plot_trajectory(solved, v0, "trajectory_v0={0}_dt={1}".format(v0, dt))

################################################################################
#
#       TASK 2
#
################################################################################

def planetary_motion2(y0, t0, t1, dt):
    def bypassing_position(t):
        return -10 + 2*t, 1.5

    def system(t, ys):
        x = ys[0]
        y = ys[1]
        u = ys[2]
        v = ys[3]
        xz, yz = bypassing_position(t)
        return numpy.vstack((u, v, -x / (x*x + y*y)**1.5 - (x - xz) / ( (x - xz)**2 + (y - yz)**2 )**1.5,
            -y / (x*x + y*y)**1.5 - (y - yz) / ( (x - xz)**2 + (y - yz)**2 )**1.5))

    solver = scipy.integrate.ode(system).set_integrator("dopri5")
    solver.set_initial_value(y0, t0)

    planet = []
    bypass = []

    while solver.t < t1:
        solver.integrate(solver.t + dt)
        planet.append(numpy.array([solver.t, solver.y]))
        bypass.append(numpy.array([solver.t, [bypassing_position(solver.t)]]))
    
    return numpy.array(planet), numpy.array(bypass)


def plot_trajectory2(planet, bypass, phi):
    fig = plt.figure(figsize=(8,7))
    plt.scatter([0], [0], c="red")

    xp = numpy.array([s[0] for s in planet[:,1]])
    yp = numpy.array([s[1] for s in planet[:,1]])
    xb = numpy.array([s[0][0] for s in bypass[:,1]])
    yb = numpy.array([s[0][1] for s in bypass[:,1]])

    plt.plot(xp, yp, linewidth=1)
    plt.plot(xb, yb, linewidth=0.5)

    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.title("Motion of a planet when star bypasses it ($\\varphi = {}\\pi$)".format(numpy.round(phi / scipy.constants.pi, 2)))
    plt.xlabel("x")
    plt.ylabel("y")

    fig.savefig("images/task2_trajectory_phi={}.pdf".format(phi), bbox_inches="tight")
    plt.show()


def task2():
    t0, t1 = 0, 100
    phi = scipy.constants.pi / 6
    y0 = [-numpy.sin(phi), numpy.cos(phi), numpy.cos(phi), numpy.sin(phi)]
    dt = 0.01

    planet, bypass = planetary_motion2(y0, t0, t1, dt)
    plot_trajectory2(planet, bypass, phi)

################################################################################
#
#       MAIN
#
################################################################################
if __name__ == "__main__":
    os.chdir("hw1-ode-initial-value-problem/")

    # task1()
    # task1_err_ellipse()
    # task1_err_ellipses()
    # task1_err_to_dt()
    # task1_circular_dt(dt=scipy.constants.pi/2)
    # task1_stability()
    # task1_small_ellipse()

    # task2()