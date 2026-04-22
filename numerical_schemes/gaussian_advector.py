import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

"""
Gaussian advection test — 1D periodic domain
Equation: dq/dt + c * dq/dx = 0
Exact solution: q(x,t) = exp(-((x - x0 - c*t) mod L)^2 / (2*sigma^2))

Spatial discretization: 2nd-order centered difference on C-grid staggered scheme
  dq/dx|_j ≈ (q_{j+1} - q_{j-1}) / (2*dx)   [collocated, for simplicity]

Schemes implemented:
  - Forward Euler         (1st order, unstable for wave eq — shown for reference)
  - Leap-Frog             (2nd order, neutrally stable, has computational mode)
  - Forward-Backward (FB) (2nd order, a_max=2, efficient)
  - LF-AM3 secondary      (3rd order, eps=0.39, beta=0.044, c_coeff=1/12)
  - RK4                   (classical 4th order, reference)

Usage:
  python gaussian_advection.py
  python gaussian_advection.py --scheme all --cfl 0.8 --nsteps 200
"""



# ── domain & initial condition ────────────────────────────────────────────────

def gaussian_ic(x, x0, sigma):
    return np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def exact(x, x0, sigma, c, t, L):
    # periodic wrap
    xc = (x - x0 - c * t) % L
    xc[xc > L / 2] -= L
    return np.exp(-(xc ** 2) / (2 * sigma ** 2))

# ── spatial derivative (2nd-order centered, periodic) ─────────────────────────

def dqdx(q, dx):
    return (np.roll(q, -1) - np.roll(q, 1)) / (2 * dx)


def dqdx_upwind(q, dx):
    return (q - np.roll(q, 1)) / dx
# ── time schemes ─────────────────────────────────────────────────────────────

def step_forward_euler(q, c, dx, dt, **_):
    return q - c * dt * dqdx_upwind(q, dx), None

def step_leapfrog(q, c, dx, dt, q_prev=None, **_):
    if q_prev is None:
        # self-start with Forward Euler
        q_prev = q.copy()
        q_new = q - c * dt * dqdx(q, dx)
    else:
        q_new = q_prev - 2 * c * dt * dqdx(q, dx)
    return q_new, q   # return (q_new, q_prev_for_next_step)

def step_fb(q, c, dx, dt, **_):
    # Forward-Backward: update zeta (q) first, then use updated q immediately
    # For scalar advection this maps to a 2-step FB:
    #   q* = q^n - c*dt * dqdx(q^n)     [forward step]
    #   q^{n+1} = q^n - c*dt * dqdx(q*) [backward correction]
    q_star = q - c * dt * dqdx(q, dx)
    q_new = q - c * dt * dqdx(q_star, dx)
    return q_new, None


def _rk4_step(q, c, dx, dt):
    k1 = -c * dqdx(q,              dx)
    k2 = -c * dqdx(q + 0.5*dt*k1, dx)
    k3 = -c * dqdx(q + 0.5*dt*k2, dx)
    k4 = -c * dqdx(q +     dt*k3, dx)
    return q + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

def step_rk4(q, c, dx, dt, **_):
    return _rk4_step(q, c, dx, dt), None

def _rk4_upwind_step(q, c, dx, dt):
    k1 = -c * (q - np.roll(q, 1)) / dx
    k2 = -c * ((q + 0.5*dt*k1) - np.roll(q + 0.5*dt*k1, 1)) / dx
    k3 = -c * ((q + 0.5*dt*k2) - np.roll(q + 0.5*dt*k2, 1)) / dx
    k4 = -c * ((q +     dt*k3) - np.roll(q +     dt*k3, 1)) / dx
    return q + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

def step_rk4_upwind(q, c, dx, dt, **_):
    return _rk4_upwind_step(q, c, dx, dt), None

def _rk2_step(q, c, dx, dt):
    k1 = -c * (q - np.roll(q, 1)) / dx
    k2 = -c * ((q + dt*k1) - np.roll(q + dt*k1, 1)) / dx
    return q + (dt / 2) * (k1 + k2)

def step_rk2(q, c, dx, dt, **_):
    return _rk2_step(q, c, dx, dt), None

def _dqdx_upwind5(q, dx):
    return ( -2*np.roll(q,  3)
            + 15*np.roll(q,  2)
            - 60*np.roll(q,  1)
            + 20*q
            + 30*np.roll(q, -1)
            -  3*np.roll(q, -2)) / (60 * dx)

def _rk4_upwind5_step(q, c, dx, dt):
    k1 = -c * _dqdx_upwind5(q,             dx)
    k2 = -c * _dqdx_upwind5(q + 0.5*dt*k1, dx)
    k3 = -c * _dqdx_upwind5(q + 0.5*dt*k2, dx)
    k4 = -c * _dqdx_upwind5(q +     dt*k3, dx)
    return q + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

def step_rk4_upwind5(q, c, dx, dt, **_):
    return _rk4_upwind5_step(q, c, dx, dt), None

# ── run one scheme ────────────────────────────────────────────────────────────

SCHEMES = {
    'forward_euler': (step_forward_euler, 'Forward Euler',     'C0', '--'),
    'leapfrog':      (step_leapfrog,      'Leap-Frog',         'C1', '--'),
    'fb':            (step_fb,            'Forward-Backward',  'C2', '--'),
    'rk2':           (step_rk2,           'RK2 (reference)',   'C5', '--'),
    'rk4_up':        (step_rk4_upwind,    'RK4 (upwind)',   'C4', '--'),
    'rk4_up5':       (step_rk4_upwind5,   'RK4 (upwind5)',   'C4', '--'),
    'rk4_centered':  (step_rk4,           'RK4 (centered)',   'C4', '--'),

}

def run(scheme_key, c, dx, dt, x, q0, nsteps, sigma, x0, L):
    fn, label, color, ls = SCHEMES[scheme_key]
    q = q0.copy()
    q_prev = None
    q_prev2 = None
    snapshots = {0: q0.copy()}

    for n in range(1, nsteps + 1):
        if scheme_key == 'leapfrog':
            q_new, q_prev_out = fn(q, c, dx, dt, q_prev=q_prev)
            q_prev = q_prev_out
            q = q_new
        elif scheme_key == 'lfam3':
            q_new, q_filtered = fn(q, c, dx, dt, q_prev=q_prev, q_prev2=q_prev2)
            q_prev2 = q_prev
            q_prev  = q_filtered if q_filtered is not None else q
            q = q_new
        else:
            q, _ = fn(q, c, dx, dt)

        if n in (nsteps // 4, nsteps // 2, 3 * nsteps // 4, nsteps):
            snapshots[n] = q.copy()

    return snapshots, label, color, ls

# ── error metrics ─────────────────────────────────────────────────────────────

def l2_error(q_num, q_ex):
    return np.sqrt(np.mean((q_num - q_ex) ** 2))

# ── plotting ──────────────────────────────────────────────────────────────────

def plot_results(results, x, c, dt, nsteps, sigma, x0, L, framerate, schemes_used):
    snap_steps = sorted({n for r in results for n in r[0]})
    a = c * dt / (x[1] - x[0])
    suptitle = f'CFL a = {a:.2f}  |  N = {len(x)}  |  c = {c}'

    outdir = f'out_{schemes_used[0]}_N{len(x)}_cfl{a:.2f}_s{sigma}_n{nsteps}'
    os.makedirs(outdir, exist_ok=True)

    for step in snap_steps:
        if step % framerate != 0:
            continue
        t = step * dt
        fig, ax = plt.subplots(figsize=(5, 4))
        q_exact = exact(x, x0, sigma, c, t, L)
        ax.plot(x, q_exact, 'k-', lw=.5, label='Exact', zorder=10)

        x_peak_exact = (x0 + c * t) % L
        ax.axvline(x_peak_exact, color='k', lw=0.8, ls='--', alpha=0.4, label='exact peak')

        for snapshots, label, color, ls in results:
            if step in snapshots:
                q_num = snapshots[step]
                ax.plot(x, q_num, color=color, ls=ls, lw=1.2, label=label)
                if np.max(np.abs(q_num)) < 10:
                    x_peak_num = x[np.argmax(q_num)]
                    ax.axvline(x_peak_num, color=color, lw=0.8, ls='--', alpha=0.6, label=f'{label} peak')

        ax.set_title(f't = {t:.2f}  (n={step})\n{suptitle}', fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('q')
        ax.set_ylim(-0.4, 1.3)
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.legend(loc='lower center', fontsize=9, framealpha=0.9, ncols=2)
        plt.tight_layout()
        fname = os.path.join(outdir, f'gaussian_advection_n{step:06d}.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {fname}')

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    steps_fine = list(range(1, nsteps + 1, max(1, nsteps // 100)))
    for snapshots, label, color, ls in results:
        errors = []
        for step in steps_fine:
            t = step * dt
            q_exact = exact(x, x0, sigma, c, t, L)
            q_num = snapshots.get(step)
            errors.append(l2_error(q_num, q_exact) if q_num is not None else np.nan)
        ax2.semilogy([s * dt for s in steps_fine], errors, color=color, ls=ls, lw=1.3, label=label)
    ax2.set_xlabel('time')
    ax2.set_ylabel('L2 error')
    ax2.set_title(f'L2 error vs time  |  {suptitle}')
    ax2.legend(fontsize=9)
    ax2.grid(True, which='both', lw=0.3, alpha=0.5)
    plt.tight_layout()
    fname2 = os.path.join(outdir, 'gaussian_advection_error.png')
    plt.savefig(fname2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved: {fname2}')
# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheme', default='all',
                        help='Scheme key or "all". Options: ' + ', '.join(SCHEMES.keys()))
    parser.add_argument('--cfl',    type=float, default=0.8)
    parser.add_argument('--N',      type=int,   default=128)
    parser.add_argument('--nsteps', type=int,   default=3000)
    parser.add_argument('--c',      type=float, default=1.0)
    parser.add_argument('--sigma',  type=float, default=0.1)
    parser.add_argument('--framerate',  type=float, default=30)

    args = parser.parse_args()

    L  = 2.0
    x  = np.linspace(0, L, args.N, endpoint=False)
    dx = x[1] - x[0]
    dt = args.cfl * dx / args.c
    x0 = 16 / 32
    q0 = gaussian_ic(x, x0, args.sigma)

    schemes_used = list(SCHEMES.keys()) if args.scheme == 'all' else [args.scheme]

    # need per-step snapshots for L2 error — store all steps
    results = []
    for key in schemes_used:
        fn, label, color, ls = SCHEMES[key]
        q = q0.copy()
        q_prev = None
        q_prev2 = None
        snapshots = {0: q0.copy()}

        for n in range(1, args.nsteps + 1):
            if key == 'leapfrog':
                q_new, q_prev_out = fn(q, args.c, dx, dt, q_prev=q_prev)
                q_prev = q_prev_out
                q = q_new
            elif key == 'lfam3':
                q_new, q_filtered = fn(q, args.c, dx, dt, q_prev=q_prev, q_prev2=q_prev2)
                q_prev2 = q_prev
                q_prev  = q_filtered if q_filtered is not None else q
                q = q_new
            else:
                q, _ = fn(q, args.c, dx, dt)
            snapshots[n] = q.copy()

        results.append((snapshots, label, color, ls))
        a = args.c * dt / dx
        print(f'{label:30s}  CFL={a:.3f}  L2_final={l2_error(q, exact(x, x0, args.sigma, args.c, args.nsteps*dt, L)):.4e}')

    plot_results(results, x, args.c, dt, args.nsteps, args.sigma, x0, L, args.framerate, schemes_used)

if __name__ == '__main__':
    main()
