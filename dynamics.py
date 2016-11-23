import numpy as np
import sympy as sym
from sympy.vector import CoordSysCartesian
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation


K = CoordSysCartesian('K')

N = 4 # NUMBER OF PIECES
Length = 1
mass_density = 1
kappa = 10
expand = False

# LENGHT DISTRIBUTION
np_lghs = np.linspace(0, 1, N+1)
# VARIABLES
t, l, l1, l2, lam, g = sym.symbols('t l l1 l2 lambda g', real = True)

# FUNCTTION DEFINING THE MASS DENSITY OF THE ROPE
density_func = lam

# FUNCTIONS FOR THE POSITIONS OF: MASS, CENTER OF MASS AND MOMENT OF INERTIA
ms_func = sym.integrate(density_func, (l, l1,l2))
cms_func = sym.integrate(density_func*l, (l, l1,l2))/ms_func
Is_func = sym.integrate(density_func*l**2, (l, l1,l2))

# CALCULATE THESE PROPERTIES FOR EACH PIECE
ms, cms, lghs, thts, Is = [], [], [], [], []
str_thts, str_dthts, str_ddthts = '','',''
for n in range(N):
    str_thts += 'tht%i ' %n
    str_dthts += 'dtht%i ' %n
    str_ddthts += 'ddtht%i ' %n

    ms.append(ms_func.subs([(l1, np_lghs[n]*l), (l2, np_lghs[n+1]*l)]))
    cms.append(cms_func.subs([(l1, np_lghs[n]*l), (l2, np_lghs[n+1]*l)]))
    Is_low = np_lghs[n]*l - cms[-1]
    Is_up = np_lghs[n+1]*l - cms[-1]
    Is.append(Is_func.subs([(l1, Is_low), (l2, Is_up)]))
    lghs.append((np_lghs[n+1] - np_lghs[n])*l)
    thts.append(sym.Function('f_tht%i' %n)(t))

symbols_thetas = sym.symbols(str_thts, reals = True)
symbols_dthetas = sym.symbols(str_dthts, reals = True)
symbols_ddthetas = sym.symbols(str_ddthts, reals = True)

# POSITIONS, VELOCITIES; ANGULAR VELOCITIES
rcms, vcms, vthts = [], [], []
for n in range(N):
    if 0 < n:
        rcm = lghs[0]*sym.sin(thts[0])*K.i - lghs[0]*sym.cos(thts[0])*K.j
        for i in range(1, n):
            rcm += lghs[i]*sym.sin(thts[i])*K.i - lghs[i]*sym.cos(thts[i])*K.j
        rcm += (cms[n]-np_lghs[n]*l)*sym.sin(thts[n])*K.i \
                - (cms[n]-np_lghs[n]*l)*sym.cos(thts[n])*K.j
    else:
        rcm = cms[0]*sym.sin(thts[0])*K.i - cms[0]*sym.cos(thts[0])*K.j
    rcms.append(rcm)
    vcms.append(sym.diff(rcm, t))
    vthts.append(sym.diff(thts[n], t))


# ENERGIES
KE = 1/2*ms[0]*sym.diff(rcms[0], t).dot(sym.diff(rcms[0], t))
IE = 1/2*Is[0]*sym.diff(thts[0], t)**2 # vthts[0]**2
UE = ms[0]*g*rcms[0].dot(K.j)
BE = 0
print('Energies init!')

for n in range(1, N):
    KE += 1/2*ms[n]*sym.diff(rcms[n], t).dot(sym.diff(rcms[n], t)) # kinetic e of pole n
    IE += 1/2*Is[n]*sym.diff(thts[n], t)**2 # rotation energy of pole n
    UE += ms[n]*g*rcms[n].dot(K.j) # Potential energy of pole n
    BE += 1/2*kappa*(thts[n] - thts[n-1])**2 # Bending energy of joint n
tKE = sym.simplify(KE + IE)
tUE = sym.simplify(UE + BE)

#print(KE)
#print(IE)
#print(UE)
#print()

# LAGRANGE
L = sym.simplify(tKE - tUE)
print('Lagrange made!')

# EULER LAGRANGE EQS
print('Create Euler-Lagrange set!')
EL = []
#ddthts = []
ddthts2 = []

for n in range(N):
    EL.append(sym.simplify(sym.diff(L, thts[n]) - sym.diff(sym.diff(L, sym.diff(thts[n], t)), t)))
    #EL.append(sym.diff(L, thts[n]) - sym.diff(sym.diff(L, sym.diff(thts[n], t)), t))
    #ddthts.append(sym.diff(vthts[n], t))
    ddthts2.append(sym.Derivative(symbols_dthetas[n], t))

# LAMBDIFY THE DDTHETAS
print('Start to lambdify!')
params_set = []
dderivs = []
if N != 1:
    for n in range(N):
        params_set.append((vthts[n], symbols_dthetas[n]))
        params_set.append((thts[n], symbols_thetas[n]))
else:
    params_set.append((thts[n], symbols_thetas))
    params_set.append((vthts[n], symbols_dthetas))


print('Solve the ddthetas!')
#print(params_set)
#ddtht_dic = sym.simplify(sym.solve(EL, ddthts))

for n in range(N):
    EL[n] = EL[n].subs(params_set)
    if expand:
        for i in range(len(symbols_thetas)):
            EL[n] = EL[n].series(symbols_thetas[i], n=2).removeO() #.subs(O(symbols_thetas[i]**2, 0)) #+ sym.Order(symbols_thetas[i])
            EL[n] = sym.simplify(EL[n].series(symbols_dthetas[i], n=2).removeO())

#ddtht_dic = sym.solve(EL, ddthts)
ddtht_dic = sym.solve(EL, ddthts2)
fys_params = [(l, Length), (lam, mass_density), (g, 9.81)]
for n in range(N):
    tmp_func = sym.utilities.lambdify((t, symbols_thetas, symbols_dthetas),
                ddtht_dic[ddthts2[n]].subs(fys_params))
    dderivs.append(tmp_func)

print('Lambdified everything!')

# SET OF oIDES and Solve
def dz(z, t):
    hh = int(len(z)/2)
    thetas = z[:hh]
    dthetas = z[hh:]
    ddthetas = np.zeros(N)

    for n in range(N):
        ddthetas[n] = dderivs[n](t, thetas, dthetas)
    return np.concatenate((dthetas, ddthetas))

t = np.linspace(0,10,5000)
init_thetas = np.ones(N)*.0
init_thetas[1] = np.pi/4
init_vthetas = np.ones(N)*.0
#init_vthetas[2] = 20
y0 = np.concatenate((init_thetas, init_vthetas))

print('Starting to propagate!')
sol = odeint(dz, y0, t)
print('Propagation ready!')

# Animate
print('Plot!')
xcoords = np.zeros((len(sol), N+1))
ycoords = np.zeros((len(sol), N+1))

for k in range(len(sol)):
    for n in range(N):
        xcoords[k, n+1] = xcoords[k,n] + lghs[n].subs(l, Length)*np.sin(sol[k,n])
        ycoords[k, n+1] = ycoords[k,n] - lghs[n].subs(l, Length)*np.cos(sol[k,n])


fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False,
        xlim=(-Length, Length), ylim=(-Length, Length))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
#line2, = ax.plot([], [], '-', lw=.5)

time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
dt = t[1] - t[0]

def init():
    line.set_data([], [])
    #line2.set_data([],[])
    time_text.set_text('')
    return line, time_text #line2,

def animate(i):
    thisx = xcoords[i] #[0, x1[i], x2[i]]
    thisy = ycoords[i] # [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    #line2.set_data(x2[:i], y2[:i])

    time_text.set_text(time_template%(i*dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(ycoords)),
    interval=dt*1000, blit=True, init_func=init)

plt.show()
Writer = animation.writers['ffmpeg']
writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1000)
ani.save('poles.mp4', writer=writer)
#ani.save('double_pendulum.mp4', fps=15)
