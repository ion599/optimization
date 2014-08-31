import BB
def solve(x0, f, nabla_f, stopping, record_every=500, proj=None, log=None, options=None, solve = BB.solve):
    z_l = x0
    N = 10
    for i in range(0, N+1):
        lamb = i/float(N)
        f_l = lambda x: f(x, lamb)
        nabla_f_l = lambda   x: nabla_f(x, lamb)
        z_l = solve(z_l, f_l, nabla_f_l, stopping,record_every=record_every,proj=proj, log=log,options=options)

    return z_l