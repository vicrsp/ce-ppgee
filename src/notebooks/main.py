# importar o otimizador
from pso import PSO
# criar instancia da função de teste quadrática
def quad_func(x): return x[0]**2 + x[1]**2


# Executar o algoritmo
pso_unimodal = PSO(quad_func, [-10, -10], [10, 10], max_feval=1000,
                   swarm_size=30, acceleration=[1.5, 1.5], constrition=0.5, topology='lbest')
pso_unimodal.run()
